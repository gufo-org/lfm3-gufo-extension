"""Phase plan of the prefill xclbin, shared by the design (model.py) and the host.

Buffers (bf16 elements unless noted):
  a: token records, M x REC: [xn | x | o | pad]
  b: GEMM weights, bfp16ebs8 blocks (9 bytes per 8 values), in phase order
  c: scratch: SwiGLU output, per-conv-layer [bx | C] rows (6 state rows first),
     conv output rows, and A blocks (128 x 512) of small per-phase constants

Every phase is a dict with "kind" in gemm, norm, conv, qkr, attn. GEMM phases
carry an A source and a C destination:
  A ("rec", [row0])          xn of the records (from row row0)
  A ("half", off, stride)    columns 0..63 of every 128 of a scratch matrix
  A ("plain", off, stride)   a scratch matrix
  A ("heads", off)           attention output, head-major [16][M][128]
  C ("o",)                   the records' o slot
  C ("plain", off, stride)   a scratch matrix
  C ("halves", off, stride)  columns 64..127 of every 128, two blocks per tile
"""

H, I, REC = 2048, 7680, 8192
M_TILE, N_TILE, K_TILE, MTK = 128, 128, 64, 512
CONV_ROW, Y_ROW, QKV_ROW = 4096, 2560, 4608
PIECE = 16384                      # one core's C object
LAYERS = 24
ATTN_LAYERS = [l for l in range(LAYERS) if l >= 2 and l % 2 == 0]
SWA_LAYERS = {4, 8, 12, 16, 20}
WINDOW = 256
VOCAB, HEAD_ROWS = 128000, 512
NARROW_VOCAB = 131072  # the LM head of a narrow plan: whole groups of 4 tiles per column


def narrow_tiles(N):
    """Tiles per column and K step of a narrow GEMM (rows < 512): the largest T <= 4
    dividing the column's tiles; core rows >= T idle."""
    per_col = N // N_TILE // 8
    return next(t for t in (4, 3, 2, 1) if per_col % t == 0)


def lfm3_kinds(n=LAYERS):
    return ["swa" if l in SWA_LAYERS else "attn" if l in ATTN_LAYERS else "conv" for l in range(n)]
STATE = 6
ABLOCK = M_TILE * MTK


class Plan:
    def __init__(self, kinds, M=1024, head_row0=None):
        self.M, self.kinds = M, list(kinds)
        self.c_off = 0
        self.b_off = 0
        self.sw = self._c(M * 15360)
        self.ys = self._c(M * Y_ROW + 4096)
        self.cs, self.pieces = {}, {}
        self.qs = self._c(M * QKV_ROW + 512)
        self.oh = 16 * M * 128               # head-major attention output
        self.ao = self._c(4 * self.oh) + 3 * self.oh  # three junk copies first
        self.ablocks = []          # (c offset, what) for the host to fill
        self.phases = [self._norm(("embed",))]
        if self.kinds == ["swtest"]:  # precision test: embedding norm, then layer 0's gate/up GEMM
            self.phases.append(self._gemm(H, 2 * I, "swiglu", ("rec",), ("plain", self.sw, 2 * I), (0, "gate_up")))
            self.kinds = []
        for l, kind in enumerate(self.kinds):
            if kind in ("attn", "swa"):
                piece = self.pieces[l] = self._c(M // 16 * 4 * PIECE)
                swa = int(kind == "swa")
                self.phases += [
                    self._gemm(H, 4096, "plain", ("rec",), ("plain", self.qs, QKV_ROW), (l, "qkv")),
                    dict(kind="qkr", w=self._ablock((l, "qkn")), qs=self.qs, piece=piece, mode=swa),
                    dict(kind="attn", piece=piece, ao=self.ao, mode=swa),
                    self._gemm(H, H, "plain", ("heads", self.ao), ("o",), (l, "attn_out")),
                ] + self._ffn(l)
                continue
            assert kind == "conv", kind
            cs = self.cs[l] = self._c((STATE + M) * CONV_ROW + 512)
            rows = cs + STATE * CONV_ROW
            self.phases += [
                self._gemm(H, 4096, "bx", ("rec",), ("plain", rows, CONV_ROW), (l, "conv_bx")),
                self._gemm(H, H, "plain", ("rec",), ("halves", rows, CONV_ROW), (l, "conv_c")),
                dict(kind="conv", cs=cs, ys=self.ys, taps=[self._ablock((l, "taps", q)) for q in range(4)]),
                self._gemm(H, H, "plain", ("plain", self.ys, Y_ROW), ("o",), (l, "conv_out")),
            ] + self._ffn(l)
        self.junk = self._c(3 * M_TILE * N_TILE)  # idle core rows of narrow GEMMs drain here
        if head_row0 is not None and M < HEAD_ROWS:
            # narrow plan: the head over all M rows
            self.head_rows, self.head_n, self.head_row0 = M, NARROW_VOCAB, 0
            self.logits = self._c(M * NARROW_VOCAB)
            self.phases.append(self._gemm(H, NARROW_VOCAB, "plain", ("rec", 0), ("plain", self.logits, NARROW_VOCAB), ("head",),
                                          rows=M))
        elif head_row0 is not None:
            self.head_rows, self.head_n = HEAD_ROWS, VOCAB
            # LM head over the 512 rows that hold the last token: output_norm is folded into
            # the weights, logits land as bf16 [512][VOCAB]. The core program does not depend
            # on head_row0, so its variants share one xclbin.
            self.logits = self._c(HEAD_ROWS * VOCAB)
            self.head_row0 = head_row0
            self.phases.append(self._gemm(H, VOCAB, "plain", ("rec", head_row0), ("plain", self.logits, VOCAB), ("head",),
                                          rows=HEAD_ROWS))
        self.b_size = self.b_off
        self.c_size = self.c_off

    def _ffn(self, l):
        return [
            self._norm((l, "post")),
            self._gemm(H, 2 * I, "swiglu", ("rec",), ("plain", self.sw, 2 * I), (l, "gate_up")),
            self._gemm(I, H, "plain", ("half", self.sw, 2 * I), ("o",), (l, "down")),
            self._norm((l, "ffn_post")),
        ]

    def codes(self):
        """One code per phase for the core interpreter (see prefill_next)."""
        out = []
        for ph in self.phases:
            if ph["kind"] == "gemm":
                if ph["rows"] < HEAD_ROWS:  # narrow: groups of T tiles per column
                    t_ = narrow_tiles(ph["N"])
                    tiles = ph["N"] // N_TILE // 8 // t_
                else:
                    t_, tiles = 0, (ph["rows"] // M_TILE) * (ph["N"] // N_TILE) // 32
                assert tiles < 128
                mode = {"plain": 0, "swiglu": 1, "bx": 2}[ph["mode"]]
                out.append(0 | mode << 3 | tiles << 5 | (ph["K"] // K_TILE) << 12 | t_ << 20)
            else:
                out.append({"norm": 1, "conv": 2, "qkr": 3, "attn": 4}[ph["kind"]] | ph.get("mode", 0) << 3)
        return out

    def _c(self, n):
        at = self.c_off
        self.c_off += (n + 63) // 64 * 64
        return at

    def _ablock(self, what):
        at = self._c(ABLOCK)
        self.ablocks.append((at, what))
        return at

    def _norm(self, what):
        return dict(kind="norm", w=self._ablock(what))

    def _gemm(self, K, N, mode, a, c, what, rows=None):
        at = self.b_off
        self.b_off += K * N // 8
        return dict(kind="gemm", K=K, N=N, mode=mode, a=a, c=c, b=at, what=what, rows=rows or self.M)


def ablock_put(vec):
    """A 128 x 512 A block whose first L1 objects hold vec linearly, 2048 values each
    (object i: rows 32 i .. 32 i + 31 of columns 0..63, blocked [m/r][k/s][r][s])."""
    import numpy as np
    vec = np.asarray(vec, np.uint16).reshape(-1)
    assert vec.size <= 4 * 2048
    z = np.zeros((M_TILE, MTK), np.uint16)
    p = np.arange(vec.size)
    obj, q = p // 2048, p % 2048
    z[obj * 32 + (q // 512) * 8 + (q // 8) % 8, ((q // 64) % 8) * 8 + q % 8] = vec
    return z.reshape(-1)
