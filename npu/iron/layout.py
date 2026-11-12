"""Buffer layouts and per-phase transfers of token.py, shared with the host.

Hidden rows: core (c, j) owns rows ((c * 8 + t) * 4 + j) * 8 + e. Every drain
of a column takes one 32-byte object from each of its four cores per step.
"""

H, I, VOCAB, D = 2048, 7680, 128000, 128
COLS, ROWS = 8, 4
REC = 4 * H + 4 * (H // 32) * 2   # 8704
BLOCK = 12 * H                    # [h f32 | o f32 | post-norm w bf16 | pre-norm w bf16]
LAYERS = 24
ATTN_LAYERS = [l for l in range(LAYERS) if l >= 2 and l % 2 == 0]
SWA_LAYERS = {4, 8, 12, 16, 20}
N_ATTN = len(ATTN_LAYERS)
CONV_LAYERS = [l for l in range(LAYERS) if l not in ATTN_LAYERS]
PHASES = 1 + 5 * len(CONV_LAYERS) + 5 * N_ATTN + 1
EMB_ROW = 4 * H                   # dequantized f32 row
QKV_ROWS = 3328                   # q 2048 | k 512 | v 512 | gate 16 | pad
QKV_BYTES = 2 * REC               # rows f32, then q-norm and k-norm bf16
CONVIN_RECS, OUT_RECS, QKV_RECS, GU_RECS, DOWN_RECS, HEAD_RECS = 48, 16, 26, 120, 60, 1000
STATE_CHUNKS, TAP_CHUNKS = 96, 112  # per conv core: 6 x 128 state f32, then 7 x 128 taps f32
STATE_COL = (REC // 32) * ROWS * 32  # per column [chunk][core][32]: state (drained) then taps (static)
BCX_COL = 2 * 3 * 128 * 4           # ConvIn output per column: [conv core][B | C | x][128 ch] f32
WINDOW = 256
CompileTimeCtx = int


def t(total, offset, sizes, strides):
    from aie.helpers.taplib import TensorAccessPattern  # the host imports this file without mlir-aie
    return TensorAccessPattern((total,), offset, sizes, strides)


class Layout:
    def __init__(self, ctx):
        self.ctx = ctx
        o = 0
        def take(n):
            nonlocal o
            at = o
            o += n
            return at
        self.blocks = take(49 * BLOCK)
        self.act = take(2 * REC)
        self.vec = take(64 + 136 * 128 + 64)
        self.att = take(17 * 1024 + 512)
        self.junk = take(COLS * 4 * 32)
        self.qkv = [take(QKV_BYTES) for _ in ATTN_LAYERS]
        self.bcx = take(COLS * BCX_COL + REC)  # the mix phase reads a whole record per conv core
        self.state = [take(COLS * STATE_COL) for _ in CONV_LAYERS]
        take(272 * 128)  # reads of the last column's state run past it
        self.rope = {kind: take((ctx + 1) * 512 + REC) for kind in ("global", "swa")}
        self.kv = [take(COLS * ROWS * ctx * 512 + REC) for _ in ATTN_LAYERS]
        self.logits = take(VOCAB * 4)
        self.emb = take(VOCAB * EMB_ROW + REC)  # in S: offsets must stay below 2^31 in both buffers
        self.s_bytes = o
        o = 0
        self.w = {}
        for l in range(LAYERS):
            if l in ATTN_LAYERS:
                self.w[(l, "qkv")] = take(2 * COLS * 2 * QKV_RECS * REC)
                self.w[(l, "attn_out")] = take(2 * COLS * 2 * OUT_RECS * REC)
            else:
                self.w[(l, "conv_in")] = take(2 * COLS * 2 * CONVIN_RECS * REC)
                self.w[(l, "conv_out")] = take(2 * COLS * 2 * OUT_RECS * REC)
            self.w[(l, "gate_up")] = take(2 * COLS * 2 * GU_RECS * REC)
            self.w[(l, "down")] = take(2 * COLS * 2 * DOWN_RECS * REC)
        self.w[(LAYERS, "head")] = take(2 * COLS * 2 * HEAD_RECS * REC)
        self.w_bytes = o
        assert self.s_bytes < 2 ** 31 and self.w_bytes < 2 ** 31

    def block(self, b):
        return self.blocks + b * BLOCK

    # ---- transfers

    def head_records(self, offset, records):
        """`records` records of the same bytes for both of a stream's cores (one copy: broadcast)."""
        # streams broadcast to both of their cores: one copy
        return t(self.s_bytes, offset, [1, records, REC // 512, 512], [0, REC, 512, 1])

    def weights(self, name, s, records):
        n = 2 * records * REC
        base = self.w[name]
        base += s * n
        return t(self.w_bytes, base, [1, 2 * records, REC // 512, 512], [0, REC, 512, 1])

    def hidden(self, offset, c):
        return t(self.s_bytes, offset + c * 1024, [1, 1, 32, 32], [0, 0, 32, 1])

    def contiguous(self, offset, length):
        assert length % 64 == 0
        rows = length // 64
        if rows < 1024:
            return t(self.s_bytes, offset, [1, 1, rows, 64], [0, 0, 64, 1])
        assert rows % 1000 == 0
        return t(self.s_bytes, offset, [1, rows // 1000, 1000, 64], [0, 64000, 64, 1])

    def phases(self, d):
        """Phases of one token; d holds the dispatch-time scalars."""
        yield Phase(heads={s: [("S", t(self.s_bytes, self.emb + d["tok"] * EMB_ROW,
                                       [1, 1, REC // 512, 512], [0, 0, 512, 1]))] for s in range(16)},
                    drains=lambda c: [self.hidden(self.block(0), c)])
        for l in range(LAYERS):
            op_in, ffn_in, nxt = self.block(2 * l), self.block(2 * l + 1), self.block(2 * l + 2)
            if l in CONV_LAYERS:
                i = CONV_LAYERS.index(l)
                st = self.state[i]
                # ConvIn: B, C and x rows on every core; column c writes its conv cores' channels.
                yield Phase(
                    heads={s: [("S", self.head_records(op_in, 3))] for s in range(16)},
                    stream={s: [("W", self.weights((l, "conv_in"), s, CONVIN_RECS))] for s in range(16)},
                    drains=lambda c, ffn_in=ffn_in: [
                        self.hidden(ffn_in, c),
                        t(self.s_bytes, self.bcx + c * BCX_COL, [1, 1, BCX_COL // 64, 64], [0, 0, 64, 1])])
                # ConvMix: the conv cores' channels, state and taps; no weights.
                yield Phase(
                    heads={s: [("S", t(self.s_bytes, self.bcx + (s // 2) * BCX_COL, [1, 2, REC // 512, 512], [0, BCX_COL // 2, 512, 1])),
                               ("S", t(self.s_bytes, st + (s // 2) * STATE_COL + 2 * 32, [1, 2, REC // 32, 32], [0, 32, ROWS * 32, 1]))]
                           for s in range(1, 16, 2)},
                    drains=lambda c, st=st: [
                        t(self.s_bytes, self.vec + c * 1024, [1, 1, 32, 32], [0, 0, 32, 1]),
                        t(self.s_bytes, st + c * STATE_COL, [1, 1, STATE_CHUNKS * ROWS * 32 // 64, 64], [0, 0, 64, 1])])
                yield Phase(
                    heads={s: [("S", t(self.s_bytes, self.vec + 64, [1, 1, 136, 64], [0, 0, 128, 1]))] for s in range(16)},
                    stream={s: [("W", self.weights((l, "conv_out"), s, OUT_RECS))] for s in range(16)},
                    drains=lambda c, ffn_in=ffn_in: [self.hidden(ffn_in + 4 * H, c)])
            else:
                i = ATTN_LAYERS.index(l)
                q, kv = self.qkv[i], self.kv[i]
                swa = l in SWA_LAYERS
                rope = self.rope["swa" if swa else "global"]
                first = d["swa_first"] if swa else 0
                recs = d["swa_recs"] if swa else d["global_recs"]
                yield Phase(
                    heads={s: [("S", self.head_records(op_in, 3))] for s in range(16)},
                    stream={s: [("W", self.weights((l, "qkv"), s, QKV_RECS))] for s in range(16)},
                    drains=lambda c, ffn_in=ffn_in, q=q: [
                        self.hidden(ffn_in, c),
                        t(self.s_bytes, q + c * 1664, [1, 1, 26, 64], [0, 0, 64, 1])])
                yield Phase(
                    heads={s: [("S", self.head_records(q, 2)),
                               ("S", t(self.s_bytes, rope + d["pos"] * 512, [1, 1, REC // 512, 512], [0, 0, 512, 1]))]
                           for s in range(0, 16, 2)},
                    stream={s: ([("S", t(self.s_bytes, kv + (s // 2) * ROWS * self.ctx * 512 + first * 512,
                                         [1, recs, REC // 512, 512], [0, 16 * 512, 512, 1]))] if swa else
                                [("S", t(self.s_bytes, kv + (s // 2) * ROWS * self.ctx * 512 + at * 16 * 512,
                                         [1, n, REC // 512, 512], [0, 16 * 512, 512, 1]))
                                 for at, n in ((0, d["global_a"]), (d["global_a"], d["global_b"]))])
                            for s in range(0, 16, 2)},
                    drains=lambda c, kv=kv: [
                        t(self.s_bytes, self.junk + c * 128, [1, 1, 2, 64], [0, 0, 64, 1]),
                        t(self.s_bytes, self.att + c * 1024, [1, 8, 4, 32], [0, 32, 256, 1]),
                        t(self.s_bytes, kv + c * ROWS * self.ctx * 512 + d["pos"] * 512,
                          [1, 16, 4, 32], [0, 32, self.ctx * 512, 1])])
                yield Phase(
                    heads={s: [("S", t(self.s_bytes, self.att, [1, 1, 17, 512], [0, 0, 1024, 1]))] for s in range(16)},
                    stream={s: [("W", self.weights((l, "attn_out"), s, OUT_RECS))] for s in range(16)},
                    drains=lambda c, ffn_in=ffn_in: [self.hidden(ffn_in + 4 * H, c)])
            yield Phase(
                heads={s: [("S", self.head_records(ffn_in, 3))] for s in range(16)},
                stream={s: [("W", self.weights((l, "gate_up"), s, GU_RECS))] for s in range(16)},
                drains=lambda c, nxt=nxt: [self.hidden(nxt, c),
                                           t(self.s_bytes, self.act + c * 1920, [1, 1, 30, 64], [0, 0, 64, 1])])
            yield Phase(
                heads={s: [("S", self.head_records(self.act, 2))] for s in range(16)},
                stream={s: [("W", self.weights((l, "down"), s, DOWN_RECS))] for s in range(16)},
                drains=lambda c, nxt=nxt: [self.hidden(nxt + 4 * H, c)])
        yield Phase(
            heads={s: [("S", self.head_records(self.block(48), 3))] for s in range(16)},
            stream={s: [("W", self.weights((LAYERS, "head"), s, HEAD_RECS))] for s in range(16)},
            drains=lambda c: [self.contiguous(self.logits + c * 64000, 64000)])


class Phase:
    def __init__(self, heads=None, stream=None, drains=None):
        self._heads, self._stream, self._drains = heads or {}, stream or {}, drains or (lambda c: [])

    def heads(self, s):
        return self._heads.get(s, [])

    def stream(self, s):
        return self._stream.get(s, [])

    def drains(self, c):
        return self._drains(c)
