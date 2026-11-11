"""Writes the device images liblfm3_xrt.so loads: the decode state and weights for
lfm3_token.py's xclbin, the prefill weights and scratch for prefill/model.py's
xclbin, and the offsets the library needs to hand a prefill over to decode.

Usage: bundle.py <gguf> <out dir> <decode xclbin> <dispatch.so> <prefill xclbin> <prefill insts>
"""
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "prefill"))
import layout as L  # noqa: E402
import plan as P  # noqa: E402


def main():
    gguf, out, dxclbin, dso, pxclbin, pinsts = sys.argv[1:7]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ctx = 4096

    import host as PH  # prefill host (imports pyxrt lazily only in Runner)
    md = PH.Model(gguf)
    plan = P.Plan(P.lfm3_kinds())
    E, Cs = PH.pack(plan, md)
    E.tofile(out / "prefill_weights.bin")
    Cs.tofile(out / "prefill_scratch.bin")
    del E, Cs
    md.drop()

    import token_host as TH
    lay, S, W, *_ = TH.pack(gguf, ctx)
    S.tofile(out / "decode_state.bin")
    W.tofile(out / "decode_weights.bin")
    del S, W

    for src, name in ((dxclbin, "decode.xclbin"), (dso, "decode_dispatch.so"), (pxclbin, "prefill.xclbin"),
                      (pinsts, "prefill.insts")):
        shutil.copyfile(src, out / name)

    meta = {
        "vocab": L.VOCAB, "hidden": L.H, "ctx": ctx, "window": L.WINDOW,
        "decode": {
            "s_bytes": lay.s_bytes, "w_bytes": lay.w_bytes, "logits": lay.logits, "emb": lay.emb,
            "emb_row": L.EMB_ROW, "kv": lay.kv, "state": lay.state, "state_col": L.STATE_COL,
            "attn_layers": L.ATTN_LAYERS, "conv_layers": L.CONV_LAYERS, "swa_layers": sorted(L.SWA_LAYERS),
        },
        "prefill": {
            "M": plan.M, "rec": P.REC, "c_size": plan.c_size,
            "conv_row": P.CONV_ROW, "state_rows": P.STATE, "piece": P.PIECE,
            "cs": {str(l): v for l, v in plan.cs.items()}, "pieces": {str(l): v for l, v in plan.pieces.items()},
        },
    }
    (out / "bundle.json").write_text(json.dumps(meta, indent=1))
    print("bundle written to", out, {f.name: os.path.getsize(f) for f in out.iterdir()})


if __name__ == "__main__":
    main()
