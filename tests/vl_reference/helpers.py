"""Helpers of the LFM3-VL reference, copied from the Gufo lfm3 branch
(500df164): tools/d1/reference.py (GGUF to teacher mapping),
tests/functional/lfm3_logits.py (logit comparison) and
tests/functional/lfm3_vl.py (PNG fixtures)."""
import math
import struct
import zlib

import numpy as np
import torch
from gguf.quants import dequantize


def load_weights(module, reader, mapper, device):
    tensors = {t.name: t for t in reader.tensors}

    def get(name):
        t = tensors[name]
        return dequantize(t.data, t.tensor_type).astype(np.float32).reshape(tuple(int(d) for d in t.shape[::-1]))

    state = {}
    for name, parameter in module.state_dict().items():
        mapped = mapper(name)
        if isinstance(mapped, tuple):
            first, second = (get(n).reshape(1024, 3, 16, 16) for n in mapped)
            values = np.stack([first, second], axis=2)
        else:
            values = get(mapped)
        state[name] = torch.from_numpy(values.copy().reshape(parameter.shape))
    module.load_state_dict(state, strict=True, assign=True)
    return module.eval().to(device)


def text_name(name):
    if name in ("model.embed_tokens.weight", "lm_head.weight"):
        return "token_embd.weight"
    if name == "model.embedding_norm.weight":
        return "output_norm.weight"
    m = re.fullmatch(r"model.layers.(\d+).(.+)", name)
    if not m:
        raise ValueError(name)
    fields = {"operator_norm": "attn_norm", "ffn_norm": "ffn_norm",
              "post_operator_norm": "post_attention_norm", "post_ffn_norm": "post_ffw_norm",
              "feed_forward.w1": "ffn_gate", "feed_forward.w3": "ffn_up", "feed_forward.w2": "ffn_down",
              "conv.conv": "shortconv.conv", "conv.in_proj": "shortconv.in_proj", "conv.out_proj": "shortconv.out_proj",
              "self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v",
              "self_attn.o_proj": "attn_output", "self_attn.q_norm": "attn_q_norm", "self_attn.k_norm": "attn_k_norm",
              "self_attn.gate_proj": "attn_gate"}
    return f"blk.{m[1]}.{fields[m[2].removesuffix('.weight')]}.weight"


def vision_name(name):
    if name == "patchifier.weight":
        return ("v.patch_embd.weight", "v.patch_embd.weight.1")
    if name == "embedding_norm.weight":
        return "v.embedding_norm.weight"
    m = re.fullmatch(r"layers.(\d+).(.+)", name)
    if not m:
        raise ValueError(name)
    fields = {"operator_norm": "ln1", "ffn_norm": "ln2",
              "feed_forward.w1": "ffn_gate", "feed_forward.w3": "ffn_up", "feed_forward.w2": "ffn_down",
              "conv.in_proj": "conv2d_in_proj", "conv.spatial_conv": "conv2d_spatial", "conv.out_proj": "conv2d_out_proj",
              "self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v",
              "self_attn.out_proj": "attn_out", "self_attn.q_layernorm": "attn_q_norm", "self_attn.k_layernorm": "attn_k_norm"}
    return f"v.blk.{m[1]}.{fields[m[2].removesuffix('.weight')]}.weight"


def read(path):
    data = path.read_bytes()
    return struct.unpack("<" + "f" * (len(data) // 4), data)


def compare(actual, teacher):
    assert len(actual) == len(teacher) == 128000
    assert all(math.isfinite(x) for x in actual + teacher), "nonfinite logits"
    def probabilities(values):
        maximum = max(values)
        p = [math.exp(x - maximum) for x in values]
        total = sum(p)
        return [x / total for x in p]
    p, q = probabilities(teacher), probabilities(actual)
    return {"max_abs": max(abs(a - b) for a, b in zip(actual, teacher)),
            "rms": math.sqrt(sum((a - b) ** 2 for a, b in zip(actual, teacher)) / len(actual)),
            "kl_teacher_candidate": sum(a * math.log(a / b) for a, b in zip(p, q) if a and b),
            "total_variation": 0.5 * sum(abs(a - b) for a, b in zip(p, q)),
            "top1_equal": max(range(len(actual)), key=actual.__getitem__) ==
                          max(range(len(teacher)), key=teacher.__getitem__)}


def png(color, width=256, height=256):
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    rows = (b'\0' + bytes(color) * width) * height
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(rows)) + chunk(b'IEND', b''))
