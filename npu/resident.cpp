#include "resident.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <utility>
#include <vector>

#include "program.hpp"
#include "rowops.hpp"

namespace gufo::models::lfm3::npu {
namespace {

std::string S(std::uint64_t v) { return std::to_string(v); }
std::string Fill(std::string text, const std::vector<std::pair<std::string, std::string>>& values) {
  for (const auto& [key, value] : values)
    for (auto at = text.find(key); at != std::string::npos; at = text.find(key, at + value.size()))
      text.replace(at, key.size(), value);
  return text;
}
std::string F(double v) {
  char b[64];
  std::snprintf(b, sizeof b, "%.9e", v);
  return b;
}

// Waited DMA between a global view and a tile view.
std::string Dma(const std::string& tag, const std::string& from, const std::string& from_t, const std::string& to,
                const std::string& to_t, bool in) {
  return "      %" + tag + "_c = kernel.async.copy %" + from + " to %" + to +
         " {cache_scope = workgroup, cache_temporal = regular, direction = " +
         (in ? "global_to_workgroup" : "workgroup_to_global") + "} : " + from_t + " to " + to_t +
         " -> kernel.async.token\n      %" + tag + "_g = kernel.async.group %" + tag +
         "_c : kernel.async.token -> kernel.async.group\n      kernel.async.wait %" + tag +
         "_g {newer_groups = 0} : kernel.async.group\n";
}
// %<tag>_at = byte offset `elements` * esize (elements is an index SSA name or constant).
std::string At(const std::string& tag, std::uint64_t elements, std::uint32_t esize) {
  return "      %" + tag + "_at = index.constant " + S(elements * esize) + " : offset\n";
}

// RMS scale of n floats of view %v (view type vt) into %<tag>.
std::string RmsScale(const std::string& v, const std::string& vt, std::uint32_t off, std::uint32_t n,
                     const std::string& tag) {
  return Fill(R"(      %@T_lo = index.constant @OFF : index
      %@T_hi = index.constant @END : index
      %@T_sq = scf.for %@T_i = [%@T_lo to %@T_hi step %c16](%@T_a = %zero16 : vector<16xf32>) -> (vector<16xf32>) {
        %@T_v = vector.load %@V[%@T_i] : @VT -> vector<16xf32>
        %@T_p = func.call @fmul2(%@T_v, %@T_v) : (vector<16xf32>, vector<16xf32>) -> (vector<16xf32>)
        %@T_s = vector.addf %@T_a, %@T_p : vector<16xf32>
        scf.yield %@T_s : vector<16xf32>
      }
      %@T_t = vector.reduce<addf> %@T_sq, %fzero : vector<16xf32>, f32
      %@T_tv = vector.splat %@T_t : vector<16xf32>
      %@T_m = func.call @fmul2(%@T_tv, %inv_n) : (vector<16xf32>, vector<16xf32>) -> (vector<16xf32>)
      %@T_me = vector.addf %@T_m, %eps : vector<16xf32>
      %@T = func.call @rsqrt(%@T_me) : (vector<16xf32>) -> (vector<16xf32>)
)",
              {{"@T", tag}, {"@OFF", S(off)}, {"@END", S(off + n)}, {"@VT", vt}, {"@V", v}});
}

}  // namespace

std::string ResidentFfnProgram(std::uint32_t H, std::uint32_t I, float eps, const std::string& name, bool last) {
  // Per column c: worker core (c, 2) computes and never touches DRAM; the
  // loader core (c, 3) runs a fixed schedule of every input (weights into a
  // ring, everything else through one 16 KiB record) and the storer core
  // (c, 4) of every output, so each column needs one inbound and one
  // outbound shim path. Channels between them live in (c, 3), visible from
  // all three cores. After each barrier the worker publishes a request token
  // so the loader reads inputs written by other columns only once they are
  // final; the storer signals a flush after each stage's outputs.
  constexpr std::uint32_t C = kColumns;
  // Experiment only: emit the first N stages (bisecting the protocol).
  const char* stages_env = std::getenv("GUFO_LFM3_NPU_RESIDENT_STAGES");
  const std::uint32_t stages = stages_env ? std::uint32_t(std::atoi(stages_env)) : 5;
  const char* exp_env = std::getenv("GUFO_LFM3_NPU_RESIDENT_EXP");  // experiment only
  const int exp = exp_env ? std::atoi(exp_env) : 0;  // 1: one DMA per stage-3 record, 2: no stage-3 outputs
  const std::uint32_t gu_groups = 2 * I / 8, d_groups = H / 8;
  const std::uint32_t gu_tiles = H / 8 / C, d_tiles = I / 8 / C;
  const std::uint32_t gu_rec = gu_tiles * 72, d_rec = d_tiles * 72;
  // Batching: eight gate/up records per weight slot and eight output tiles
  // per output record (consecutive groups are contiguous in both).
  constexpr std::uint32_t kBatch = 8;
  // Down records per weight DMA (two consecutive groups share a slot when
  // they fit): larger transfers stream closer to the DRAM ceiling.
  const std::uint32_t dpair = 2 * d_tiles * 72 <= std::max(kBatch * gu_tiles * 72, d_tiles * 72) && (H / 8 / C) % 2 == 0 ? 2 : 1;
  const std::uint32_t slot = std::max(kBatch * gu_rec, d_rec);
  const std::uint32_t sw_groups = I / 8 / C, sw_piece = 30, sw_n = sw_piece * 8, sw_pn = 8 * sw_n;
  const std::uint32_t red_piece = 32, red_n = red_piece * 8;
  constexpr std::uint32_t kIn = 4096;  // floats in the input record
  if (H % 1024 || I % (8 * C * sw_piece) || gu_tiles % 8 || d_tiles % 8 || (H / 8) % red_piece || H > kIn ||
      2 * sw_pn > kIn || d_tiles * 32 > kIn || 8 * red_n > kIn || 2 * H > 6144)
    throw std::invalid_argument("unsupported resident FFN shape");
  const std::string rec_n = S(slot) + "xi8", gu_rec_t = "view<" + S(gu_rec) + "xi8>";
  const std::string in_t = "view<" + S(kIn) + "xf32>", out_t = "view<" + S(H) + "xf32>";
  const std::string chan_in = "channel<tile<" + S(kIn) + "xf32>>", chan_out = "channel<tile<" + S(H) + "xf32>>";
  const std::string tok_ch = "channel<tile<16xi32>>";
  auto acq = [&](const std::string& n, const std::string& ch, const std::string& ct, const std::string& vt) {
    return "      %" + n + "_r, %" + n + " = channel.acquire %" + ch + " : " + ct + " -> (read<" + ct.substr(8, ct.size() - 9) +
           ">, " + vt + ")\n";
  };
  auto rel = [&](const std::string& n, const std::string& ct) {
    return "      channel.release %" + n + "_r : read<" + ct.substr(8, ct.size() - 9) + ">\n";
  };
  auto rsv = [&](const std::string& n, const std::string& ch, const std::string& ct, const std::string& vt) {
    return "      %" + n + "_w, %" + n + " = channel.reserve %" + ch + " : " + ct + " -> (write<" + ct.substr(8, ct.size() - 9) +
           ">, " + vt + ")\n";
  };
  auto pub = [&](const std::string& n, const std::string& ct) {
    return "      channel.publish %" + n + "_w : write<" + ct.substr(8, ct.size() - 9) + ">\n";
  };
  // Typed subview of a record view (via F32 subview then bitcast).
  auto as = [&](const std::string& n, const std::string& rec, const std::string& rec_t, std::uint32_t off_f32,
                std::uint32_t floats, const std::string& type) {
    std::string r = "      %" + n + "_f = view.subview %" + rec + "[" + S(off_f32) + "] : " + rec_t + " -> view<" + S(floats) +
                    "xf32>\n";
    if (type == "view<" + S(floats) + "xf32>") return r + "      %" + n + " = view.bitcast %" + n + "_f : view<" + S(floats) + "xf32> -> " + type + "\n";
    return r + "      %" + n + " = view.bitcast %" + n + "_f : view<" + S(floats) + "xf32> -> " + type + "\n";
  };
  // Strided global view of g groups of 8 elements (stride 64), rows `parts` apart by `ps` elements.
  // dyn: an index SSA value of the first element, instead of the constant elem.
  auto groups_view = [&](const std::string& n, const std::string& buf, std::uint64_t elem, std::uint32_t esize,
                         std::uint32_t parts, std::uint32_t g, std::uint64_t ps, const std::string& e,
                         const std::string& dyn = "") {
    std::string r = dyn.empty() ? At(n, elem, esize)
                                : "      %" + n + "_es = index.constant " + S(esize) + " : index\n      %" + n +
                                      "_eb = index.mul " + dyn + ", %" + n + "_es : index\n      %" + n + "_at = index.cast %" +
                                      n + "_eb : index to offset\n";
    if (parts == 1) {
      r += "      %" + n + "_d0 = buffer.view %" + buf + "[%" + n + "_at] : buffer -> view<" + S(g) + "x64x" + e + ">\n" +
           "      %" + n + "_d1 = view.refine %" + n + "_d0 : view<" + S(g) + "x64x" + e + "> -> view<" + S(g) + "x64x" + e +
           ", #encoding.layout.strided<strides=[64, 1]>>\n" + "      %" + n + " = view.subview %" + n + "_d1[0, 0] : view<" +
           S(g) + "x64x" + e + ", #encoding.layout.strided<strides=[64, 1]>> -> view<" + S(g) + "x8x" + e +
           ", #encoding.layout.strided<strides=[64, 1]>>\n";
    } else {
      const std::string st = "#encoding.layout.strided<strides=[" + S(ps) + ", 64, 1]>";
      r += "      %" + n + "_d0 = buffer.view %" + buf + "[%" + n + "_at] : buffer -> view<" + S(parts) + "x" + S(g) + "x64x" +
           e + ">\n" + "      %" + n + "_d1 = view.refine %" + n + "_d0 : view<" + S(parts) + "x" + S(g) + "x64x" + e +
           "> -> view<" + S(parts) + "x" + S(g) + "x64x" + e + ", " + st + ">\n" + "      %" + n + " = view.subview %" + n +
           "_d1[0, 0, 0] : view<" + S(parts) + "x" + S(g) + "x64x" + e + ", " + st + "> -> view<" + S(parts) + "x" + S(g) +
           "x8x" + e + ", " + st + ">\n";
    }
    return r;
  };
  auto groups_t = [&](std::uint32_t parts, std::uint32_t g, std::uint64_t ps, const std::string& e) {
    return parts == 1 ? "view<" + S(g) + "x8x" + e + ", #encoding.layout.strided<strides=[64, 1]>>"
                      : "view<" + S(parts) + "x" + S(g) + "x8x" + e + ", #encoding.layout.strided<strides=[" + S(ps) +
                            ", 64, 1]>>";
  };
  auto dense_t = [&](std::uint32_t parts, std::uint32_t g, const std::string& e) {
    return parts == 1 ? "view<" + S(g) + "x8x" + e + ">" : "view<" + S(parts) + "x" + S(g) + "x8x" + e + ">";
  };
  auto flat = [&](const std::string& n, const std::string& buf, std::uint64_t elem, std::uint32_t esize,
                  const std::string& type) {
    return At(n, elem, esize) + "      %" + n + " = buffer.view %" + buf + "[%" + n + "_at] : buffer -> " + type + "\n";
  };

  std::string t = "aie2p.target<array> @npu\n";
  t += kMathLibrary;
  // Gate/up records sit eight to a weight slot: the dot reads the slot at a byte offset.
  t += Fill(ProjectionDot("dot_gu", gu_tiles, gu_tiles),
            {{"%w: view<" + S(gu_rec) + "xi8>, %tile0: index", "%w: view<" + S(slot) + "xi8>, %wbase: index, %tile0: index"},
             {"%w_at = index.mul %chunk, %cbytes : index", "%w_at0 = index.mul %chunk, %cbytes : index\n    %w_at = index.add %w_at0, %wbase : index"},
             {"view<" + S(gu_rec) + "xi8>", "view<" + S(slot) + "xi8>"}});
  t += ProjectionDot("dot_d", d_tiles, d_tiles);
  std::string top = "\npipeline.def<kernel> public retain target(@npu) @" + name +
                    "() run(%a0: buffer, %a1: buffer, %a2: buffer, %a3: buffer, %a4: buffer, %a5: buffer, %a6: buffer, "
                    "%a7: buffer, %a8: buffer, %a9: buffer, %a10: buffer" + std::string(last ? ", %a11: buffer" : "") + ") {\n";
  top += R"(  %bx, %by, %bwo, %bx2, %bxb, %bwgu, %bygu, %bxf, %bwd, %byd, %bwf = buffer.assume.alignment %a0, %a1, %a2, %a3, %a4, %a5, %a6, %a7, %a8, %a9, %a10 {minimum_alignment = 64} : buffer, buffer, buffer, buffer, buffer, buffer, buffer, buffer, buffer, buffer, buffer
@ZBIND  %origin = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c2 = index.constant 2 : index
  %c8 = index.constant 8 : index
  %c16 = index.constant 16 : index
  %tokb = index.constant 64 : offset
)";
  top = Fill(top, {{"@ZBIND", last ? "  %bz = buffer.assume.alignment %a11 {minimum_alignment = 64} : buffer\n" : ""}});
  auto bind = [&](const std::string& n, const std::string& pool, std::uint64_t bytes, std::uint32_t slots,
                  const std::string& rec) {
    return "  %" + n + "_s = buffer.alloca<workgroup> pool(%" + pool + ") align(64) %" + n + "_b : buffer\n" + "  %" + n +
           "_v = buffer.view %" + n + "_s[%origin] : buffer -> view<" + S(slots) + "x" + rec + ">\n" + "  %" + n +
           " = channel.bind<fifo> %" + n + "_v, %c" + S(slots) + " : view<" + S(slots) + "x" + rec +
           ">, index -> channel<tile<" + rec + ">>\n";
  };
  auto bytes = [&](const std::string& n, std::uint64_t b) { return "  %" + n + "_b = index.constant " + S(b) + " : offset\n"; };
  for (std::uint32_t c = 0; c < C; ++c) {
    const std::string cs = S(c);
    // Rows: loader 2, worker 3, storer 4, weight stream 5.
    top += "  %im" + cs + " = pipeline.memory<workgroup>[" + cs + ", 2] : pool\n  %wm" + cs +
           " = pipeline.memory<workgroup>[" + cs + ", 3] : pool\n  %lm" + cs + " = pipeline.memory<workgroup>[" + cs +
           ", 4] : pool\n";
    top += bytes("tok" + cs, 64) + bind("tok" + cs, "wm" + cs, 64, 1, "16xi32");
    top += bytes("go" + cs, 64) + bind("go" + cs, "wm" + cs, 64, 1, "16xi32");
    top += bytes("sc" + cs, 6144 * 4) + bind("sc" + cs, "wm" + cs, 6144 * 4, 1, "6144xf32");
    top += bytes("wr" + cs, 2 * slot) + bind("wr" + cs, "lm" + cs, slot, 2, rec_n);
    top += bytes("inb" + cs, kIn * 4) + bind("inb" + cs, "im" + cs, kIn * 4, 1, S(kIn) + "xf32");
    top += bytes("outb" + cs, H * 4) + bind("outb" + cs, "lm" + cs, H * 4, 1, S(H) + "xf32");
    top += bytes("yr" + cs, 2 * kBatch * 256) + bind("yr" + cs, "lm" + cs, kBatch * 256, 2, S(kBatch * 64) + "xf32");
    top += bytes("fl" + cs, 64) + bind("fl" + cs, "lm" + cs, 64, 1, "16xi32");
    top += bytes("rq" + cs, 64) + bind("rq" + cs, "im" + cs, 64, 1, "16xi32");
  }
  // Go multicast: column 7's token fans out to every column (DMA only, beside its worker).
  {
    std::string fans, types, res, cps;
    for (std::uint32_t c = 0; c < C; ++c) {
      fans += (c ? ", %f" : "%f") + S(c);
      types += c ? ", read<tile<16xi32>>" : "read<tile<16xi32>>";
      res += "      %gw" + S(c) + ", %gq" + S(c) + " = channel.reserve %go" + S(c) + " : " + tok_ch +
             " -> (write<tile<16xi32>>, view<16xi32>)\n";
      cps += "      channel.copy %f" + S(c) + " -> %gw" + S(c) + " : read<tile<16xi32>>, write<tile<16xi32>>\n";
    }
    if (stages > 1) top += "  %nbar = index.constant " + S(stages - 1) + " : index\n  pipeline.strand workers([" + S(C - 1) +
           ", 3], [1, 1], [1, 1]) {\n    scf.for %r = [%c0 to %nbar step %c1] {\n      %dr, %dp = channel.acquire %tok" +
           S(C - 1) + " : " + tok_ch + " -> (read<tile<16xi32>>, view<16xi32>)\n      " + fans +
           " = channel.fanout %dr : read<tile<16xi32>> -> (" + types + ")\n" + res + cps + "    }\n  }\n";
  }
  const std::string tok_t = "view<16xi32>";
  for (std::uint32_t c = 0; c < C; ++c) {
    const std::string cs = S(c);
    const bool first = c == 0;
    // Column 0 runs the first norm stage, column 1 the last (one core's
    // program memory does not hold both); x2 passes through DRAM.
    const bool n5 = c == 1;
    auto token_wait = [&](const std::string& n, const std::string& ch) { return acq(n, ch, tok_ch, tok_t) + rel(n, tok_ch); };
    auto token_post = [&](const std::string& n, const std::string& ch) { return rsv(n, ch, tok_ch, tok_t) + pub(n, tok_ch); };
    auto barrier = [&](const std::string& k) {
      std::string b = first ? "" : token_wait("bi" + k, "tok" + S(c - 1));
      return b + token_post("bo" + k, "tok" + cs) + token_wait("bg" + k, "go" + cs) + token_post("rqt" + k, "rq" + cs);
    };
    const std::string inc = chan_in, outc = chan_out, wrc = "channel<tile<" + rec_n + ">>", yrc = "channel<tile<8x8xf32>>";
    // ---------------- Loader (c, 3).
    std::string L = "  pipeline.strand workers([" + cs + ", 2], [1, 1], [1, 1]) {\n";
    auto load = [&](const std::string& n, const std::string& src, const std::string& src_t, const std::string& dst,
                    const std::string& dst_t) { return Dma(n, src, src_t, dst, dst_t, true); };
    if (first) {
      // Stage 1 inputs: x, y_out row 0, post norm weight, ffn norm weight.
      L += rsv("l1x", "inb" + cs, inc, in_t) + as("l1xd", "l1x", in_t, 0, H, out_t) + flat("l1xs", "bx", 0, 4, out_t) +
           load("l1x", "l1xs", out_t, "l1xd", out_t) + pub("l1x", inc);
      L += rsv("l1y", "inb" + cs, inc, in_t) + as("l1yd", "l1y", in_t, 0, H, dense_t(1, H / 8, "f32")) +
           groups_view("l1ys", "by", 0, 4, 1, H / 8, 0, "f32") +
           load("l1y", "l1ys", groups_t(1, H / 8, 0, "f32"), "l1yd", dense_t(1, H / 8, "f32")) + pub("l1y", inc);
      for (int k = 0; k < 2; ++k) {
        const std::string n = "l1w" + S(k);
        L += rsv(n, "inb" + cs, inc, in_t) + as(n + "d", n, in_t, 0, H, out_t) + flat(n + "s", "bwo", std::uint64_t(k) * H, 4, out_t) +
             load(n, n + "s", out_t, n + "d", out_t) + pub(n, inc);
      }
    }
    L += "@@S2@@";
    // Stage 2: x slice (after barrier 1), then gate/up weight records.
    const std::string gx_t = "view<" + S(gu_tiles * 64) + "xbf16>";
    L += token_wait("lq1", "rq" + cs) + rsv("l2x", "inb" + cs, inc, in_t) + as("l2xd", "l2x", in_t, 0, gu_tiles * 32, gx_t) +
         flat("l2xs", "bxb", std::uint64_t(c) * gu_tiles * 64, 2, gx_t) + load("l2x", "l2xs", gx_t, "l2xd", gx_t) + pub("l2x", inc);
    L += "@@S3@@";
    // Stage 3: per piece, gate and up partials (after barrier 2).
    L += token_wait("lq2", "rq" + cs);
    {
      const std::string n = "l3", pt = groups_t(8, sw_piece, std::uint64_t(gu_groups) * 64, "f32"),
                        dt = dense_t(8, sw_piece, "f32");
      L += "      %l3_np = index.constant " + S(sw_groups / sw_piece) + " : index\n      %l3_ps = index.constant " +
           S(sw_piece * 64) + " : index\n      %l3_g0 = index.constant " + S(std::uint64_t(c) * sw_groups * 64) +
           " : index\n      %l3_u0 = index.constant " + S((std::uint64_t(c) * sw_groups + I / 8) * 64) +
           " : index\n      scf.for %l3_pc = [%c0 to %l3_np step %c1] {\n      %l3_po = index.mul %l3_pc, %l3_ps : index\n" +
           "      %l3_ge = index.add %l3_g0, %l3_po : index\n      %l3_ue = index.add %l3_u0, %l3_po : index\n";
      L += rsv(n, "inb" + cs, inc, in_t) + as(n + "gd", n, in_t, 0, sw_pn, dt) + as(n + "ud", n, in_t, sw_pn, sw_pn, dt) +
           groups_view(n + "gs", "bygu", 0, 4, 8, sw_piece, std::uint64_t(gu_groups) * 64, "f32", "%l3_ge") +
           groups_view(n + "us", "bygu", 0, 4, 8, sw_piece, std::uint64_t(gu_groups) * 64, "f32", "%l3_ue") +
           load(n + "g", n + "gs", pt, n + "gd", dt) + load(n + "u", n + "us", pt, n + "ud", dt) + pub(n, inc) + "      }\n";
    }
    L += "@@S4@@";
    // Stage 4: x slice (after barrier 3), then down weight records.
    const std::string dx_t = "view<" + S(d_tiles * 64) + "xbf16>";
    L += token_wait("lq3", "rq" + cs) + rsv("l4x", "inb" + cs, inc, in_t) + as("l4xd", "l4x", in_t, 0, d_tiles * 32, dx_t) +
         flat("l4xs", "bxf", std::uint64_t(c) * d_tiles * 64, 2, dx_t) + load("l4x", "l4xs", dx_t, "l4xd", dx_t) + pub("l4x", inc);
    L += "@@S5@@";
    // Stage 5 (column 1): x2, down partials in pieces, then the norm weights (after barrier 4).
    L += token_wait("lq4", "rq" + cs);
    if (n5) {
      L += rsv("l5x", "inb" + cs, inc, in_t) + as("l5xd", "l5x", in_t, 0, H, out_t) + flat("l5xs", "bx2", 0, 4, out_t) +
           load("l5x", "l5xs", out_t, "l5xd", out_t) + pub("l5x", inc);
      {
        const std::string n = "l5", pt = groups_t(8, red_piece, std::uint64_t(d_groups) * 64, "f32"),
                          dt = dense_t(8, red_piece, "f32");
        L += "      %l5_np = index.constant " + S(H / 8 / red_piece) + " : index\n      %l5_ps = index.constant " +
             S(red_piece * 64) + " : index\n      scf.for %l5_pc = [%c0 to %l5_np step %c1] {\n" +
             "      %l5_pe = index.mul %l5_pc, %l5_ps : index\n";
        L += rsv(n, "inb" + cs, inc, in_t) + as(n + "d", n, in_t, 0, 8 * red_n, dt) +
             groups_view(n + "s", "byd", 0, 4, 8, red_piece, std::uint64_t(d_groups) * 64, "f32", "%l5_pe") +
             load(n, n + "s", pt, n + "d", dt) + pub(n, inc) + "      }\n";
      }
      for (int k = 0; k < 2; ++k) {
        const std::string n = "l5w" + S(k);
        L += rsv(n, "inb" + cs, inc, in_t) + as(n + "d", n, in_t, 0, H, out_t) + flat(n + "s", "bwf", std::uint64_t(k) * H, 4, out_t) +
             load(n, n + "s", out_t, n + "d", out_t) + pub(n, inc);
      }
    }
    L += "  }\n";
    // ---------------- Weight stream (DMA only, beside the worker): gate/up
    // records, then down records; the ring's credits pace it.
    std::string Wi = "  pipeline.strand workers([" + cs + ", 5], [1, 1], [1, 1]) {\n";
    const std::string gub_t = "view<" + S(kBatch * gu_rec) + "xi8>";
    Wi += Fill(R"(      %ngu = index.constant @NGU : index
      %rgu = index.constant @RGU : index
      %fgu = index.constant @FGU : index
      scf.for %g = [%c0 to %ngu step %c1] {
        %gb0 = index.mul %g, %rgu : index
        %gb = index.add %gb0, %fgu : index
        %gat = index.cast %gb : index to offset
        %src = buffer.view %bwgu[%gat] : buffer -> @GUBT
        %ww, %wp = channel.reserve %wr@C : channel<tile<@RECN>> -> (write<tile<@RECN>>, view<@RECN>)
        %wsub = view.subview %wp[0] : view<@RECN> -> @GUBT
@DMA        channel.publish %ww : write<tile<@RECN>>
      }
)",
              {{"@C", cs}, {"@NGU", S(gu_groups / kBatch)}, {"@RGU", S(kBatch * gu_rec)},
               {"@FGU", S(std::uint64_t(c) * gu_groups * gu_rec)}, {"@GUBT", gub_t}, {"@RECN", rec_n},
               {"@DMA", Fill(Dma("wgu", "src", gub_t, "wsub", gub_t, true), {{"\n      ", "\n        "}, {"      %wgu_c", "        %wgu_c"}})}});
    Wi += Fill(R"(      %nd = index.constant @ND : index
      %rd = index.constant @RD : index
      %fd = index.constant @FD : index
      scf.for %g = [%c0 to %nd step %c1] {
        %gi = index.add %g, %fd : index
        %gb = index.mul %gi, %rd : index
        %gat = index.cast %gb : index to offset
        %src = buffer.view %bwd[%gat] : buffer -> view<@DRN>
        %ww, %wp = channel.reserve %wr@C : channel<tile<@RECN>> -> (write<tile<@RECN>>, view<@RECN>)
        %wsub = view.subview %wp[0] : view<@RECN> -> view<@DRN>
@DMA        channel.publish %ww : write<tile<@RECN>>
      }
)",
              {{"@C", cs}, {"@ND", S(d_groups / dpair)}, {"@RD", S(std::uint64_t(dpair) * d_rec)}, {"@FD", S(std::uint64_t(c) * d_groups / dpair)}, {"@RECN", rec_n},
               {"@DRN", S(dpair * d_rec) + "xi8"},
               {"@DMA", Fill(Dma("wd", "src", "view<" + S(dpair * d_rec) + "xi8>", "wsub", "view<" + S(dpair * d_rec) + "xi8>", true), {{"\n      ", "\n        "}, {"      %wd_c", "        %wd_c"}})}});
    Wi += "  }\n";
    // ---------------- Storer (c, 4).
    std::string St = "  pipeline.strand workers([" + cs + ", 4], [1, 1], [1, 1]) {\n";
    auto store = [&](const std::string& n, const std::string& src, const std::string& src_t, const std::string& dst,
                     const std::string& dst_t) { return Dma(n, src, src_t, dst, dst_t, false); };
    auto store_xb = [&](const std::string& n) {  // outb as BF16 row 0 into Xb
      return acq(n, "outb" + cs, outc, out_t) + as(n + "s", n, out_t, 0, H / 2, dense_t(1, H / 8, "bf16")) +
             groups_view(n + "d", "bxb", 0, 2, 1, H / 8, 0, "bf16") +
             store(n, n + "s", dense_t(1, H / 8, "bf16"), n + "d", groups_t(1, H / 8, 0, "bf16")) + rel(n, outc);
    };
    auto tiles = [&](const std::string& n, std::uint32_t count, std::uint64_t first_tile, const std::string& buf) {
      const std::string yt = "view<" + S(kBatch * 64) + "xf32>", yc = "channel<tile<" + S(kBatch * 64) + "xf32>>";
      return Fill(R"(      %@N_n = index.constant @CNT : index
      %@N_f = index.constant @F : index
      %@N_tb = index.constant @TB : index
      scf.for %g = [%c0 to %@N_n step %c1] {
        %yrr, %yp = channel.acquire %yr@C : @YC -> (read<@YCT>, @YT)
        %gb = index.mul %g, %@N_tb : index
        %ob = index.add %gb, %@N_f : index
        %oat = index.cast %ob : index to offset
        %dst = buffer.view %@BUF[%oat] : buffer -> @YT
@DMA        channel.release %yrr : read<@YCT>
      }
)",
                  {{"@N", n}, {"@CNT", S(count / kBatch)}, {"@F", S(first_tile * 256)}, {"@TB", S(kBatch * 256)},
                   {"@C", cs}, {"@BUF", buf}, {"@YCT", yc.substr(8, yc.size() - 9)}, {"@YC", yc}, {"@YT", yt},
                   {"@DMA", Fill(Dma(n + "y", "yp", yt, "dst", yt, false), {{"\n      ", "\n        "}, {"      %" + n + "y_c", "        %" + n + "y_c"}})}});
    };
    if (first)
      St += acq("s1x", "outb" + cs, outc, out_t) + flat("s1xd", "bx2", 0, 4, out_t) + store("s1x", "s1x", out_t, "s1xd", out_t) +
            rel("s1x", outc) + store_xb("s1") + token_post("f1", "fl" + cs);
    St += "@@S2@@";
    St += tiles("s2", gu_groups, std::uint64_t(c) * gu_groups, "bygu") + token_post("f2", "fl" + cs);
    St += "@@S3@@";
    {
      const std::string n = "s3";
      St += "      %s3_np = index.constant " + S(sw_groups / sw_piece) + " : index\n      %s3_ps = index.constant " +
            S(sw_piece * 64) + " : index\n      %s3_g0 = index.constant " + S(std::uint64_t(c) * sw_groups * 64) +
            " : index\n      scf.for %s3_pc = [%c0 to %s3_np step %c1] {\n      %s3_po = index.mul %s3_pc, %s3_ps : index\n" +
            "      %s3_ge = index.add %s3_g0, %s3_po : index\n";
      St += acq(n, "outb" + cs, outc, out_t) + as(n + "s", n, out_t, 0, sw_n / 2, dense_t(1, sw_piece, "bf16")) +
            groups_view(n + "d", "bxf", 0, 2, 1, sw_piece, 0, "bf16", "%s3_ge") +
            store(n, n + "s", dense_t(1, sw_piece, "bf16"), n + "d", groups_t(1, sw_piece, 0, "bf16")) + rel(n, outc) + "      }\n";
    }
    St += token_post("f3", "fl" + cs);
    St += "@@S4@@";
    St += tiles("s4", d_groups, std::uint64_t(c) * d_groups, "byd") + token_post("f4", "fl" + cs);
    St += "@@S5@@";
    if (n5) {
      St += acq("s5x", "outb" + cs, outc, out_t) + flat("s5xd", "bx", 0, 4, out_t) + store("s5x", "s5x", out_t, "s5xd", out_t) +
            rel("s5x", outc);
      if (last) {  // the normed F32 row instead of the residual and packed row
        St = St.substr(0, St.rfind("      %s5x_r, %s5x"));
        St += acq("s5z", "outb" + cs, outc, out_t) + flat("s5zd", "bz", 0, 4, out_t) + store("s5z", "s5z", out_t, "s5zd", out_t) +
              rel("s5z", outc) + token_post("f5", "fl" + cs);
      } else {
        St += store_xb("s5b") + token_post("f5", "fl" + cs);
      }
    }
    St += "  }\n";
    // ---------------- Worker (c, 2).
    std::string W = "  pipeline.strand workers([" + cs + ", 3], [1, 1], [1, 1]) {\n";
    W += Fill(R"(    %zero16 = vector.splatf 0.0 : vector<16xf32>
    %fzero = scalar.constant 0.0 : f32
    %inv_n = vector.splatf @INV : vector<16xf32>
    %eps = vector.splatf @EPS : vector<16xf32>
    %scw, %sc = channel.reserve %sc@C : channel<tile<6144xf32>> -> (write<tile<6144xf32>>, view<6144xf32>)
)",
              {{"@C", cs}, {"@INV", F(1.0 / H)}, {"@EPS", F(eps)}});
    const std::string sc_t = "view<6144xf32>";
    auto copy_in = [&](const std::string& n, std::uint32_t sc_off) {  // inb[0:H] -> sc[sc_off:]
      return acq(n, "inb" + cs, inc, in_t) + Fill(R"(      %@N_o = index.constant @O : index
      %@N_h = index.constant @H : index
      scf.for %@N_i = [%c0 to %@N_h step %c16] {
        %@N_v = vector.load %@N[%@N_i] : @INT -> vector<16xf32>
        %@N_j = index.add %@N_i, %@N_o : index
        vector.store %@N_v, %sc[%@N_j] : vector<16xf32>, view<6144xf32>
      }
)",
                                                          {{"@N", n}, {"@O", S(sc_off)}, {"@H", S(H)}, {"@INT", in_t}}) +
             rel(n, inc);
    };
    // x[xo] += post_norm(sc[yo]) * w, w from the input record n.
    auto residual = [&](const std::string& n, std::uint32_t xo, std::uint32_t yo) {
      return acq(n, "inb" + cs, inc, in_t) + RmsScale("sc", sc_t, yo, H, n + "_s") + Fill(R"(      %@N_xo = index.constant @XO : index
      %@N_yo = index.constant @YO : index
      %@N_h = index.constant @H : index
      scf.for %@N_i = [%c0 to %@N_h step %c16] {
        %@N_xi = index.add %@N_i, %@N_xo : index
        %@N_yi = index.add %@N_i, %@N_yo : index
        %@N_x = vector.load %sc[%@N_xi] : view<6144xf32> -> vector<16xf32>
        %@N_y = vector.load %sc[%@N_yi] : view<6144xf32> -> vector<16xf32>
        %@N_w = vector.load %@N[%@N_i] : @INT -> vector<16xf32>
        %@N_ys = func.call @fmul2(%@N_y, %@N_s) : (vector<16xf32>, vector<16xf32>) -> (vector<16xf32>)
        %@N_yw = func.call @fmul2(%@N_ys, %@N_w) : (vector<16xf32>, vector<16xf32>) -> (vector<16xf32>)
        %@N_r = vector.addf %@N_x, %@N_yw : vector<16xf32>
        vector.store %@N_r, %sc[%@N_xi] : vector<16xf32>, view<6144xf32>
      }
)",
                                                                                     {{"@N", n}, {"@XO", S(xo)}, {"@YO", S(yo)}, {"@H", S(H)}, {"@INT", in_t}}) +
             rel(n, inc);
    };
    // outb (BF16) = pack(norm(sc[xo]) * w), w from the input record n.
    auto pack = [&](const std::string& n, std::uint32_t xo) {
      return acq(n, "inb" + cs, inc, in_t) + RmsScale("sc", sc_t, xo, H, n + "_s") + rsv(n + "o", "outb" + cs, outc, out_t) +
             as(n + "_ob", n + "o", out_t, 0, H / 2, "view<" + S(H) + "xbf16>") + Fill(R"(      %@N_xo = index.constant @XO : index
      %@N_h = index.constant @H : index
      scf.for %@N_i = [%c0 to %@N_h step %c16] {
        %@N_xi = index.add %@N_i, %@N_xo : index
        %@N_x = vector.load %sc[%@N_xi] : view<6144xf32> -> vector<16xf32>
        %@N_w = vector.load %@N[%@N_i] : @INT -> vector<16xf32>
        %@N_a = func.call @fmul2(%@N_x, %@N_s) : (vector<16xf32>, vector<16xf32>) -> (vector<16xf32>)
        %@N_b = func.call @fmul2(%@N_a, %@N_w) : (vector<16xf32>, vector<16xf32>) -> (vector<16xf32>)
        %@N_c = vector.fptrunc %@N_b : vector<16xf32> to vector<16xbf16>
        vector.store %@N_c, %@N_ob[%@N_i] : vector<16xbf16>, view<@Hxbf16>
      }
)",
                                                                                                   {{"@N", n}, {"@XO", S(xo)}, {"@H", S(H)}, {"@INT", in_t}}) +
             pub(n + "o", outc) + rel(n, inc);
    };
    auto flush = [&](const std::string& k) { return token_wait("wf" + k, "fl" + cs); };
    auto emit_x = [&](const std::string& n) {  // sc[0:H] -> outb (F32)
      return rsv(n, "outb" + cs, outc, out_t) + Fill(R"(      %@N_h = index.constant @H : index
      scf.for %@N_i = [%c0 to %@N_h step %c16] {
        %@N_v = vector.load %sc[%@N_i] : view<6144xf32> -> vector<16xf32>
        vector.store %@N_v, %@N[%@N_i] : vector<16xf32>, @OUTT
      }
)",
                                                     {{"@N", n}, {"@H", S(H)}, {"@OUTT", out_t}}) +
             pub(n, outc);
    };
    if (first) W += copy_in("w1x", 0) + copy_in("w1y", H) + residual("w1r", 0, H) + emit_x("w1e") + pack("w1p", 0) + flush("1");
    W += "@@S2@@" + barrier("1");
    // Stage 2: gate/up partials.
    W += acq("w2x", "inb" + cs, inc, in_t) + as("w2xv", "w2x", in_t, 0, gu_tiles * 32, gx_t) +
         Fill(R"(      %ngu = index.constant @NGU : index
      %kb = index.constant @KB : index
      %rb = index.constant @RB : index
      %c64t = index.constant 64 : index
      scf.for %g = [%c0 to %ngu step %c1] {
        %wrr, %wv = channel.acquire %wr@C : channel<tile<@RECN>> -> (read<tile<@RECN>>, view<@RECN>)
        %yw, %yv = channel.reserve %yr@C : @YC -> (write<@YCT>, @YT)
        scf.for %k = [%c0 to %kb step %c1] {
          %wo = index.mul %k, %rb : index
          %z = vector.splatf 0.0 : vector<64xf32>
          %init = vector.fragment<init> %z shape [%c8, %c8] : vector<64xf32>
          %acc = func.call @dot_gu(%w2xv, %wv, %wo, %c0, %init) : (@GXT, view<@RECN>, index, index, vector<64xf32>) -> (vector<64xf32>)
          %yo = index.mul %k, %c64t : index
          %ys = view.subview %yv[%yo] : @YT -> view<64xf32>
          %yt = view.bitcast %ys : view<64xf32> -> view<8x8xf32>
          vector.fragment.store<result> %acc, %yt[0, 0] shape [%c8, %c8] : vector<64xf32>, view<8x8xf32>
        }
        channel.publish %yw : write<@YCT>
        channel.release %wrr : read<tile<@RECN>>
      }
)",
              {{"@C", cs}, {"@NGU", S(gu_groups / kBatch)}, {"@KB", S(kBatch)}, {"@RB", S(gu_rec)}, {"@RECN", rec_n},
               {"@GXT", gx_t}, {"@YCT", "tile<" + S(kBatch * 64) + "xf32>"}, {"@YC", "channel<tile<" + S(kBatch * 64) + "xf32>>"},
               {"@YT", "view<" + S(kBatch * 64) + "xf32>"}}) +
         rel("w2x", inc) + flush("2") + "@@S3@@" + barrier("2");
    // Stage 3: SwiGLU pieces.
    {
      const std::string n = "w3";
      W += "      %w3_np = index.constant " + S(sw_groups / sw_piece) + " : index\n      scf.for %w3_pc = [%c0 to %w3_np step %c1] {\n";
      W += acq(n, "inb" + cs, inc, in_t) + rsv(n + "o", "outb" + cs, outc, out_t) +
           as(n + "_ob", n + "o", out_t, 0, sw_n / 2, "view<" + S(sw_n) + "xbf16>") + Fill(R"(      %@N_n = index.constant @SN : index
      %@N_un = index.constant @PN : index
      scf.for %@N_i = [%c0 to %@N_n step %c16] {
        %@N_gv = scf.for %@N_p = [%c0 to %c8 step %c1](%@N_a = %zero16 : vector<16xf32>) -> (vector<16xf32>) {
          %@N_po = index.mul %@N_p, %@N_n : index
          %@N_pi = index.add %@N_po, %@N_i : index
          %@N_v = vector.load %@N[%@N_pi] : @INT -> vector<16xf32>
          %@N_s = vector.addf %@N_a, %@N_v : vector<16xf32>
          scf.yield %@N_s : vector<16xf32>
        }
        %@N_uv = scf.for %@N_p = [%c0 to %c8 step %c1](%@N_a = %zero16 : vector<16xf32>) -> (vector<16xf32>) {
          %@N_po = index.mul %@N_p, %@N_n : index
          %@N_pi0 = index.add %@N_po, %@N_i : index
          %@N_pi = index.add %@N_pi0, %@N_un : index
          %@N_v = vector.load %@N[%@N_pi] : @INT -> vector<16xf32>
          %@N_s = vector.addf %@N_a, %@N_v : vector<16xf32>
          scf.yield %@N_s : vector<16xf32>
        }
        %@N_sg = func.call @sigmoid(%@N_gv) : (vector<16xf32>) -> (vector<16xf32>)
        %@N_si = func.call @fmul2(%@N_gv, %@N_sg) : (vector<16xf32>, vector<16xf32>) -> (vector<16xf32>)
        %@N_o = func.call @fmul2(%@N_si, %@N_uv) : (vector<16xf32>, vector<16xf32>) -> (vector<16xf32>)
        %@N_ob2 = vector.fptrunc %@N_o : vector<16xf32> to vector<16xbf16>
        vector.store %@N_ob2, %@N_ob[%@N_i] : vector<16xbf16>, view<@SNxbf16>
      }
)",
                                                                                          {{"@N", n}, {"@SN", S(sw_n)}, {"@PN", S(sw_pn)}, {"@INT", in_t}}) +
           pub(n + "o", outc) + rel(n, inc);
      W += "      }\n";
    }
    W += flush("3") + "@@S4@@" + barrier("3");
    // Stage 4: down partials.
    W += acq("w4x", "inb" + cs, inc, in_t) + as("w4xv", "w4x", in_t, 0, d_tiles * 32, dx_t) +
         Fill(R"(      %nd = index.constant @ND : index
      %kb4 = index.constant @KB : index
      %c64t4 = index.constant 64 : index
      scf.for %g = [%c0 to %nd step %c1] {
        %yw, %yv = channel.reserve %yr@C : @YC -> (write<@YCT>, @YT)
        %kbp = index.constant @KBP : index
        scf.for %kp = [%c0 to %kbp step %c1] {
          %wrr, %wv = channel.acquire %wr@C : channel<tile<@RECN>> -> (read<tile<@RECN>>, view<@RECN>)
#PAIR#          channel.release %wrr : read<tile<@RECN>>
        }
        channel.publish %yw : write<@YCT>
      }
)",
              {{"#PAIR#", [&] {
                  std::string t;
                  for (std::uint32_t j = 0; j < dpair; ++j)
                    t += Fill(R"(          %wd@J = view.subview %wv[@OFF] : view<@RECN> -> view<@DRN>
          %z@J = vector.splatf 0.0 : vector<64xf32>
          %init@J = vector.fragment<init> %z@J shape [%c8, %c8] : vector<64xf32>
          %acc@J = func.call @dot_d(%w4xv, %wd@J, %c0, %init@J) : (@DXT, view<@DRN>, index, vector<64xf32>) -> (vector<64xf32>)
          %kk@J0 = index.mul %kp, %dp2 : index
          %kj@J = index.constant @J : index
          %k@J = index.add %kk@J0, %kj@J : index
          %yo@J = index.mul %k@J, %c64t4 : index
          %ys@J = view.subview %yv[%yo@J] : @YT -> view<64xf32>
          %yt@J = view.bitcast %ys@J : view<64xf32> -> view<8x8xf32>
          vector.fragment.store<result> %acc@J, %yt@J[0, 0] shape [%c8, %c8] : vector<64xf32>, view<8x8xf32>
)", {{"@J", S(j)}, {"@OFF", S(std::uint64_t(j) * d_rec)}});
                  return "          %dp2 = index.constant " + S(dpair) + " : index\n" + t;
                }()},
               {"@KBP", S(kBatch / dpair)},
               {"@C", cs}, {"@ND", S(d_groups / kBatch)}, {"@KB", S(kBatch)}, {"@RECN", rec_n}, {"@DRN", S(d_rec) + "xi8"},
               {"@DXT", dx_t}, {"@YCT", "tile<" + S(kBatch * 64) + "xf32>"}, {"@YC", "channel<tile<" + S(kBatch * 64) + "xf32>>"},
               {"@YT", "view<" + S(kBatch * 64) + "xf32>"}}) +
         rel("w4x", inc) + flush("4") + "@@S5@@" + barrier("4");
    // Stage 5 (column 1): x2 into sc[0:H], o = sum of partials into sc[H:2H];
    // x = x2 + post(o) w; xb = pack(norm(x) w').
    if (n5) {
      W += copy_in("w5c", 0);
      {
        const std::string n = "w5";
        W += "      %w5_np = index.constant " + S(H / 8 / red_piece) + " : index\n      %w5_rn = index.constant " + S(red_n) +
             " : index\n      %w5_hb = index.constant " + S(H) + " : index\n      scf.for %w5_pc = [%c0 to %w5_np step %c1] {\n" +
             "      %w5_pb = index.mul %w5_pc, %w5_rn : index\n      %w5_ob = index.add %w5_pb, %w5_hb : index\n";
        W += acq(n, "inb" + cs, inc, in_t) + Fill(R"(      %@N_n = index.constant @RN : index
      scf.for %@N_i = [%c0 to %@N_n step %c16] {
        %@N_s = scf.for %@N_p = [%c0 to %c8 step %c1](%@N_a = %zero16 : vector<16xf32>) -> (vector<16xf32>) {
          %@N_po = index.mul %@N_p, %@N_n : index
          %@N_pi = index.add %@N_po, %@N_i : index
          %@N_v = vector.load %@N[%@N_pi] : @INT -> vector<16xf32>
          %@N_r = vector.addf %@N_a, %@N_v : vector<16xf32>
          scf.yield %@N_r : vector<16xf32>
        }
        %@N_oi = index.add %@N_i, %@N_ob : index
        vector.store %@N_s, %sc[%@N_oi] : vector<16xf32>, view<6144xf32>
      }
)",
                                                            {{"@N", n}, {"@RN", S(red_n)}, {"@INT", in_t}}) +
             rel(n, inc) + "      }\n";
      }
      W += residual("w5r", 0, H);
      // The new residual, then its packed norm.
      if (last) {  // outb = norm(sc[0:H]) * w (F32)
        W += acq("w5z", "inb" + cs, inc, in_t) + RmsScale("sc", sc_t, 0, H, "w5z_s") + rsv("w5zo", "outb" + cs, outc, out_t) +
             Fill(R"(      %w5z_h = index.constant @H : index
      scf.for %w5z_i = [%c0 to %w5z_h step %c16] {
        %w5z_x = vector.load %sc[%w5z_i] : view<6144xf32> -> vector<16xf32>
        %w5z_w = vector.load %w5z[%w5z_i] : @INT -> vector<16xf32>
        %w5z_a = func.call @fmul2(%w5z_x, %w5z_s) : (vector<16xf32>, vector<16xf32>) -> (vector<16xf32>)
        %w5z_b = func.call @fmul2(%w5z_a, %w5z_w) : (vector<16xf32>, vector<16xf32>) -> (vector<16xf32>)
        vector.store %w5z_b, %w5zo[%w5z_i] : vector<16xf32>, @OUTT
      }
)",
                  {{"@H", S(H)}, {"@INT", in_t}, {"@OUTT", out_t}}) +
             pub("w5zo", outc) + rel("w5z", inc) + flush("5");
      } else {
        W += emit_x("w5x") + pack("w5p", 0) + flush("5");
      }
    }
    W += "    channel.publish %scw : write<tile<6144xf32>>\n  }\n";
    auto cut = [&](std::string text, const std::string& tail) {
      if (stages < 5) {
        const auto at = text.find("@@S" + S(stages + 1) + "@@");
        if (at != std::string::npos) text = text.substr(0, at) + tail;
      }
      for (int k = 2; k <= 5; ++k) text = Fill(text, {{"@@S" + S(k) + "@@", ""}});
      return text;
    };
    top += (stages >= 2 ? Wi : std::string()) + cut(L, "  }\n") + cut(St, "  }\n") + cut(W, "    channel.publish %scw : write<tile<6144xf32>>\n  }\n");
  }
  top += "}\n";
  return t + top;
}

}  // namespace gufo::models::lfm3::npu
