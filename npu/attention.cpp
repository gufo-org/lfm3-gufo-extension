#include "attention.hpp"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <utility>
#include <vector>

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
std::string Compose(const std::string& name, const std::string& args, const std::string& names,
                    const std::string& types) {
  std::string t = "\npipeline.def<kernel> public retain target(@npu) @" + name + "() run(" + args + ") {\n";
  for (int c = 0; c < 8; ++c) t += "  %k" + S(c) + " = index.constant " + S(c) + " : index\n";
  for (int c = 0; c < 8; ++c)
    t += "  pipeline.compose @" + name + "_col[%k" + S(c) + "](" + names + ") : [index](" + types + ")\n";
  return t + "}\n";
}
// A channel slot reserved for the whole strand, used as tile scratch.
std::string Scratch(const std::string& name, const std::string& shape, std::uint64_t bytes) {
  return Fill(R"(  %@N_size = index.constant @B : offset
  %@N_storage = buffer.alloca<workgroup> pool(%memory) align(64) %@N_size : buffer
  %@N_slots = buffer.view %@N_storage[%origin] : buffer -> view<1x@S>
  %@N_ch = channel.bind<fifo> %@N_slots, %c1 : view<1x@S>, index -> channel<tile<@S>>
)",
              {{"@N", name}, {"@S", shape}, {"@B", S(bytes)}});
}
std::string Reserve(const std::string& name, const std::string& shape) {
  return "    %" + name + "_w, %" + name + " = channel.reserve %" + name + "_ch : channel<tile<" + shape +
         ">> -> (write<tile<" + shape + ">>, view<" + shape + ">)\n";
}
std::string Publish(const std::string& name, const std::string& shape) {
  return "    channel.publish %" + name + "_w : write<tile<" + shape + ">>\n";
}
// DMA between a global view expression and a tile view, waited.
std::string Copy(const std::string& tag, const std::string& from, const std::string& from_type,
                 const std::string& to, const std::string& to_type, bool in) {
  return "      %" + tag + "_cp = kernel.async.copy %" + from + " to %" + to +
         " {cache_scope = workgroup, cache_temporal = regular, direction = " +
         (in ? "global_to_workgroup" : "workgroup_to_global") + "} : " + from_type + " to " + to_type +
         " -> kernel.async.token\n      %" + tag + "_g = kernel.async.group %" + tag +
         "_cp : kernel.async.token -> kernel.async.group\n      kernel.async.wait %" + tag +
         "_g {newer_groups = 0} : kernel.async.group\n";
}
// Byte offset value %<tag>_at = elements * esize.
std::string Offset(const std::string& tag, const std::string& elements, std::uint32_t esize) {
  return "      %" + tag + "_bytes = index.mul " + elements + ", %c" + S(esize) + " : index\n      %" + tag +
         "_at = index.cast %" + tag + "_bytes : index to offset\n";
}
// Loads row meta field i (bounded by mask) as index %<name>.
std::string Meta(const std::string& name, int field, std::uint32_t mask) {
  return "      %" + name + "_raw = view.load %meta_t[%c" + S(field) + "] : view<16xi32> -> i32\n" +
         "      %" + name + "_mask = scalar.constant " + S(mask) + " : i32\n" +
         "      %" + name + "_b = scalar.andi %" + name + "_raw, %" + name + "_mask : i32\n" +
         "      %" + name + " = index.cast %" + name + "_b : i32 to index\n";
}
const char* kConsts = R"(  %origin = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c2 = index.constant 2 : index
  %c3 = index.constant 3 : index
  %c4 = index.constant 4 : index
  %c5 = index.constant 5 : index
  %c8 = index.constant 8 : index
  %c16 = index.constant 16 : index
  %c32 = index.constant 32 : index
  %c64 = index.constant 64 : index
)";

// Per-head RMSNorm and rotary of `heads` heads of x (view<NxF32>) at head
// offset base, in place.
std::string NormRope(const AttentionShape& s, const std::string& view, const std::string& heads,
                     const std::string& base, const std::string& weight, const std::string& tag) {
  const std::uint32_t D = s.dim, half = D / 2;
  return Fill(R"(      scf.for %@T_h = [%c0 to %@T_heads step %c1] {
        %@T_h0 = index.mul %@T_h, %dim : index
        %@T_hi = index.cast %@T_h : index to i32
        %@T_hv = vector.splat %@T_hi : vector<16xi32>
        %@T_qh = vector.splati @QH : vector<16xi32>
        %@T_isk = vector.cmpi sge, %@T_hv, %@T_qh : vector<16xi32> -> vector<16xi1>
        %@T_at = index.add %@T_h0, @BASE : index
        %@T_sq = scf.for %@T_i = [%c0 to %dim step %c16](%@T_acc = %vzero : vector<16xf32>) -> (vector<16xf32>) {
          %@T_j = index.add %@T_at, %@T_i : index
          %@T_v = vector.load %@V[%@T_j] : view<@VTxf32> -> vector<16xf32>
          %@T_p = func.call @fmul(%@T_v, %@T_v) : (vector<16xf32>, vector<16xf32>) -> (vector<16xf32>)
          %@T_s = vector.addf %@T_acc, %@T_p : vector<16xf32>
          scf.yield %@T_s : vector<16xf32>
        }
        %@T_total = vector.reduce<addf> %@T_sq, %fzero : vector<16xf32>, f32
        %@T_tv = vector.splat %@T_total : vector<16xf32>
        %@T_mean = func.call @fmul(%@T_tv, %inv_dim) : (vector<16xf32>, vector<16xf32>) -> (vector<16xf32>)
        %@T_me = vector.addf %@T_mean, %eps : vector<16xf32>
        %@T_scale = func.call @rsqrt(%@T_me) : (vector<16xf32>) -> (vector<16xf32>)
        scf.for %@T_i = [%c0 to %half step %c16] {
          %@T_ja = index.add %@T_at, %@T_i : index
          %@T_jb = index.add %@T_ja, %half : index
          %@T_ib = index.add %@T_i, %half : index
          %@T_a0 = vector.load %@V[%@T_ja] : view<@VTxf32> -> vector<16xf32>
          %@T_b0 = vector.load %@V[%@T_jb] : view<@VTxf32> -> vector<16xf32>
          %@T_qa = vector.load %qnw[%@T_i] : view<@Dxf32> -> vector<16xf32>
          %@T_qb = vector.load %qnw[%@T_ib] : view<@Dxf32> -> vector<16xf32>
          %@T_ka = vector.load %knw[%@T_i] : view<@Dxf32> -> vector<16xf32>
          %@T_kb = vector.load %knw[%@T_ib] : view<@Dxf32> -> vector<16xf32>
          %@T_wa = vector.select %@T_isk, %@T_ka, %@T_qa : vector<16xf32>
          %@T_wb = vector.select %@T_isk, %@T_kb, %@T_qb : vector<16xf32>
          %@T_a1 = func.call @fmul2(%@T_a0, %@T_scale) : (vector<16xf32>, vector<16xf32>) -> (vector<16xf32>)
          %@T_a = func.call @fmul2(%@T_a1, %@T_wa) : (vector<16xf32>, vector<16xf32>) -> (vector<16xf32>)
          %@T_b1 = func.call @fmul2(%@T_b0, %@T_scale) : (vector<16xf32>, vector<16xf32>) -> (vector<16xf32>)
          %@T_b = func.call @fmul2(%@T_b1, %@T_wb) : (vector<16xf32>, vector<16xf32>) -> (vector<16xf32>)
          %@T_cs = vector.load %cos_t[%@T_i] : view<@HALFxf32> -> vector<16xf32>
          %@T_sn = vector.load %sin_t[%@T_i] : view<@HALFxf32> -> vector<16xf32>
          %@T_ac = func.call @fmul2(%@T_a, %@T_cs) : (vector<16xf32>, vector<16xf32>) -> (vector<16xf32>)
          %@T_bs = func.call @fmul2(%@T_b, %@T_sn) : (vector<16xf32>, vector<16xf32>) -> (vector<16xf32>)
          %@T_bc = func.call @fmul2(%@T_b, %@T_cs) : (vector<16xf32>, vector<16xf32>) -> (vector<16xf32>)
          %@T_as = func.call @fmul2(%@T_a, %@T_sn) : (vector<16xf32>, vector<16xf32>) -> (vector<16xf32>)
          %@T_ra = vector.subf %@T_ac, %@T_bs : vector<16xf32>
          %@T_rb = vector.addf %@T_bc, %@T_as : vector<16xf32>
          vector.store %@T_ra, %@V[%@T_ja] : vector<16xf32>, view<@VTxf32>
          vector.store %@T_rb, %@V[%@T_jb] : vector<16xf32>, view<@VTxf32>
        }
      }
)",
              {{"@T_heads", heads}, {"@BASE", base}, {"@VT", view.substr(view.find(':') + 1)},
               {"@V", view.substr(0, view.find(':'))}, {"@W", weight}, {"@HALF", S(half)}, {"@D", S(D)},
               {"@QH", S(s.heads)},
               {"@T", tag}});
}

}  // namespace

std::string QkPrepProgram(const AttentionShape& s, const std::string& name) {
  const std::uint32_t D = s.dim, Q = s.heads * D, KV = s.kv_heads * D, half = D / 2;
  const std::uint32_t row = s.qkvg_groups * 8, G = s.qkvg_groups;
  const std::uint32_t gstride = 16;  // gate floats per row in the gate output
  if (D % 32 || s.heads > 16 || (s.capacity & (s.capacity - 1)) || (s.slots & (s.slots - 1)) ||
      (s.table_rows & (s.table_rows - 1)))
    throw std::invalid_argument("unsupported NPU attention shape");
  const std::string args =
      "%a0: buffer, %a1: buffer, %a2: buffer, %a3: buffer, %a4: buffer, %a5: buffer, %a6: buffer, %a7: buffer, "
      "%a8: buffer, %a9: buffer";
  const std::string names = "%a0, %a1, %a2, %a3, %a4, %a5, %a6, %a7, %a8, %a9";
  const std::string types = "buffer, buffer, buffer, buffer, buffer, buffer, buffer, buffer, buffer, buffer";
  std::string t = "aie2p.target<array> @npu\n";
  t += kMathLibrary;
  t += "\npipeline.def<kernel> @" + name + "_col(%column: index) run(" + args + ") {\n";
  t += "  %proj, %qn, %kn, %cosb, %sinb, %meta, %qout, %gout, %kpool, %vpool = buffer.assume.alignment " + names +
       " {minimum_alignment = 64} : " + types + "\n";
  t += kConsts;
  t += Fill(R"(  %c7 = index.constant 7 : index
  %rows = index.constant @ROWS : index
  %rest = index.sub %rows, %column : index
  %rest7 = index.add %rest, %c7 : index
  %my_rows = index.div %rest7, %c8 : index
  %dim = index.constant @D : index
  %half = index.constant @HALF : index
  %qheads = index.constant @H : index
  %kvheads = index.constant @KVH : index
  %allheads = index.constant @ALLH : index
  %qfloats = index.constant @Q : index
  %kvfloats = index.constant @KV : index
  %gfloats = index.constant 16 : index
  %block = index.constant @BLOCK : index
  %cap = index.constant @CAP : index
  %memory = pipeline.memory<workgroup>[%column, 2] : pool
)",
            {{"@ROWS", S(s.rows)}, {"@D", S(D)}, {"@HALF", S(half)}, {"@H", S(s.heads)}, {"@KVH", S(s.kv_heads)}, {"@ALLH", S(s.heads + s.kv_heads)},
             {"@Q", S(Q)}, {"@KV", S(KV)}, {"@BLOCK", S(std::uint64_t(G) * 64)}, {"@CAP", S(s.capacity)}});
  const std::string prow = S(G) + "x8xf32";
  // Few channels per tile (hardware semaphores): small operands share aux
  // [qnw D | knw D | cos half | sin half | gate 16], K and V share kvh.
  const std::uint32_t aux = 3 * D + 16;
  const std::string auxs = S(aux) + "xf32", kvs = S(2 * KV) + "xf16";
  t += Scratch("p", S(row) + "xf32", row * 4) + Scratch("aux", auxs, aux * 4) + Scratch("mt", "16xi32", 64) +
       Scratch("kvh", kvs, 4 * KV);
  t += "  pipeline.strand workers([%column, 2], [1, 1], [1, 1]) {\n";
  for (auto [n, sh] : std::vector<std::pair<std::string, std::string>>{
           {"p", S(row) + "xf32"}, {"aux", auxs}, {"mt", "16xi32"}, {"kvh", kvs}})
    t += Reserve(n, sh);
  t += Fill(R"(    %qnw = view.subview %aux[0] : view<@AUX> -> view<@Dxf32>
    %knw = view.subview %aux[@D] : view<@AUX> -> view<@Dxf32>
    %cos = view.subview %aux[@D2] : view<@AUX> -> view<@HALFxf32>
    %sin = view.subview %aux[@D25] : view<@AUX> -> view<@HALFxf32>
    %gt = view.subview %aux[@D3] : view<@AUX> -> view<16xf32>
    %kh = view.subview %kvh[0] : view<@KVS> -> view<@KVxf16>
    %vh = view.subview %kvh[@KV] : view<@KVS> -> view<@KVxf16>
)",
            {{"@AUX", auxs}, {"@D25", S(2 * D + half)}, {"@D2", S(2 * D)}, {"@D3", S(3 * D)}, {"@D", S(D)},
             {"@HALF", S(half)}, {"@KVS", kvs}, {"@KV", S(KV)}});
  t += Fill(R"(    %vzero = vector.splatf 0.0 : vector<16xf32>
    %fzero = scalar.constant 0.0 : f32
    %inv_dim = vector.splatf @INV : vector<16xf32>
    %eps = vector.splatf @EPS : vector<16xf32>
    %qnw_src = buffer.view %qn[%origin] : buffer -> view<@Dxf32>
)" + Copy("qnw", "qnw_src", "view<@Dxf32>", "qnw", "view<@Dxf32>", true) +
                R"(    %knw_src = buffer.view %kn[%origin] : buffer -> view<@Dxf32>
)" + Copy("knw", "knw_src", "view<@Dxf32>", "knw", "view<@Dxf32>", true),
            {{"@D", S(D)}, {"@INV", "7.8125e-3"}, {"@EPS", "1.0e-5"}});
  {
    char inv[64];
    std::snprintf(inv, sizeof inv, "%.9e", 1.0 / D);
    t = Fill(t, {{"7.8125e-3", inv}});
  }
  t += "    scf.for %local = [%c0 to %my_rows step %c1] {\n"
       "      %lr = index.mul %local, %c8 : index\n"
       "      %row = index.add %lr, %column : index\n";
  // meta row
  t += "      %m_e = index.mul %row, %c16 : index\n" + Offset("m", "%m_e", 4) +
       "      %m_src = buffer.view %meta[%m_at] : buffer -> view<16xi32>\n" +
       Copy("m", "m_src", "view<16xi32>", "mt", "view<16xi32>", true);
  t += Fill(Meta("slot", 0, s.slots - 1) + Meta("pos", 1, s.table_rows - 1) + Meta("ring", 1, s.capacity - 1),
            {{"%meta_t", "%mt"}});
  // qkvg row from the blocked projection output
  t += Fill(R"(      %pb = index.div %row, %c8 : index
      %pr = index.rem %row, %c8 : index
      %pb0 = index.mul %pb, %block : index
      %pr0 = index.mul %pr, %c8 : index
      %pe = index.add %pb0, %pr0 : index
)" + Offset("p", "%pe", 4) +
                R"(      %p_dense = buffer.view %proj[%p_at] : buffer -> view<@Gx64xf32>
      %p_strided = view.refine %p_dense : view<@Gx64xf32> -> view<@Gx64xf32, #encoding.layout.strided<strides=[64, 1]>>
      %p_global = view.subview %p_strided[0, 0] : view<@Gx64xf32, #encoding.layout.strided<strides=[64, 1]>> -> view<@Gx8xf32, #encoding.layout.strided<strides=[64, 1]>>
      %p_shaped = view.bitcast %p : view<@ROWxf32> -> view<@PROW>
)" + Copy("pin", "p_global", "view<@Gx8xf32, #encoding.layout.strided<strides=[64, 1]>>", "p_shaped", "view<@PROW>", true) +
                // rotary tables at the position
                R"(      %t_e = index.mul %pos, %half : index
)" + Offset("t", "%t_e", 4) +
                R"(      %cos_src = buffer.view %cosb[%t_at] : buffer -> view<@HALFxf32>
      %sin_src = buffer.view %sinb[%t_at] : buffer -> view<@HALFxf32>
)" + Copy("cs", "cos_src", "view<@HALFxf32>", "cos", "view<@HALFxf32>", true) +
                Copy("sn", "sin_src", "view<@HALFxf32>", "sin", "view<@HALFxf32>", true),
            {{"@G", S(G)}, {"@ROW", S(row)}, {"@PROW", prow}, {"@HALF", S(half)}});
  t = Fill(t, {{"%cos_t", "%cos"}, {"%sin_t", "%sin"}});
  // Q heads then K heads (contiguous in the row): one loop, weight per head.
  t += Fill(NormRope(s, "p:" + S(row), "allheads", "%c0", "hw", "q"), {{"%cos_t", "%cos"}, {"%sin_t", "%sin"}});
  // K and V to F16, gates through the sigmoid.
  t += Fill(R"(      scf.for %i = [%c0 to %kvfloats step %c16] {
        %ki = index.add %i, %qfloats : index
        %vi0 = index.add %ki, %kvfloats : index
        %kv = vector.load %p[%ki] : view<@ROWxf32> -> vector<16xf32>
        %vv = vector.load %p[%vi0] : view<@ROWxf32> -> vector<16xf32>
        %k16 = func.call @tof16(%kv) : (vector<16xf32>) -> (vector<16xf16>)
        %v16 = func.call @tof16(%vv) : (vector<16xf32>) -> (vector<16xf16>)
        vector.store %k16, %kh[%i] : vector<16xf16>, view<@KVxf16>
        vector.store %v16, %vh[%i] : vector<16xf16>, view<@KVxf16>
      }
)",
            {{"@ROW", S(row)}, {"@KV", S(KV)}});
  if (s.gate) {
    t += Fill(R"(      %g_at0 = index.add %qfloats, %kvfloats : index
      %g_at = index.add %g_at0, %kvfloats : index
      %graw = vector.load %p[%g_at] : view<@ROWxf32> -> vector<16xf32>
      vector.store %graw, %gt[%c0] : vector<16xf32>, view<16xf32>
)",
              {{"@ROW", S(row)}});
  } else {
    t += "      %gone = vector.splatf 1.0e30 : vector<16xf32>\n"
         "      vector.store %gone, %gt[%c0] : vector<16xf32>, view<16xf32>\n";
  }
  // Store q rows, gates and the K/V ring row.
  t += Fill(R"(      %q_e = index.mul %row, %qfloats : index
)" + Offset("q", "%q_e", 4) +
                R"(      %q_dst = buffer.view %qout[%q_at] : buffer -> view<@Qxf32>
      %q_src = view.subview %p[0] : view<@ROWxf32> -> view<@Qxf32>
)" + Copy("qo", "q_src", "view<@Qxf32>", "q_dst", "view<@Qxf32>", false) +
                R"(      %g_e = index.mul %row, %gfloats : index
)" + Offset("go", "%g_e", 4) +
                R"(      %g_dst = buffer.view %gout[%go_at] : buffer -> view<16xf32>
)" + Copy("gw", "gt", "view<16xf32>", "g_dst", "view<16xf32>", false) +
                R"(      %r0 = index.mul %slot, %cap : index
      %r1 = index.add %r0, %ring : index
      %r2 = index.mul %r1, %kvfloats : index
)" + Offset("kv", "%r2", 2) +
                R"(      %k_dst = buffer.view %kpool[%kv_at] : buffer -> view<@KVxf16>
      %v_dst = buffer.view %vpool[%kv_at] : buffer -> view<@KVxf16>
)" + Copy("kw", "kh", "view<@KVxf16>", "k_dst", "view<@KVxf16>", false) +
                Copy("vw", "vh", "view<@KVxf16>", "v_dst", "view<@KVxf16>", false),
            {{"@Q", S(Q)}, {"@ROW", S(row)}, {"@KV", S(KV)}});
  t += "    }\n";
  for (auto [n, sh] : std::vector<std::pair<std::string, std::string>>{
           {"p", S(row) + "xf32"}, {"aux", auxs}, {"mt", "16xi32"}, {"kvh", kvs}})
    t += Publish(n, sh);
  t += "  }\n}\n";
  (void)gstride;
  return t + Compose(name, args, names, types);
}

std::string AttendProgram(const AttentionShape& s, const std::string& name) {
  // Two cores per column (each has 16 KiB of program memory): the scorer on
  // [c, 2] computes masked scores and the online softmax; the accumulator on
  // [c, 3] sums probability-weighted V rows and writes the gated output.
  // Per chunk the scorer sends [probs 32 | alpha 16 | unused 16]; after a
  // head's chunks it sends [den 16 | e^-gate 16 | unused 32].
  const std::uint32_t D = s.dim, Q = s.heads * D, KV = s.kv_heads * D;
  const std::uint32_t hpc = s.heads / 8, group = s.heads / s.kv_heads;
  if (s.heads % 8 || D % 16 || (s.capacity & (s.capacity - 1)) || s.capacity < 32 ||
      (s.slots & (s.slots - 1)) || hpc * D > 512)
    throw std::invalid_argument("unsupported NPU attention shape");
  char scale[64];
  std::snprintf(scale, sizeof scale, "%.9e", 1.0 / std::sqrt(double(D)));
  const std::string body = R"(
pipeline.def<kernel> @@NAME_col(%column: index) run(%a0: buffer, %a1: buffer, %a2: buffer, %a3: buffer, %a4: buffer, %a5: buffer, %a6: buffer, %a7: buffer) {
  %qb, %gb, %kpool, %vpool, %meta, %kgroups, %out, %cntb = buffer.assume.alignment %a0, %a1, %a2, %a3, %a4, %a5, %a6, %a7 {minimum_alignment = 64} : buffer, buffer, buffer, buffer, buffer, buffer, buffer, buffer
  %origin = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c2 = index.constant 2 : index
  %c3 = index.constant 3 : index
  %c4 = index.constant 4 : index
  %c8 = index.constant 8 : index
  %c16 = index.constant 16 : index
  %c32 = index.constant 32 : index
  %c48 = index.constant 48 : index
  %dim = index.constant @D : index
  %hpc = index.constant @HPC : index
  %qc = index.constant @QC : index
  %qfloats = index.constant @Q : index
  %kvfloats = index.constant @KV : index
  %cap = index.constant @CAP : index
  %group = index.constant @GROUP : index
  %oblock = index.constant @OBLOCK : index
  %chunk = index.constant 32 : index
  %first_head = index.mul %column, %hpc : index
  %pool2 = pipeline.memory<workgroup>[%column, 2] : pool
  %pool3 = pipeline.memory<workgroup>[%column, 3] : pool
  %sa_size = index.constant @SABYTES : offset
  %sa_storage = buffer.alloca<workgroup> pool(%pool2) align(64) %sa_size : buffer
  %sa_slots = buffer.view %sa_storage[%origin] : buffer -> view<1x@SAxf32>
  %sa_ch = channel.bind<fifo> %sa_slots, %c1 : view<1x@SAxf32>, index -> channel<tile<@SAxf32>>
  %kc_size = index.constant @KCBYTES : offset
  %kc_storage = buffer.alloca<workgroup> pool(%pool2) align(64) %kc_size : buffer
  %kc_slots = buffer.view %kc_storage[%origin] : buffer -> view<1x32x@Dxf16>
  %kc_ch = channel.bind<fifo> %kc_slots, %c1 : view<1x32x@Dxf16>, index -> channel<tile<32x@Dxf16>>
  %m2_size = index.constant 192 : offset
  %m2_storage = buffer.alloca<workgroup> pool(%pool2) align(64) %m2_size : buffer
  %m2_slots = buffer.view %m2_storage[%origin] : buffer -> view<1x48xi32>
  %m2_ch = channel.bind<fifo> %m2_slots, %c1 : view<1x48xi32>, index -> channel<tile<48xi32>>
  %pc_size = index.constant 512 : offset
  %pc_storage = buffer.alloca<workgroup> pool(%pool3) align(64) %pc_size : buffer
  %pc_slots = buffer.view %pc_storage[%origin] : buffer -> view<2x64xf32>
  %pc_ch = channel.bind<fifo> %pc_slots, %c2 : view<2x64xf32>, index -> channel<tile<64xf32>>
  %vc_size = index.constant @KCBYTES : offset
  %vc_storage = buffer.alloca<workgroup> pool(%pool3) align(64) %vc_size : buffer
  %vc_slots = buffer.view %vc_storage[%origin] : buffer -> view<1x32x@Dxf16>
  %vc_ch = channel.bind<fifo> %vc_slots, %c1 : view<1x32x@Dxf16>, index -> channel<tile<32x@Dxf16>>
  %pa_size = index.constant @PABYTES : offset
  %pa_storage = buffer.alloca<workgroup> pool(%pool3) align(64) %pa_size : buffer
  %pa_slots = buffer.view %pa_storage[%origin] : buffer -> view<1x@PAxf32>
  %pa_ch = channel.bind<fifo> %pa_slots, %c1 : view<1x@PAxf32>, index -> channel<tile<@PAxf32>>
  %ot_size = index.constant @OTBYTES : offset
  %ot_storage = buffer.alloca<workgroup> pool(%pool3) align(64) %ot_size : buffer
  %ot_slots = buffer.view %ot_storage[%origin] : buffer -> view<1x@QCxbf16>
  %ot_ch = channel.bind<fifo> %ot_slots, %c1 : view<1x@QCxbf16>, index -> channel<tile<@QCxbf16>>

  // Scorer.
  pipeline.strand workers([%column, 2], [1, 1], [1, 1]) {
    %sa_w, %sa = channel.reserve %sa_ch : channel<tile<@SAxf32>> -> (write<tile<@SAxf32>>, view<@SAxf32>)
    %kc_w, %kc = channel.reserve %kc_ch : channel<tile<32x@Dxf16>> -> (write<tile<32x@Dxf16>>, view<32x@Dxf16>)
    %m2_w, %m2 = channel.reserve %m2_ch : channel<tile<48xi32>> -> (write<tile<48xi32>>, view<48xi32>)
    %qv = view.subview %sa[0] : view<@SAxf32> -> view<@QCxf32>
    %gv = view.subview %sa[@QC] : view<@SAxf32> -> view<16xf32>
    %scores = view.subview %sa[@SS] : view<@SAxf32> -> view<32xf32>
    %ex = view.subview %sa[@SX] : view<@SAxf32> -> view<48xf32>
    %mrow = view.subview %m2[0] : view<48xi32> -> view<16xi32>
    %kg = view.subview %m2[16] : view<48xi32> -> view<32xi32>
    %vzero = vector.splatf 0.0 : vector<16xf32>
    %fzero = scalar.constant 0.0 : f32
    %neg = vector.splatf -1.0e30 : vector<16xf32>
    %scale = vector.splatf @SCALE : vector<16xf32>
    %lanes = vector.constanti [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<16xi32>
    %c16i = vector.splati 16 : vector<16xi32>
    %lanes_hi = vector.addi %lanes, %c16i : vector<16xi32>
    %zero_i = vector.splati 0 : vector<16xi32>
    %cn_src = buffer.view %cntb[%origin] : buffer -> view<16xi32>
    %cnc = kernel.async.copy %cn_src to %mrow {cache_scope = workgroup, cache_temporal = regular, direction = global_to_workgroup} : view<16xi32> to view<16xi32> -> kernel.async.token
    %cng = kernel.async.group %cnc : kernel.async.token -> kernel.async.group
    kernel.async.wait %cng {newer_groups = 0} : kernel.async.group
    %cn_raw = view.load %mrow[%c0] : view<16xi32> -> i32
    %cn_mask = scalar.constant 255 : i32
    %cn_b = scalar.andi %cn_raw, %cn_mask : i32
    %rows = index.cast %cn_b : i32 to index
    scf.for %row = [%c0 to %rows step %c1] {
@META
      %qgroup_raw = view.load %mrow[%c4] : view<16xi32> -> i32
      %qgroup = vector.splat %qgroup_raw : vector<16xi32>
      %q_r = index.mul %row, %qfloats : index
      %q_h = index.mul %first_head, %dim : index
      %q_e = index.add %q_r, %q_h : index
      %q_bytes = index.mul %q_e, %c4 : index
      %q_at = index.cast %q_bytes : index to offset
      %q_src = buffer.view %qb[%q_at] : buffer -> view<@QCxf32>
@COPY_Q
      scf.for %i = [%c0 to %qc step %c16] {
        %v = vector.load %qv[%i] : view<@QCxf32> -> vector<16xf32>
        %w = func.call @fmul2(%v, %scale) : (vector<16xf32>, vector<16xf32>) -> (vector<16xf32>)
        vector.store %w, %qv[%i] : vector<16xf32>, view<@QCxf32>
      }
      %g_e = index.mul %row, %c16 : index
      %g_bytes = index.mul %g_e, %c4 : index
      %g_at = index.cast %g_bytes : index to offset
      %g_src = buffer.view %gb[%g_at] : buffer -> view<16xf32>
@COPY_G
      %gneg0 = vector.load %gv[%c0] : view<16xf32> -> vector<16xf32>
      %gneg = vector.subf %vzero, %gneg0 : vector<16xf32>
      %geg = func.call @exp_fast(%gneg) : (vector<16xf32>) -> (vector<16xf32>)
      vector.store %geg, %gv[%c0] : vector<16xf32>, view<16xf32>
@CHUNKS
      %first_v = vector.splat %first_b : vector<16xi32>
      %last_v = vector.splat %last_b : vector<16xi32>
      scf.for %hl = [%c0 to %hpc step %c1] {
        %h = index.add %first_head, %hl : index
        %kh = index.div %h, %group : index
        %qoff = index.mul %hl, %dim : index
        %m_f, %d_f = scf.for %ch = [%c0 to %chunks step %c1](%m = %neg : vector<16xf32>, %den = %vzero : vector<16xf32>) -> (vector<16xf32>, vector<16xf32>) {
@KADDR
          %k_dense = buffer.view %kpool[%k_at] : buffer -> view<32x@KVxf16>
          %k_str = view.refine %k_dense : view<32x@KVxf16> -> view<32x@KVxf16, #encoding.layout.strided<strides=[@KV, 1]>>
          %k_glob = view.subview %k_str[0, 0] : view<32x@KVxf16, #encoding.layout.strided<strides=[@KV, 1]>> -> view<32x@Dxf16, #encoding.layout.strided<strides=[@KV, 1]>>
          %kcp = kernel.async.copy %k_glob to %kc {cache_scope = workgroup, cache_temporal = regular, direction = global_to_workgroup} : view<32x@Dxf16, #encoding.layout.strided<strides=[@KV, 1]>> to view<32x@Dxf16> -> kernel.async.token
          %kcg = kernel.async.group %kcp : kernel.async.token -> kernel.async.group
          kernel.async.wait %kcg {newer_groups = 0} : kernel.async.group
@GROUPS
          scf.for %j = [%c0 to %chunk step %c1] {
            %dot = scf.for %i = [%c0 to %dim step %c16](%a = %vzero : vector<16xf32>) -> (vector<16xf32>) {
              %qi = index.add %qoff, %i : index
              %qq = vector.load %qv[%qi] : view<@QCxf32> -> vector<16xf32>
              %k16 = vector.load %kc[%j, %i] : view<32x@Dxf16> -> vector<16xf16>
              %kk = vector.extf %k16 : vector<16xf16> to vector<16xf32>
              %p = func.call @fmul2(%qq, %kk) : (vector<16xf32>, vector<16xf32>) -> (vector<16xf32>)
              %n = vector.addf %a, %p : vector<16xf32>
              scf.yield %n : vector<16xf32>
            }
            %sj = vector.reduce<addf> %dot, %fzero : vector<16xf32>, f32
            view.store %sj, %scores[%j] : f32, view<32xf32>
          }
          %t0i = index.cast %t0 : index to i32
          %t0v = vector.splat %t0i : vector<16xi32>
          %keys_lo = vector.addi %t0v, %lanes : vector<16xi32>
          %keys_hi = vector.addi %t0v, %lanes_hi : vector<16xi32>
          %s_lo0 = vector.load %scores[%c0] : view<32xf32> -> vector<16xf32>
          %s_hi0 = vector.load %scores[%c16] : view<32xf32> -> vector<16xf32>
          %lo_ge = vector.cmpi sge, %keys_lo, %first_v : vector<16xi32> -> vector<16xi1>
          %lo_le = vector.cmpi sle, %keys_lo, %last_v : vector<16xi32> -> vector<16xi1>
          %hi_ge = vector.cmpi sge, %keys_hi, %first_v : vector<16xi32> -> vector<16xi1>
          %hi_le = vector.cmpi sle, %keys_hi, %last_v : vector<16xi32> -> vector<16xi1>
          %lo_in = vector.andi %lo_ge, %lo_le : vector<16xi1>
          %hi_in = vector.andi %hi_ge, %hi_le : vector<16xi1>
@MASK
          %s_lo = vector.select %lo_ok, %s_lo0, %neg : vector<16xf32>
          %s_hi = vector.select %hi_ok, %s_hi0, %neg : vector<16xf32>
          %mx0 = vector.maxnumf %s_lo, %s_hi : vector<16xf32>
          %mxs = vector.reduce<maxnumf> %mx0, %fzero : vector<16xf32>, f32
          %mxv = vector.splat %mxs : vector<16xf32>
          %m_new = vector.maxnumf %m, %mxv : vector<16xf32>
          %da = vector.subf %m, %m_new : vector<16xf32>
          %dl = vector.subf %s_lo, %m_new : vector<16xf32>
          %dh = vector.subf %s_hi, %m_new : vector<16xf32>
          vector.store %dl, %ex[%c0] : vector<16xf32>, view<48xf32>
          vector.store %dh, %ex[%c16] : vector<16xf32>, view<48xf32>
          vector.store %da, %ex[%c32] : vector<16xf32>, view<48xf32>
          scf.for %ei = [%c0 to %c48 step %c16] {
            %ev = vector.load %ex[%ei] : view<48xf32> -> vector<16xf32>
            %ee = func.call @exp_fast(%ev) : (vector<16xf32>) -> (vector<16xf32>)
            vector.store %ee, %ex[%ei] : vector<16xf32>, view<48xf32>
          }
          %p_lo = vector.load %ex[%c0] : view<48xf32> -> vector<16xf32>
          %p_hi = vector.load %ex[%c16] : view<48xf32> -> vector<16xf32>
          %alpha = vector.load %ex[%c32] : view<48xf32> -> vector<16xf32>
          %pc_w, %pc = channel.reserve %pc_ch : channel<tile<64xf32>> -> (write<tile<64xf32>>, view<64xf32>)
          vector.store %p_lo, %pc[%c0] : vector<16xf32>, view<64xf32>
          vector.store %p_hi, %pc[%c16] : vector<16xf32>, view<64xf32>
          vector.store %alpha, %pc[%c32] : vector<16xf32>, view<64xf32>
          channel.publish %pc_w : write<tile<64xf32>>
          %psum = vector.addf %p_lo, %p_hi : vector<16xf32>
          %ps = vector.reduce<addf> %psum, %fzero : vector<16xf32>, f32
          %psv = vector.splat %ps : vector<16xf32>
          %den_a = func.call @fmul2(%den, %alpha) : (vector<16xf32>, vector<16xf32>) -> (vector<16xf32>)
          %den_n = vector.addf %den_a, %psv : vector<16xf32>
          scf.yield %m_new, %den_n : vector<16xf32>, vector<16xf32>
        }
        %gh = view.load %gv[%h] : view<16xf32> -> f32
        %ghv = vector.splat %gh : vector<16xf32>
        %fin_w, %fin = channel.reserve %pc_ch : channel<tile<64xf32>> -> (write<tile<64xf32>>, view<64xf32>)
        vector.store %d_f, %fin[%c0] : vector<16xf32>, view<64xf32>
        vector.store %ghv, %fin[%c16] : vector<16xf32>, view<64xf32>
        channel.publish %fin_w : write<tile<64xf32>>
      }
    }
    channel.publish %sa_w : write<tile<@SAxf32>>
    channel.publish %kc_w : write<tile<32x@Dxf16>>
    channel.publish %m2_w : write<tile<48xi32>>
  }

  // Accumulator.
  pipeline.strand workers([%column, 3], [1, 1], [1, 1]) {
    %vc_w, %vc = channel.reserve %vc_ch : channel<tile<32x@Dxf16>> -> (write<tile<32x@Dxf16>>, view<32x@Dxf16>)
    %pa_w, %pa = channel.reserve %pa_ch : channel<tile<@PAxf32>> -> (write<tile<@PAxf32>>, view<@PAxf32>)
    %ot_w, %ot = channel.reserve %ot_ch : channel<tile<@QCxbf16>> -> (write<tile<@QCxbf16>>, view<@QCxbf16>)
    %acc = view.subview %pa[0] : view<@PAxf32> -> view<@Dxf32>
    %mbits = view.subview %pa[@D] : view<@PAxf32> -> view<16xf32>
    %mrow = view.bitcast %mbits : view<16xf32> -> view<16xi32>
    %vzero = vector.splatf 0.0 : vector<16xf32>
    %onev = vector.splatf 1.0 : vector<16xf32>
    %cn_src = buffer.view %cntb[%origin] : buffer -> view<16xi32>
    %cnc = kernel.async.copy %cn_src to %mrow {cache_scope = workgroup, cache_temporal = regular, direction = global_to_workgroup} : view<16xi32> to view<16xi32> -> kernel.async.token
    %cng = kernel.async.group %cnc : kernel.async.token -> kernel.async.group
    kernel.async.wait %cng {newer_groups = 0} : kernel.async.group
    %cn_raw = view.load %mrow[%c0] : view<16xi32> -> i32
    %cn_mask = scalar.constant 255 : i32
    %cn_b = scalar.andi %cn_raw, %cn_mask : i32
    %rows = index.cast %cn_b : i32 to index
    scf.for %row = [%c0 to %rows step %c1] {
@META
@CHUNKS
      scf.for %hl = [%c0 to %hpc step %c1] {
        %h = index.add %first_head, %hl : index
        %kh = index.div %h, %group : index
        %qoff = index.mul %hl, %dim : index
        scf.for %i = [%c0 to %dim step %c16] {
          vector.store %vzero, %acc[%i] : vector<16xf32>, view<@Dxf32>
        }
        scf.for %ch = [%c0 to %chunks step %c1] {
@KADDR
          %v_dense = buffer.view %vpool[%k_at] : buffer -> view<32x@KVxf16>
          %v_str = view.refine %v_dense : view<32x@KVxf16> -> view<32x@KVxf16, #encoding.layout.strided<strides=[@KV, 1]>>
          %v_glob = view.subview %v_str[0, 0] : view<32x@KVxf16, #encoding.layout.strided<strides=[@KV, 1]>> -> view<32x@Dxf16, #encoding.layout.strided<strides=[@KV, 1]>>
          %vcp = kernel.async.copy %v_glob to %vc {cache_scope = workgroup, cache_temporal = regular, direction = global_to_workgroup} : view<32x@Dxf16, #encoding.layout.strided<strides=[@KV, 1]>> to view<32x@Dxf16> -> kernel.async.token
          %vcg = kernel.async.group %vcp : kernel.async.token -> kernel.async.group
          kernel.async.wait %vcg {newer_groups = 0} : kernel.async.group
          %pc_r, %pc = channel.acquire %pc_ch : channel<tile<64xf32>> -> (read<tile<64xf32>>, view<64xf32>)
          %alpha = vector.load %pc[%c32] : view<64xf32> -> vector<16xf32>
          scf.for %i = [%c0 to %dim step %c16] {
            %av = vector.load %acc[%i] : view<@Dxf32> -> vector<16xf32>
            %as = func.call @fmul2(%av, %alpha) : (vector<16xf32>, vector<16xf32>) -> (vector<16xf32>)
            %sum = scf.for %j = [%c0 to %chunk step %c1](%b = %as : vector<16xf32>) -> (vector<16xf32>) {
              %pj = view.load %pc[%j] : view<64xf32> -> f32
              %pv = vector.splat %pj : vector<16xf32>
              %v16 = vector.load %vc[%j, %i] : view<32x@Dxf16> -> vector<16xf16>
              %vv = vector.extf %v16 : vector<16xf16> to vector<16xf32>
              %t = func.call @fmul2(%pv, %vv) : (vector<16xf32>, vector<16xf32>) -> (vector<16xf32>)
              %bn = vector.addf %b, %t : vector<16xf32>
              scf.yield %bn : vector<16xf32>
            }
            vector.store %sum, %acc[%i] : vector<16xf32>, view<@Dxf32>
          }
          channel.release %pc_r : read<tile<64xf32>>
        }
        %fin_r, %fin = channel.acquire %pc_ch : channel<tile<64xf32>> -> (read<tile<64xf32>>, view<64xf32>)
        %dv = vector.load %fin[%c0] : view<64xf32> -> vector<16xf32>
        %gv = vector.load %fin[%c16] : view<64xf32> -> vector<16xf32>
        channel.release %fin_r : read<tile<64xf32>>
        %g1 = vector.addf %onev, %gv : vector<16xf32>
        %dg = func.call @fmul2(%dv, %g1) : (vector<16xf32>, vector<16xf32>) -> (vector<16xf32>)
        %scale_o = func.call @recip(%dg) : (vector<16xf32>) -> (vector<16xf32>)
        scf.for %i = [%c0 to %dim step %c16] {
          %av = vector.load %acc[%i] : view<@Dxf32> -> vector<16xf32>
          %o = func.call @fmul2(%av, %scale_o) : (vector<16xf32>, vector<16xf32>) -> (vector<16xf32>)
          %ob = vector.fptrunc %o : vector<16xf32> to vector<16xbf16>
          %oi = index.add %qoff, %i : index
          vector.store %ob, %ot[%oi] : vector<16xbf16>, view<@QCxbf16>
        }
      }
      %ob_blk = index.div %row, %c8 : index
      %ob_b = index.rem %row, %c8 : index
      %ob0 = index.mul %ob_blk, %oblock : index
      %ob1 = index.mul %first_head, %dim : index
      %ob2 = index.mul %ob1, %c8 : index
      %ob3 = index.mul %ob_b, %c8 : index
      %ob4 = index.add %ob0, %ob2 : index
      %ob5 = index.add %ob4, %ob3 : index
      %o_bytes = index.mul %ob5, %c2 : index
      %o_at = index.cast %o_bytes : index to offset
      %o_dense = buffer.view %out[%o_at] : buffer -> view<@OGx64xbf16>
      %o_str = view.refine %o_dense : view<@OGx64xbf16> -> view<@OGx64xbf16, #encoding.layout.strided<strides=[64, 1]>>
      %o_glob = view.subview %o_str[0, 0] : view<@OGx64xbf16, #encoding.layout.strided<strides=[64, 1]>> -> view<@OGx8xbf16, #encoding.layout.strided<strides=[64, 1]>>
      %o_shaped = view.bitcast %ot : view<@QCxbf16> -> view<@OGx8xbf16>
      %ocp = kernel.async.copy %o_shaped to %o_glob {cache_scope = workgroup, cache_temporal = regular, direction = workgroup_to_global} : view<@OGx8xbf16> to view<@OGx8xbf16, #encoding.layout.strided<strides=[64, 1]>> -> kernel.async.token
      %ocg = kernel.async.group %ocp : kernel.async.token -> kernel.async.group
      kernel.async.wait %ocg {newer_groups = 0} : kernel.async.group
    }
    channel.publish %vc_w : write<tile<32x@Dxf16>>
    channel.publish %pa_w : write<tile<@PAxf32>>
    channel.publish %ot_w : write<tile<@QCxbf16>>
  }
}
)";
  const std::string meta = R"(      %m_e = index.mul %row, %c16 : index
      %m_bytes = index.mul %m_e, %c4 : index
      %m_at = index.cast %m_bytes : index to offset
      %m_src = buffer.view %meta[%m_at] : buffer -> view<16xi32>
      %mcp = kernel.async.copy %m_src to %mrow {cache_scope = workgroup, cache_temporal = regular, direction = global_to_workgroup} : view<16xi32> to view<16xi32> -> kernel.async.token
      %mcg = kernel.async.group %mcp : kernel.async.token -> kernel.async.group
      kernel.async.wait %mcg {newer_groups = 0} : kernel.async.group
      %slot_raw = view.load %mrow[%c0] : view<16xi32> -> i32
      %first_raw = view.load %mrow[%c2] : view<16xi32> -> i32
      %last_raw = view.load %mrow[%c3] : view<16xi32> -> i32
      %slot_mask = scalar.constant @SLOTM : i32
      %pos_mask = scalar.constant 1048575 : i32
      %slot_b = scalar.andi %slot_raw, %slot_mask : i32
      %first_b = scalar.andi %first_raw, %pos_mask : i32
      %last_b = scalar.andi %last_raw, %pos_mask : i32
      %slot = index.cast %slot_b : i32 to index)";
  const std::string chunks = R"(      %sh5 = scalar.constant 5 : i32
      %one_i32 = scalar.constant 1 : i32
      %cm_i32 = scalar.constant 32767 : i32
      %fc_i = scalar.shrui %first_b, %sh5 : i32
      %lc_i = scalar.shrui %last_b, %sh5 : i32
      %nc0 = scalar.subi %lc_i, %fc_i : i32
      %nc1 = scalar.addi %nc0, %one_i32 : i32
      %nc2 = scalar.andi %nc1, %cm_i32 : i32
      %chunks = index.cast %nc2 : i32 to index
      %fcm = scalar.andi %fc_i, %cm_i32 : i32
      %first_chunk = index.cast %fcm : i32 to index
      %c_lo = index.mul %first_chunk, %chunk : index)";
  const std::string kaddr = R"(          %ch32 = index.mul %ch, %chunk : index
          %t0 = index.add %c_lo, %ch32 : index
          %t0_r = index.cast %t0 : index to i32
          %capm_i = scalar.constant @CAPM : i32
          %ring_i = scalar.andi %t0_r, %capm_i : i32
          %ring = index.cast %ring_i : i32 to index
          %sc = index.mul %slot, %cap : index
          %rr = index.add %sc, %ring : index
          %re = index.mul %rr, %kvfloats : index
          %ho = index.mul %kh, %dim : index
          %ke = index.add %re, %ho : index
          %k_bytes = index.mul %ke, %c2 : index
          %k_at = index.cast %k_bytes : index to offset)";
  const std::string groups = s.groups ? R"(          %kg_bytes = index.mul %t0, %c4 : index
          %kg_at = index.cast %kg_bytes : index to offset
          %kg_src = buffer.view %kgroups[%kg_at] : buffer -> view<32xi32>
          %kgp = kernel.async.copy %kg_src to %kg {cache_scope = workgroup, cache_temporal = regular, direction = global_to_workgroup} : view<32xi32> to view<32xi32> -> kernel.async.token
          %kgg = kernel.async.group %kgp : kernel.async.token -> kernel.async.group
          kernel.async.wait %kgg {newer_groups = 0} : kernel.async.group)"
                                      : "";
  const std::string mask = s.groups ? R"(          %g_lo = vector.load %kg[%c0] : view<32xi32> -> vector<16xi32>
          %g_hi = vector.load %kg[%c16] : view<32xi32> -> vector<16xi32>
          %qstate = vector.cmpi slt, %qgroup, %zero_i : vector<16xi32> -> vector<16xi1>
          %kst_lo = vector.cmpi slt, %g_lo, %zero_i : vector<16xi32> -> vector<16xi1>
          %kst_hi = vector.cmpi slt, %g_hi, %zero_i : vector<16xi32> -> vector<16xi1>
          %same_lo = vector.cmpi eq, %g_lo, %qgroup : vector<16xi32> -> vector<16xi1>
          %same_hi = vector.cmpi eq, %g_hi, %qgroup : vector<16xi32> -> vector<16xi1>
          %v1_lo = vector.ori %qstate, %kst_lo : vector<16xi1>
          %v1_hi = vector.ori %qstate, %kst_hi : vector<16xi1>
          %vis_lo = vector.ori %v1_lo, %same_lo : vector<16xi1>
          %vis_hi = vector.ori %v1_hi, %same_hi : vector<16xi1>
          %lo_ok = vector.andi %lo_in, %vis_lo : vector<16xi1>
          %hi_ok = vector.andi %hi_in, %vis_hi : vector<16xi1>)"
                                    : R"(          %lo_ok = vector.andi %lo_in, %lo_in : vector<16xi1>
          %hi_ok = vector.andi %hi_in, %hi_in : vector<16xi1>)";
  const std::string copy_q = R"(      %qcp = kernel.async.copy %q_src to %qv {cache_scope = workgroup, cache_temporal = regular, direction = global_to_workgroup} : view<@QCxf32> to view<@QCxf32> -> kernel.async.token
      %qcg = kernel.async.group %qcp : kernel.async.token -> kernel.async.group
      kernel.async.wait %qcg {newer_groups = 0} : kernel.async.group)";
  const std::string copy_g = R"(      %gcp = kernel.async.copy %g_src to %gv {cache_scope = workgroup, cache_temporal = regular, direction = global_to_workgroup} : view<16xf32> to view<16xf32> -> kernel.async.token
      %gcg = kernel.async.group %gcp : kernel.async.token -> kernel.async.group
      kernel.async.wait %gcg {newer_groups = 0} : kernel.async.group)";
  const std::uint32_t sa = hpc * D + 16 + 32 + 48;  // q | gates | scores | ex
  const std::uint32_t pa = D + 16;                   // acc | meta
  std::string text = Fill(body, {{"@META", meta}, {"@CHUNKS", chunks}, {"@KADDR", kaddr}, {"@GROUPS", groups},
                                 {"@MASK", mask}, {"@COPY_Q", copy_q}, {"@COPY_G", copy_g}});
  // The accumulator reads meta into its own tile record.
  {
    const auto acc_at = text.find("  // Accumulator.");
    std::string tail = text.substr(acc_at);
    tail = Fill(tail, {{"%mcp = kernel.async.copy %m_src to %mrow", "%mcp = kernel.async.copy %m_src to %mrow"}});
    text = text.substr(0, acc_at) + tail;
  }
  text = Fill(text, {{"@NAME", name}, {"@ROWS", S(s.rows)}, {"@HPC", S(hpc)}, {"@QCxf32", S(hpc * D) + "xf32"},
                     {"@QCxbf16", S(hpc * D) + "xbf16"}, {"@QC", S(hpc * D)}, {"@Q", S(Q)}, {"@KVxf16", S(KV) + "xf16"},
                     {"@KV", S(KV)}, {"@CAPM", S(s.capacity - 1)}, {"@CAP", S(s.capacity)}, {"@GROUP", S(group)},
                     {"@OBLOCK", S(std::uint64_t(Q / 8) * 64)}, {"@SABYTES", S(sa * 4)}, {"@SAxf32", S(sa) + "xf32"},
                     {"@SS", S(hpc * D + 16)}, {"@SX", S(hpc * D + 48)}, {"@KCBYTES", S(32 * D * 2)},
                     {"@PABYTES", S(pa * 4)}, {"@PAxf32", S(pa) + "xf32"}, {"@OTBYTES", S(hpc * D * 2)},
                     {"@OG", S(hpc * D / 8)}, {"@SLOTM", S(s.slots - 1)}, {"@SCALE", scale}, {"@D", S(D)}});
  std::string head = "aie2p.target<array> @npu\n";
  head += kMathLibrary;
  return head + text +
         Compose(name, "%a0: buffer, %a1: buffer, %a2: buffer, %a3: buffer, %a4: buffer, %a5: buffer, %a6: buffer, %a7: buffer",
                 "%a0, %a1, %a2, %a3, %a4, %a5, %a6, %a7", "buffer, buffer, buffer, buffer, buffer, buffer, buffer, buffer");
}

// Attention with BFP16 matrix units (8x8x8 vector.mma). Column c serves
// query heads [c H/8, (c + 1) H/8) and their KV head. A block holds `rpb`
// rows (1, or 2/4 consecutive rows of one session in position order) times
// the column's heads as query vectors, lane q = row * heads_per_column + h
// (and q + 8). Two cores per column:
//   [c, 2] scorer: S^T = K Q^T per 32-key chunk of the session's ring
//          (key-major, so softmax reductions run over tile rows), masking,
//          online softmax in base 2 with a lazily moved max (O and l are
//          rescaled only when a query's max grows by more than 8);
//   [c, 3] accumulator: O^T += V^T P (BF16 transposes only: AIE2P cannot
//          lower F32 ones), normalization, gate, blocked BF16 output.
// Per block the scorer sends a header record, one record per chunk
// [P | alpha | rescale flag] and a final record [l | gate].
// The caller's Q must carry MmaQueryScale(D) = log2(e) / sqrt(D).
// No d1 group masks.
double MmaQueryScale(std::uint32_t dim) { return 1.4426950408889634 / std::sqrt(double(dim)); }

std::string AttendMmaProgram(const AttentionShape& s, const std::string& name, std::uint32_t rpb) {
  const std::uint32_t D = s.dim, Q = s.heads * D, KV = s.kv_heads * D, group = s.heads / s.kv_heads;
  const std::uint32_t hpc = s.heads / 8, JD = D / 8;
  if (s.heads % 8 || hpc * rpb > 8 || (rpb != 1 && rpb != 2 && rpb != 4) || D % 8 || D > 128 || group % hpc ||
      hpc * 8 != 16 || s.groups || (s.capacity & (s.capacity - 1)) || s.capacity < 32 || (s.slots & (s.slots - 1)))
    throw std::invalid_argument("unsupported NPU MMA attention shape");
  const std::string common = R"(  %origin = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c2 = index.constant 2 : index
  %c3 = index.constant 3 : index
  %c4 = index.constant 4 : index
  %c8 = index.constant 8 : index
  %c16 = index.constant 16 : index
  %c32 = index.constant 32 : index
  %c64 = index.constant 64 : index
  %c128 = index.constant 128 : index
  %c144 = index.constant 144 : index
  %c160 = index.constant 160 : index
  %c176 = index.constant 176 : index
  %c256 = index.constant 256 : index
  %jd = index.constant @JD : index
  %hpc = index.constant @HPC : index
  %rpb = index.constant @RPB : index
  %dim = index.constant @D : index
  %ofloats = index.constant @OF : index
  %qfloats = index.constant @Q : index
  %kvfloats = index.constant @KV : index
  %cap = index.constant @CAP : index
  %oblock = index.constant @OBLOCK : index
  %group = index.constant @GROUP : index
  %head0 = index.mul %column, %hpc : index
  %kvh = index.div %head0, %group : index
)";
  const std::string body = R"(
pipeline.def<kernel> @@NAME_col(%column: index) run(%a0: buffer, %a1: buffer, %a2: buffer, %a3: buffer, %a4: buffer, %a5: buffer, %a6: buffer, %a7: buffer) {
  %qb, %gb, %kpool, %vpool, %meta, %kgroups, %out, %cntb = buffer.assume.alignment %a0, %a1, %a2, %a3, %a4, %a5, %a6, %a7 {minimum_alignment = 64} : buffer, buffer, buffer, buffer, buffer, buffer, buffer, buffer
@COMMON
  %pool2 = pipeline.memory<workgroup>[%column, 2] : pool
  %pool3 = pipeline.memory<workgroup>[%column, 3] : pool
@SCRATCH
  %pc_size = index.constant 1536 : offset
  %pc_storage = buffer.alloca<workgroup> pool(%pool3) align(64) %pc_size : buffer
  %pc_slots = buffer.view %pc_storage[%origin] : buffer -> view<2x192xf32>
  %pc_ch = channel.bind<fifo> %pc_slots, %c2 : view<2x192xf32>, index -> channel<tile<192xf32>>

  // Scorer.
  pipeline.strand workers([%column, 2], [1, 1], [1, 1]) {
@RESERVE_A
    %ml = view.subview %ss[256] : view<352xf32> -> view<32xf32>
    %gv = view.subview %ss[288] : view<352xf32> -> view<64xf32>
    %fbits = view.subview %ss[256] : view<352xf32> -> view<32xf32>
    %mbits = view.subview %qs[0] : view<@QSZxbf16> -> view<32xbf16>
    %mrow = view.bitcast %mbits : view<32xbf16> -> view<16xi32>
    %fl0 = view.subview %ss[0] : view<352xf32> -> view<32xf32>
    %fl = view.bitcast %fl0 : view<32xf32> -> view<32xi32>
    %zero16 = vector.splatf 0.0 : vector<16xf32>
    %zero64 = vector.splatf 0.0 : vector<64xf32>
    %neg = vector.splatf -1.0e30 : vector<16xf32>
    %onev = vector.splatf 1.0 : vector<16xf32>
    %slack = vector.splatf 8.0 : vector<16xf32>
    %one_i = vector.splati 1 : vector<16xi32>
    %zero_iv = vector.splati 0 : vector<16xi32>
    %izero_s = scalar.constant 0 : i32
    %lanes = vector.constanti [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1] : vector<16xi32>
    %two = vector.splati 2 : vector<16xi32>
    @SCHEMA
    %cn_src = buffer.view %cntb[%origin] : buffer -> view<16xi32>
)" R"(@COPYCNT
    %cn_raw = view.load %mrow[%c0] : view<16xi32> -> i32
    %cn_mask = scalar.constant 255 : i32
    %cn_b = scalar.andi %cn_raw, %cn_mask : i32
    %rows = index.cast %cn_b : i32 to index
    %rows_p = index.add %rows, %rpb : index
    %rows_p1 = index.sub %rows_p, %c1 : index
    %nblocks = index.div %rows_p1, %rpb : index
    %rows_m1 = index.sub %rows, %c1 : index
    %slot_mask = scalar.constant @SLOTM : i32
    %pos_mask = scalar.constant 1048575 : i32
    %one_s = scalar.constant 1 : i32
    scf.for %blk = [%c0 to %nblocks step %c1] {
      %r0 = index.mul %blk, %rpb : index
      // First/last key per lane go to %lanes_fl (32 i32 in the score tile).
      %slot_f, %lo_f, %hi_f = scf.for %rr = [%c0 to %rpb step %c1](%sl = %izero_s : i32, %lo = %izero_s : i32, %hi = %izero_s : i32) -> (i32, i32, i32) {
        // Rows past the end reuse the last row's data with an empty key range.
        %row0 = index.add %r0, %rr : index
        %rowx = index.min %row0, %rows_m1 : index
        %row = index.assume %rowx [range(%rowx, 0, 255)] : index
        %valid = index.cmp ult, %row0, %rows : index
        %m_e = index.mul %row, %c16 : index
@OFF_M
        %m_src = buffer.view %meta[%m_at] : buffer -> view<16xi32>
@COPY_M
        %slot_raw = view.load %mrow[%c0] : view<16xi32> -> i32
        %first_raw = view.load %mrow[%c2] : view<16xi32> -> i32
        %last_raw = view.load %mrow[%c3] : view<16xi32> -> i32
        %slot_b = scalar.andi %slot_raw, %slot_mask : i32
        %first_b = scalar.andi %first_raw, %pos_mask : i32
        %last_b = scalar.andi %last_raw, %pos_mask : i32
        %q_r = index.mul %row, %qfloats : index
        %q_h = index.mul %head0, %dim : index
        %q_e = index.add %q_r, %q_h : index
@OFF_Q
        %q_dense = buffer.view %qb[%q_at] : buffer -> view<@JDx@HPCx8xf32>
        %q_src = view.refine %q_dense : view<@JDx@HPCx8xf32> -> view<@JDx@HPCx8xf32, #encoding.layout.strided<strides=[8, @D, 1]>>
        %qf_sub = view.subview %qf[%rr, 0, 0, 0] : view<@RPBx@JDx@HPCx8xf32> -> view<1x@JDx@HPCx8xf32>
        %qf_row = view.bitcast %qf_sub : view<1x@JDx@HPCx8xf32> -> view<@JDx@HPCx8xf32>
@COPY_Q
        %g_e = index.mul %row, %c16 : index
@OFF_G
        %g_src = buffer.view %gb[%g_at] : buffer -> view<16xf32>
        %rr16 = index.mul %rr, %c16 : index
        %gv_row = view.subview %gv[%rr16] : view<64xf32> -> view<16xf32>
@COPY_G
        %fr = scf.select %valid, %first_b, %one_s : i32
        %la = scf.select %valid, %last_b, %izero_s : i32
        %is0 = index.cmp eq, %rr, %c0 : index
        %sl2 = scf.select %is0, %slot_b, %sl : i32
        %lo2 = scf.select %is0, %first_b, %lo : i32
        scf.for %h = [%c0 to %hpc step %c1] {
          %q0 = index.mul %rr, %hpc : index
          %q = index.add %q0, %h : index
          %q8 = index.add %q, %c8 : index
          %q16 = index.add %q, %c16 : index
          %q24 = index.add %q8, %c16 : index
          view.store %fr, %fl[%q] : i32, view<32xi32>
          view.store %fr, %fl[%q8] : i32, view<32xi32>
          view.store %la, %fl[%q16] : i32, view<32xi32>
          view.store %la, %fl[%q24] : i32, view<32xi32>
        }
        scf.yield %sl2, %lo2, %last_b : i32, i32, i32
      }
      %first_v = vector.load %fl[%c0] : view<32xi32> -> vector<16xi32>
      %last_v = vector.load %fl[%c16] : view<32xi32> -> vector<16xi32>
      %slot = index.cast %slot_f : i32 to index
      // Q tiles [j][q][d] in BF16.
      scf.for %j = [%c0 to %jd step %c1] {
        scf.for %rr = [%c0 to %rpb step %c1] {
          %qslab = vector.load %qf[%rr, %j, %c0, %c0] : view<@RPBx@JDx@HPCx8xf32> -> vector<@HPCx8xf32>
          %qflat = vector.bitcast %qslab : vector<@HPCx8xf32> to vector<16xf32>
          %qb16 = vector.fptrunc %qflat : vector<16xf32> to vector<16xbf16>
          %j64 = index.mul %j, %c64 : index
          %rq = index.mul %rr, %c16 : index
          %qo = index.add %j64, %rq : index
          vector.store %qb16, %qs[%qo] : vector<16xbf16>, view<@QSZxbf16>
        }
      }
      %sh5 = scalar.constant 5 : i32
      %cm_i32 = scalar.constant 32767 : i32
      %fc_i = scalar.shrui %lo_f, %sh5 : i32
      %lc_i = scalar.shrui %hi_f, %sh5 : i32
      %nc0 = scalar.subi %lc_i, %fc_i : i32
      %nc1 = scalar.addi %nc0, %one_s : i32
      %nc2 = scalar.andi %nc1, %cm_i32 : i32
      %chunks = index.cast %nc2 : i32 to index
      %fcm = scalar.andi %fc_i, %cm_i32 : i32
      %first_chunk = index.cast %fcm : i32 to index
      %c_lo = index.mul %first_chunk, %c32 : index
      // Header: slot, first key, chunks.
      %hd_w, %hd = channel.reserve %pc_ch : channel<tile<192xf32>> -> (write<tile<192xf32>>, view<192xf32>)
      %hdi0 = view.subview %hd[144] : view<192xf32> -> view<16xf32>
      %hdi = view.bitcast %hdi0 : view<16xf32> -> view<16xi32>
      %c_lo_i = index.cast %c_lo : index to i32
      view.store %slot_f, %hdi[%c0] : i32, view<16xi32>
      view.store %c_lo_i, %hdi[%c1] : i32, view<16xi32>
      view.store %nc2, %hdi[%c2] : i32, view<16xi32>
      channel.publish %hd_w : write<tile<192xf32>>
      %m_fin, %l_fin = scf.for %ch = [%c0 to %chunks step %c1](%m_old = %neg : vector<16xf32>, %l_old = %zero16 : vector<16xf32>) -> (vector<16xf32>, vector<16xf32>) {
        %ch32 = index.mul %ch, %c32 : index
        %t0 = index.add %c_lo, %ch32 : index
        %t0_r = index.cast %t0 : index to i32
        // Chunks start 32-aligned: ring row (t0 / 32 mod cap / 32) * 32.
        %sh5r = scalar.constant 5 : i32
        %cchunk = scalar.shrui %t0_r, %sh5r : i32
        %capc_i = scalar.constant @CAPC : i32
        %ring_c = scalar.andi %cchunk, %capc_i : i32
        %ring_cx = index.cast %ring_c : i32 to index
        %ring_cb = index.assume %ring_cx [range(%ring_cx, 0, @CAPC)] : index
        %ring = index.mul %ring_cb, %c32 : index
        %sc = index.mul %slot, %cap : index
        %rr2 = index.add %sc, %ring : index
        %re = index.mul %rr2, %kvfloats : index
        %ho = index.mul %kvh, %dim : index
        %ke = index.add %re, %ho : index
@OFF_K
        %k_dense = buffer.view %kpool[%k_at] : buffer -> view<@JDx32x8xf16>
        %k_src = view.refine %k_dense : view<@JDx32x8xf16> -> view<@JDx32x8xf16, #encoding.layout.strided<strides=[8, @KV, 1]>>
@COPY_K
        // S^T tiles [kt][key][q].
        scf.for %kt = [%c0 to %c4 step %c1] {
          %s = scf.for %j = [%c0 to %jd step %c1](%acc = %zero64 : vector<64xf32>) -> (vector<64xf32>) {
            %qo = index.mul %j, %c64 : index
            %qv = vector.load %qs[%qo] : view<@QSZxbf16> -> vector<64xbf16>
            %qe = vector.encode %qv using %schema : vector<64xbf16>, encoding<schema> -> vector<72xi8>
            %rhs = vector.fragment<rhs> %qe shape [%c8, %c8] using {schema = %schema : encoding<schema>} : vector<72xi8>
            %kt8 = index.mul %kt, %c8 : index
            %km = vector.load %ks[%j, %kt8, %c0] : view<@JDx32x8xf16> -> vector<8x8xf16>
            %kh = vector.bitcast %km : vector<8x8xf16> to vector<64xf16>
            %kf = vector.extf %kh : vector<64xf16> to vector<64xf32>
            %kv = vector.fptrunc %kf : vector<64xf32> to vector<64xbf16>
            %ke2 = vector.encode %kv using %schema : vector<64xbf16>, encoding<schema> -> vector<72xi8>
            %lhs = vector.fragment<lhs> %ke2 shape [%c8, %c8] using {schema = %schema : encoding<schema>} : vector<72xi8>
            %init = vector.fragment<init> %acc shape [%c8, %c8] : vector<64xf32>
            %n = vector.mma %lhs, %rhs, %init : vector<72xi8>, vector<72xi8>, vector<64xf32>
            scf.yield %n : vector<64xf32>
          }
          %so = index.mul %kt, %c64 : index
          %so2 = index.add %so, %c32 : index
          vector.store %s, %ss[%so2] : vector<64xf32>, view<352xf32>
        }
        // Mask keys outside [first, last] per lane; chunk i holds keys
        // t0 + 2i (lanes 0..7) and t0 + 2i + 1 (lanes 8..15).
        %t0v = vector.splat %t0_r : vector<16xi32>
        %keys0 = vector.addi %t0v, %lanes : vector<16xi32>
        %c288 = index.constant 288 : index
        %mx, %kend = scf.for %i = [%c32 to %c288 step %c16](%acc = %neg : vector<16xf32>, %key = %keys0 : vector<16xi32>) -> (vector<16xf32>, vector<16xi32>) {
          %v0 = vector.load %ss[%i] : view<352xf32> -> vector<16xf32>
          %ge = vector.cmpi sge, %key, %first_v : vector<16xi32> -> vector<16xi1>
          %le = vector.cmpi sle, %key, %last_v : vector<16xi32> -> vector<16xi1>
          %in = vector.andi %ge, %le : vector<16xi1>
          %v = vector.select %in, %v0, %neg : vector<16xf32>
          vector.store %v, %ss[%i] : vector<16xf32>, view<352xf32>
          %gt = vector.cmpf ogt, %v, %acc : vector<16xf32> -> vector<16xi1>
          %n = vector.select %gt, %v, %acc : vector<16xf32>
          %kn = vector.addi %key, %two : vector<16xi32>
          scf.yield %n, %kn : vector<16xf32>, vector<16xi32>
        }
        %mxs = func.call @swap8(%mx) : (vector<16xf32>) -> (vector<16xf32>)
        %gt2 = vector.cmpf ogt, %mxs, %mx : vector<16xf32> -> vector<16xi1>
        %rm = vector.select %gt2, %mxs, %mx : vector<16xf32>
        %grow = vector.subf %rm, %m_old : vector<16xf32>
        %big = vector.cmpf ogt, %grow, %slack : vector<16xf32> -> vector<16xi1>
        %bigi = vector.select %big, %one_i, %zero_iv : vector<16xi32>
        %nbig = vector.reduce<addi> %bigi, %izero_s : vector<16xi32>, i32
        %need = scalar.cmpi ne, %nbig, %izero_s : i32
        %gt3 = vector.cmpf ogt, %rm, %m_old : vector<16xf32> -> vector<16xi1>
        %mn = vector.select %gt3, %rm, %m_old : vector<16xf32>
        %m_new = scf.select %need, %mn, %m_old : vector<16xf32>
        %dm = vector.subf %m_old, %m_new : vector<16xf32>
        %alpha = vector.exp2f<afn> %dm : vector<16xf32>
        %pr_w, %pr = channel.reserve %pc_ch : channel<tile<192xf32>> -> (write<tile<192xf32>>, view<192xf32>)
        %prp0 = view.subview %pr[0] : view<192xf32> -> view<128xf32>
        %prp = view.bitcast %prp0 : view<128xf32> -> view<256xbf16>
        %sum = scf.for %i = [%c0 to %c256 step %c16](%acc = %zero16 : vector<16xf32>) -> (vector<16xf32>) {
          %i32 = index.add %i, %c32 : index
          %v = vector.load %ss[%i32] : view<352xf32> -> vector<16xf32>
          %d = vector.subf %v, %m_new : vector<16xf32>
          %e = vector.exp2f<afn> %d : vector<16xf32>
          %eb = vector.fptrunc %e : vector<16xf32> to vector<16xbf16>
          vector.store %eb, %prp[%i] : vector<16xbf16>, view<256xbf16>
          %n = vector.addf %acc, %e : vector<16xf32>
          scf.yield %n : vector<16xf32>
        }
        %sums = func.call @swap8(%sum) : (vector<16xf32>) -> (vector<16xf32>)
        %rs = vector.addf %sum, %sums : vector<16xf32>
        %la = func.call @fmul2(%l_old, %alpha) : (vector<16xf32>, vector<16xf32>) -> (vector<16xf32>)
        %l_new = vector.addf %la, %rs : vector<16xf32>
        vector.store %alpha, %pr[%c128] : vector<16xf32>, view<192xf32>
        %pri0 = view.subview %pr[144] : view<192xf32> -> view<16xf32>
        %pri = view.bitcast %pri0 : view<16xf32> -> view<16xi32>
        view.store %nbig, %pri[%c0] : i32, view<16xi32>
        channel.publish %pr_w : write<tile<192xf32>>
        scf.yield %m_new, %l_new : vector<16xf32>, vector<16xf32>
      }
      // Final: l and the gate of each query lane.
      %fn_w, %fn = channel.reserve %pc_ch : channel<tile<192xf32>> -> (write<tile<192xf32>>, view<192xf32>)
      vector.store %l_fin, %fn[%c160] : vector<16xf32>, view<192xf32>
      scf.for %rr = [%c0 to %rpb step %c1] {
        scf.for %h = [%c0 to %hpc step %c1] {
          %rr16 = index.mul %rr, %c16 : index
          %gh0 = index.add %rr16, %head0 : index
          %gh = index.add %gh0, %h : index
          %gval = view.load %gv[%gh] : view<64xf32> -> f32
          %q0 = index.mul %rr, %hpc : index
          %q = index.add %q0, %h : index
          %q8 = index.add %q, %c8 : index
          %qa = index.add %q, %c176 : index
          %qb2 = index.add %q8, %c176 : index
          view.store %gval, %fn[%qa] : f32, view<192xf32>
          view.store %gval, %fn[%qb2] : f32, view<192xf32>
        }
      }
      channel.publish %fn_w : write<tile<192xf32>>
    }
@PUBLISH_A
  }

  // Accumulator.
  pipeline.strand workers([%column, 3], [1, 1], [1, 1]) {
@RESERVE_B
    %zero16 = vector.splatf 0.0 : vector<16xf32>
    %zero64 = vector.splatf 0.0 : vector<64xf32>
    %onev = vector.splatf 1.0 : vector<16xf32>
    %izero_s = scalar.constant 0 : i32
    @SCHEMA
    %cb0 = view.subview %os[0] : view<@OFxf32> -> view<16xf32>
    %cbi = view.bitcast %cb0 : view<16xf32> -> view<16xi32>
    %cn_src = buffer.view %cntb[%origin] : buffer -> view<16xi32>
@COPYCNT_B
    %cn_raw = view.load %cbi[%c0] : view<16xi32> -> i32
    %cn_mask = scalar.constant 255 : i32
    %cn_b = scalar.andi %cn_raw, %cn_mask : i32
    %rows = index.cast %cn_b : i32 to index
    %rows_p = index.add %rows, %rpb : index
    %rows_p1 = index.sub %rows_p, %c1 : index
    %nblocks = index.div %rows_p1, %rpb : index
    scf.for %blk = [%c0 to %nblocks step %c1] {
      %r0 = index.mul %blk, %rpb : index
      %hr, %hd = channel.acquire %pc_ch : channel<tile<192xf32>> -> (read<tile<192xf32>>, view<192xf32>)
      %hdi0 = view.subview %hd[144] : view<192xf32> -> view<16xf32>
      %hdi = view.bitcast %hdi0 : view<16xf32> -> view<16xi32>
      %slot_i = view.load %hdi[%c0] : view<16xi32> -> i32
      %c_lo_i = view.load %hdi[%c1] : view<16xi32> -> i32
      %nch_i = view.load %hdi[%c2] : view<16xi32> -> i32
      channel.release %hr : read<tile<192xf32>>
      %slot_mask = scalar.constant @SLOTM : i32
      %slot_b = scalar.andi %slot_i, %slot_mask : i32
      %slot = index.cast %slot_b : i32 to index
      %c_lo_x = index.cast %c_lo_i : i32 to index
      %c_lo = index.assume %c_lo_x [range(%c_lo_x, 0, 1048575)] : index
      %chunks_x = index.cast %nch_i : i32 to index
      %chunks = index.assume %chunks_x [range(%chunks_x, 0, 32767)] : index
      scf.for %i = [%c0 to %ofloats step %c64] {
        vector.store %zero64, %os[%i] : vector<64xf32>, view<@OFxf32>
      }
      scf.for %ch = [%c0 to %chunks step %c1] {
        %ch32 = index.mul %ch, %c32 : index
        %t0 = index.add %c_lo, %ch32 : index
        %t0_r = index.cast %t0 : index to i32
        // Chunks start 32-aligned: ring row (t0 / 32 mod cap / 32) * 32.
        %sh5r = scalar.constant 5 : i32
        %cchunk = scalar.shrui %t0_r, %sh5r : i32
        %capc_i = scalar.constant @CAPC : i32
        %ring_c = scalar.andi %cchunk, %capc_i : i32
        %ring_cx = index.cast %ring_c : i32 to index
        %ring_cb = index.assume %ring_cx [range(%ring_cx, 0, @CAPC)] : index
        %ring = index.mul %ring_cb, %c32 : index
        %sc = index.mul %slot, %cap : index
        %rr2 = index.add %sc, %ring : index
        %re = index.mul %rr2, %kvfloats : index
        %ho = index.mul %kvh, %dim : index
        %ke = index.add %re, %ho : index
@OFF_K
        %v_dense = buffer.view %vpool[%k_at] : buffer -> view<@JDx32x8xf16>
        %v_src = view.refine %v_dense : view<@JDx32x8xf16> -> view<@JDx32x8xf16, #encoding.layout.strided<strides=[8, @KV, 1]>>
@COPY_V
        %pr_r, %pr = channel.acquire %pc_ch : channel<tile<192xf32>> -> (read<tile<192xf32>>, view<192xf32>)
        %prp0 = view.subview %pr[0] : view<192xf32> -> view<128xf32>
        %prp = view.bitcast %prp0 : view<128xf32> -> view<256xbf16>
        %pri0 = view.subview %pr[144] : view<192xf32> -> view<16xf32>
        %pri = view.bitcast %pri0 : view<16xf32> -> view<16xi32>
        %nbig = view.load %pri[%c0] : view<16xi32> -> i32
        %need = scalar.cmpi ne, %nbig, %izero_s : i32
        scf.if %need {
          %al = vector.load %pr[%c128] : view<192xf32> -> vector<16xf32>
          scf.for %i = [%c0 to %ofloats step %c16] {
            %v = vector.load %os[%i] : view<@OFxf32> -> vector<16xf32>
            %w = func.call @fmul2(%v, %al) : (vector<16xf32>, vector<16xf32>) -> (vector<16xf32>)
            vector.store %w, %os[%i] : vector<16xf32>, view<@OFxf32>
          }
        }
        // O^T tiles [j][d][q] += V(j, kt)^T P(kt).
        scf.for %j = [%c0 to %jd step %c1] {
          %jo = index.mul %j, %c64 : index
          %o_in = vector.load %os[%jo] : view<@OFxf32> -> vector<64xf32>
          %o_new = scf.for %kt = [%c0 to %c4 step %c1](%acc = %o_in : vector<64xf32>) -> (vector<64xf32>) {
            %po = index.mul %kt, %c64 : index
            %pv = vector.load %prp[%po] : view<256xbf16> -> vector<64xbf16>
            %pm = vector.bitcast %pv : vector<64xbf16> to vector<8x8xbf16>
            %pt = vector.transpose<[1, 0]> %pm : vector<8x8xbf16> -> vector<8x8xbf16>
            %ptf = vector.bitcast %pt : vector<8x8xbf16> to vector<64xbf16>
            %pe = vector.encode %ptf using %schema : vector<64xbf16>, encoding<schema> -> vector<72xi8>
            %rhs = vector.fragment<rhs> %pe shape [%c8, %c8] using {schema = %schema : encoding<schema>} : vector<72xi8>
            %vt8 = index.mul %kt, %c8 : index
            %vm = vector.load %vs[%j, %vt8, %c0] : view<@JDx32x8xf16> -> vector<8x8xf16>
            %vh = vector.bitcast %vm : vector<8x8xf16> to vector<64xf16>
            %vf = vector.extf %vh : vector<64xf16> to vector<64xf32>
            %vb = vector.fptrunc %vf : vector<64xf32> to vector<64xbf16>
            %vbm = vector.bitcast %vb : vector<64xbf16> to vector<8x8xbf16>
            %vt = vector.transpose<[1, 0]> %vbm : vector<8x8xbf16> -> vector<8x8xbf16>
            %vtf = vector.bitcast %vt : vector<8x8xbf16> to vector<64xbf16>
            %ve = vector.encode %vtf using %schema : vector<64xbf16>, encoding<schema> -> vector<72xi8>
            %lhs = vector.fragment<lhs> %ve shape [%c8, %c8] using {schema = %schema : encoding<schema>} : vector<72xi8>
            %init = vector.fragment<init> %acc shape [%c8, %c8] : vector<64xf32>
            %n = vector.mma %lhs, %rhs, %init : vector<72xi8>, vector<72xi8>, vector<64xf32>
            scf.yield %n : vector<64xf32>
          }
          vector.store %o_new, %os[%jo] : vector<64xf32>, view<@OFxf32>
        }
        channel.release %pr_r : read<tile<192xf32>>
      }
      // o / (l (1 + e^-g)) per query lane.
      %fr, %fn = channel.acquire %pc_ch : channel<tile<192xf32>> -> (read<tile<192xf32>>, view<192xf32>)
      %l_fin = vector.load %fn[%c160] : view<192xf32> -> vector<16xf32>
      %g16 = vector.load %fn[%c176] : view<192xf32> -> vector<16xf32>
      channel.release %fr : read<tile<192xf32>>
      %nlog2e = vector.splatf -1.44269504088896341 : vector<16xf32>
      %gneg = func.call @fmul2(%g16, %nlog2e) : (vector<16xf32>, vector<16xf32>) -> (vector<16xf32>)
      %eg = vector.exp2f<afn> %gneg : vector<16xf32>
      %g1 = vector.addf %onev, %eg : vector<16xf32>
      %dg = func.call @fmul2(%l_fin, %g1) : (vector<16xf32>, vector<16xf32>) -> (vector<16xf32>)
      %inv = func.call @recip(%dg) : (vector<16xf32>) -> (vector<16xf32>)
      // Staging [rr][h][j][d] in BF16: groups (head0 + h) * JD + j of row rr.
      scf.for %j = [%c0 to %jd step %c1] {
        %jo = index.mul %j, %c64 : index
        scf.for %i = [%c0 to %c64 step %c16] {
          %oi = index.add %jo, %i : index
          %v = vector.load %os[%oi] : view<@OFxf32> -> vector<16xf32>
          %w = func.call @fmul2(%v, %inv) : (vector<16xf32>, vector<16xf32>) -> (vector<16xf32>)
          %wb = vector.fptrunc %w : vector<16xf32> to vector<16xbf16>
          vector.store %wb, %tb[%i] : vector<16xbf16>, view<64xbf16>
        }
        %ofb = vector.load %tb[%c0] : view<64xbf16> -> vector<64xbf16>
        %ofbm = vector.bitcast %ofb : vector<64xbf16> to vector<8x8xbf16>
        %oft = vector.transpose<[1, 0]> %ofbm : vector<8x8xbf16> -> vector<8x8xbf16>
        %oftf = vector.bitcast %oft : vector<8x8xbf16> to vector<64xbf16>
        vector.store %oftf, %tb[%c0] : vector<64xbf16>, view<64xbf16>
        %j8 = index.mul %j, %c8 : index
        scf.for %q = [%c0 to %c8 step %c1] {
          %q8 = index.mul %q, %c8 : index
          %hrow = vector.load %tb[%q8] : view<64xbf16> -> vector<8xbf16>
          %dst0 = index.mul %q, %dim : index
          %dst = index.add %dst0, %j8 : index
          vector.store %hrow, %st[%dst] : vector<8xbf16>, view<@STxbf16>
        }
      }
      scf.for %rr = [%c0 to %rpb step %c1] {
        %row = index.add %r0, %rr : index
        %valid = index.cmp ult, %row, %rows : index
        scf.if %valid {
          %ob_blk = index.div %row, %c8 : index
          %ob_b = index.rem %row, %c8 : index
          %ob0 = index.mul %ob_blk, %oblock : index
          %ob1 = index.mul %head0, %dim : index
          %ob2 = index.mul %ob1, %c8 : index
          %ob3 = index.mul %ob_b, %c8 : index
          %ob4 = index.add %ob0, %ob2 : index
          %ob5 = index.add %ob4, %ob3 : index
@OFF_O
          %o_dense = buffer.view %out[%o_at] : buffer -> view<@OGx64xbf16>
          %o_str = view.refine %o_dense : view<@OGx64xbf16> -> view<@OGx64xbf16, #encoding.layout.strided<strides=[64, 1]>>
          %o_glob = view.subview %o_str[0, 0] : view<@OGx64xbf16, #encoding.layout.strided<strides=[64, 1]>> -> view<@OGx8xbf16, #encoding.layout.strided<strides=[64, 1]>>
          %st0 = index.mul %rr, %c@ROWOUT : index
          %stv = view.subview %st[%st0] : view<@STxbf16> -> view<@ROWOUTxbf16>
          %o_shaped = view.bitcast %stv : view<@ROWOUTxbf16> -> view<@OGx8xbf16>
@COPY_O
        }
      }
    }
@PUBLISH_B
  }
}
)";
  const std::uint32_t OF = JD * 64, QS = JD * 64, OG = hpc * D / 8, ST = 8 * D;
  struct Rec { const char* name; std::string shape; std::uint64_t bytes; bool scorer; };
  const std::vector<Rec> recs = {
      {"qf", S(rpb) + "x" + S(JD) + "x" + S(hpc) + "x8xf32", std::uint64_t(rpb) * JD * hpc * 8 * 4, true},
      {"qs", S(QS) + "xbf16", QS * 2, true},
      {"ks", S(JD) + "x32x8xf16", JD * 32 * 8 * 2, true},
      {"ss", "352xf32", 352 * 4, true},
      {"vs", S(JD) + "x32x8xf16", JD * 32 * 8 * 2, false},
      {"os", S(OF) + "xf32", OF * 4, false},
      {"st", S(ST) + "xbf16", ST * 2, false},
      {"tb", "64xbf16", 128, false}};
  std::string scratch, reserve_a, publish_a, reserve_b, publish_b;
  for (const auto& r : recs) {
    std::string sc = Scratch(r.name, r.shape, r.bytes);
    sc = Fill(sc, {{"pool(%memory)", r.scorer ? "pool(%pool2)" : "pool(%pool3)"}});
    scratch += sc;
    (r.scorer ? reserve_a : reserve_b) += Reserve(r.name, r.shape);
    (r.scorer ? publish_a : publish_b) += Publish(r.name, r.shape);
  }
  const std::string qf_t = "view<" + S(JD) + "x" + S(hpc) + "x8xf32>";
  const std::string kv_t = "view<" + S(JD) + "x32x8xf16>";
  const std::string kv_src_t = "view<" + S(JD) + "x32x8xf16, #encoding.layout.strided<strides=[8, " + S(KV) + ", 1]>>";
  std::string text = Fill(body, {
      {"@COMMON", common}, {"@SCRATCH", scratch}, {"@RESERVE_A", reserve_a}, {"@PUBLISH_A", publish_a},
      {"@RESERVE_B", reserve_b}, {"@PUBLISH_B", publish_b},
      {"@COPYCNT_B", Copy("cn", "cn_src", "view<16xi32>", "cbi", "view<16xi32>", true)},
      {"@COPYCNT", Copy("cn", "cn_src", "view<16xi32>", "mrow", "view<16xi32>", true)},
      {"@OFF_M", Offset("m", "%m_e", 4)}, {"@COPY_M", Copy("mm", "m_src", "view<16xi32>", "mrow", "view<16xi32>", true)},
      {"@OFF_Q", Offset("q", "%q_e", 4)},
      {"@COPY_Q", Copy("qq", "q_src", "view<" + S(JD) + "x" + S(hpc) + "x8xf32, #encoding.layout.strided<strides=[8, @D, 1]>>", "qf_row", qf_t, true)},
      {"@OFF_G", Offset("g", "%g_e", 4)}, {"@COPY_G", Copy("gg", "g_src", "view<16xf32>", "gv_row", "view<16xf32>", true)},
      {"@OFF_K", Offset("k", "%ke", 2)}, {"@OFF_O", Offset("o", "%ob5", 2)},
      {"@COPY_K", Copy("kk", "k_src", kv_src_t, "ks", kv_t, true)},
      {"@COPY_V", Copy("vv", "v_src", kv_src_t, "vs", kv_t, true)},
      {"@COPY_O", Copy("oo", "o_shaped", "view<@OGx8xbf16>", "o_glob", "view<@OGx8xbf16, #encoding.layout.strided<strides=[64, 1]>>", false)},
  });
  text = Fill(text, {{"@NAME", name}, {"@QSZ", S(QS)}, {"@Q", S(Q)}, {"@KV", S(KV)}, {"@CAPM", S(s.capacity - 1)}, {"@CAPC", S(s.capacity / 32 - 1)},
                     {"@CAP", S(s.capacity)}, {"@OBLOCK", S(std::uint64_t(Q / 8) * 64)}, {"@SLOTM", S(s.slots - 1)},
                     {"@JD", S(JD)}, {"@HPC", S(hpc)}, {"@RPB", S(rpb)}, {"@D", S(D)}, {"@OF", S(OF)},
                     {"@OG", S(OG)}, {"@ROWOUT", S(hpc * D)}, {"@ST", S(ST)}, {"@GROUP", S(group)},
                     {"@SCHEMA", "%schema = encoding.define #encoding.operand<element_format=bfp16ebs8, payload_elements=64, "
                                 "payload_packing=target_fragment, payload_registers=18, rounding=flush_subnormal> : encoding<schema>"}});
  std::string head = "aie2p.target<array> @npu\n";
  head += kMathLibrary;
  head += R"(
func.def @swap8(%x: vector<16xf32>) -> (vector<16xf32>) {
  %lo = vector.slice %x[0] : vector<16xf32> -> vector<8xf32>
  %hi = vector.slice %x[8] : vector<16xf32> -> vector<8xf32>
  %r = vector.concat<0> %hi, %lo : vector<8xf32>, vector<8xf32> -> vector<16xf32>
  func.return %r : vector<16xf32>
}
)";
  return head + text +
         Compose(name, "%a0: buffer, %a1: buffer, %a2: buffer, %a3: buffer, %a4: buffer, %a5: buffer, %a6: buffer, %a7: buffer",
                 "%a0, %a1, %a2, %a3, %a4, %a5, %a6, %a7", "buffer, buffer, buffer, buffer, buffer, buffer, buffer, buffer");
}

}  // namespace gufo::models::lfm3::npu
