#include "attention.hpp"

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
  (void)s;
  (void)name;
  throw std::logic_error("attention kernel not yet generated");
}

}  // namespace gufo::models::lfm3::npu
