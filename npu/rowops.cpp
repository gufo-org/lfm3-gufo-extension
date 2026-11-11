#include "rowops.hpp"

#include <set>
#include <stdexcept>

namespace gufo::models::lfm3::npu {

const char* const kMathLibrary = R"LOOM(
// a * b per lane to about F32 precision: each operand splits into three
// BF16 parts; the six leading cross products accumulate, smallest first.
func.def @fmul(%a: vector<16xf32>, %b: vector<16xf32>) -> (vector<16xf32>) {
  %a1 = vector.fptrunc %a : vector<16xf32> to vector<16xbf16>
  %a1f = vector.extf %a1 : vector<16xbf16> to vector<16xf32>
  %ar = vector.subf %a, %a1f : vector<16xf32>
  %a2 = vector.fptrunc %ar : vector<16xf32> to vector<16xbf16>
  %a2f = vector.extf %a2 : vector<16xbf16> to vector<16xf32>
  %ar2 = vector.subf %ar, %a2f : vector<16xf32>
  %a3 = vector.fptrunc %ar2 : vector<16xf32> to vector<16xbf16>
  %b1 = vector.fptrunc %b : vector<16xf32> to vector<16xbf16>
  %b1f = vector.extf %b1 : vector<16xbf16> to vector<16xf32>
  %br = vector.subf %b, %b1f : vector<16xf32>
  %b2 = vector.fptrunc %br : vector<16xf32> to vector<16xbf16>
  %b2f = vector.extf %b2 : vector<16xbf16> to vector<16xf32>
  %br2 = vector.subf %br, %b2f : vector<16xf32>
  %b3 = vector.fptrunc %br2 : vector<16xf32> to vector<16xbf16>
  %zero = vector.splatf 0.0 : vector<16xf32>
  %l0 = vector.interleave<0> %a1, %a2 : vector<16xbf16>, vector<16xbf16> -> vector<32xbf16>
  %r0 = vector.interleave<0> %b3, %b2 : vector<16xbf16>, vector<16xbf16> -> vector<32xbf16>
  %s0 = vector.dot2f %l0, %r0, %zero : vector<32xbf16>, vector<32xbf16>, vector<16xf32>
  %l1 = vector.interleave<0> %a3, %a1 : vector<16xbf16>, vector<16xbf16> -> vector<32xbf16>
  %r1 = vector.interleave<0> %b1, %b2 : vector<16xbf16>, vector<16xbf16> -> vector<32xbf16>
  %s1 = vector.dot2f %l1, %r1, %s0 : vector<32xbf16>, vector<32xbf16>, vector<16xf32>
  %l2 = vector.interleave<0> %a2, %a1 : vector<16xbf16>, vector<16xbf16> -> vector<32xbf16>
  %r2 = vector.interleave<0> %b1, %b1 : vector<16xbf16>, vector<16xbf16> -> vector<32xbf16>
  %s2 = vector.dot2f %l2, %r2, %s1 : vector<32xbf16>, vector<32xbf16>, vector<16xf32>
  func.return %s2 : vector<16xf32>
}

// a * b to about 16 significant bits: two BF16 parts each, three cross
// products (for values stored as F16/BF16 next).
func.def @fmul2(%a: vector<16xf32>, %b: vector<16xf32>) -> (vector<16xf32>) {
  %a1 = vector.fptrunc %a : vector<16xf32> to vector<16xbf16>
  %a1f = vector.extf %a1 : vector<16xbf16> to vector<16xf32>
  %ar = vector.subf %a, %a1f : vector<16xf32>
  %a2 = vector.fptrunc %ar : vector<16xf32> to vector<16xbf16>
  %b1 = vector.fptrunc %b : vector<16xf32> to vector<16xbf16>
  %b1f = vector.extf %b1 : vector<16xbf16> to vector<16xf32>
  %br = vector.subf %b, %b1f : vector<16xf32>
  %b2 = vector.fptrunc %br : vector<16xf32> to vector<16xbf16>
  %zero = vector.splatf 0.0 : vector<16xf32>
  %l0 = vector.interleave<0> %a1, %a2 : vector<16xbf16>, vector<16xbf16> -> vector<32xbf16>
  %r0 = vector.interleave<0> %b2, %b1 : vector<16xbf16>, vector<16xbf16> -> vector<32xbf16>
  %s0 = vector.dot2f %l0, %r0, %zero : vector<32xbf16>, vector<32xbf16>, vector<16xf32>
  %zb = vector.splatf 0.0 : vector<16xbf16>
  %l1 = vector.interleave<0> %a1, %zb : vector<16xbf16>, vector<16xbf16> -> vector<32xbf16>
  %r1 = vector.interleave<0> %b1, %zb : vector<16xbf16>, vector<16xbf16> -> vector<32xbf16>
  %s1 = vector.dot2f %l1, %r1, %s0 : vector<32xbf16>, vector<32xbf16>, vector<16xf32>
  func.return %s1 : vector<16xf32>
}

// x to F16 with its sign: AIE2P's vector.fptrunc to f16 drops the sign
// bit (Loom at gufo-staging d8345774e3), so it is restored from x.
func.def @tof16(%x: vector<16xf32>) -> (vector<16xf16>) {
  %h = vector.fptrunc %x : vector<16xf32> to vector<16xf16>
  %hb = vector.bitcast %h : vector<16xf16> to vector<16xi16>
  %hw = vector.extui %hb : vector<16xi16> to vector<16xi32>
  %bits = vector.bitcast %x : vector<16xf32> to vector<16xi32>
  %c16 = vector.splati 16 : vector<16xi32>
  %mask = vector.splati 32768 : vector<16xi32>
  %top = vector.shrui %bits, %c16 : vector<16xi32>
  %sign = vector.andi %top, %mask : vector<16xi32>
  %signed = vector.ori %hw, %sign : vector<16xi32>
  %narrow = vector.trunci %signed : vector<16xi32> to vector<16xi16>
  %out = vector.bitcast %narrow : vector<16xi16> to vector<16xf16>
  func.return %out : vector<16xf16>
}

// rsqrt, recip and exp multiply with @fmul2 (about 16 significant bits).
// 1 / sqrt(x) for x > 0: bit estimate and three Newton steps.
func.def @rsqrt(%x: vector<16xf32>) -> (vector<16xf32>) {
  %magic = vector.splati 1597463007 : vector<16xi32>
  %one_i = vector.splati 1 : vector<16xi32>
  %half = vector.splatf 0.5 : vector<16xf32>
  %three_halves = vector.splatf 1.5 : vector<16xf32>
  %bits = vector.bitcast %x : vector<16xf32> to vector<16xi32>
  %shifted = vector.shrui %bits, %one_i : vector<16xi32>
  %guess = vector.subi %magic, %shifted : vector<16xi32>
  %y0 = vector.bitcast %guess : vector<16xi32> to vector<16xf32>
  %hx = func.call @fmul2(%x, %half) : (vector<16xf32>, vector<16xf32>) -> (vector<16xf32>)
  %a0 = func.call @fmul2(%y0, %y0) : (vector<16xf32>, vector<16xf32>) -> (vector<16xf32>)
  %b0 = func.call @fmul2(%hx, %a0) : (vector<16xf32>, vector<16xf32>) -> (vector<16xf32>)
  %c0 = vector.subf %three_halves, %b0 : vector<16xf32>
  %y1 = func.call @fmul2(%y0, %c0) : (vector<16xf32>, vector<16xf32>) -> (vector<16xf32>)
  %a1 = func.call @fmul2(%y1, %y1) : (vector<16xf32>, vector<16xf32>) -> (vector<16xf32>)
  %b1 = func.call @fmul2(%hx, %a1) : (vector<16xf32>, vector<16xf32>) -> (vector<16xf32>)
  %c1 = vector.subf %three_halves, %b1 : vector<16xf32>
  %y2 = func.call @fmul2(%y1, %c1) : (vector<16xf32>, vector<16xf32>) -> (vector<16xf32>)
  func.return %y2 : vector<16xf32>
}

// 1 / x for finite nonzero x: bit estimate and three Newton steps.
func.def @recip(%x: vector<16xf32>) -> (vector<16xf32>) {
  %magic = vector.splati 2129859559 : vector<16xi32>
  %two = vector.splatf 2.0 : vector<16xf32>
  %bits = vector.bitcast %x : vector<16xf32> to vector<16xi32>
  %guess = vector.subi %magic, %bits : vector<16xi32>
  %y0 = vector.bitcast %guess : vector<16xi32> to vector<16xf32>
  %a0 = func.call @fmul2(%x, %y0) : (vector<16xf32>, vector<16xf32>) -> (vector<16xf32>)
  %b0 = vector.subf %two, %a0 : vector<16xf32>
  %y1 = func.call @fmul2(%y0, %b0) : (vector<16xf32>, vector<16xf32>) -> (vector<16xf32>)
  %a1 = func.call @fmul2(%x, %y1) : (vector<16xf32>, vector<16xf32>) -> (vector<16xf32>)
  %b1 = vector.subf %two, %a1 : vector<16xf32>
  %y2 = func.call @fmul2(%y1, %b1) : (vector<16xf32>, vector<16xf32>) -> (vector<16xf32>)
  %a2 = func.call @fmul2(%x, %y2) : (vector<16xf32>, vector<16xf32>) -> (vector<16xf32>)
  %b2 = vector.subf %two, %a2 : vector<16xf32>
  %y3 = func.call @fmul2(%y2, %b2) : (vector<16xf32>, vector<16xf32>) -> (vector<16xf32>)
  func.return %y3 : vector<16xf32>
}

// e^x: x log2(e) = n + f with round-to-nearest n, a degree-6 polynomial of
// 2^f on [-0.5, 0.5] by Horner, then 2^n added to the exponent bits.
// Inputs below -87 give 0; above 88 saturate.
func.def @exp(%x: vector<16xf32>) -> (vector<16xf32>) {
  %low = vector.splatf -87.0 : vector<16xf32>
  %high = vector.splatf 88.0 : vector<16xf32>
  // Compare and select: AIE2P lowers F32 maxnumf/minnumf one lane at a time.
  %is_low = vector.cmpf olt, %x, %low : vector<16xf32> -> vector<16xi1>
  %xc0 = vector.select %is_low, %low, %x : vector<16xf32>
  %is_high = vector.cmpf ogt, %xc0, %high : vector<16xf32> -> vector<16xi1>
  %xc = vector.select %is_high, %high, %xc0 : vector<16xf32>
  %log2e = vector.splatf 1.44269504088896341 : vector<16xf32>
  %t = func.call @fmul(%xc, %log2e) : (vector<16xf32>, vector<16xf32>) -> (vector<16xf32>)
  %magic = vector.splatf 12582912.0 : vector<16xf32>
  %biased = vector.addf %t, %magic : vector<16xf32>
  %n = vector.subf %biased, %magic : vector<16xf32>
  %f = vector.subf %t, %n : vector<16xf32>
  %k6 = vector.splatf 1.5403530393381608e-4 : vector<16xf32>
  %k5 = vector.splatf 1.3333558146428443e-3 : vector<16xf32>
  %k4 = vector.splatf 9.6181291076284772e-3 : vector<16xf32>
  %k3 = vector.splatf 5.5504108664821580e-2 : vector<16xf32>
  %k2 = vector.splatf 0.24022650695910071 : vector<16xf32>
  %k1 = vector.splatf 0.69314718055994531 : vector<16xf32>
  %k0 = vector.splatf 1.0 : vector<16xf32>
  %p6 = func.call @fmul2(%k6, %f) : (vector<16xf32>, vector<16xf32>) -> (vector<16xf32>)
  %q5 = vector.addf %p6, %k5 : vector<16xf32>
  %p5 = func.call @fmul2(%q5, %f) : (vector<16xf32>, vector<16xf32>) -> (vector<16xf32>)
  %q4 = vector.addf %p5, %k4 : vector<16xf32>
  %p4 = func.call @fmul2(%q4, %f) : (vector<16xf32>, vector<16xf32>) -> (vector<16xf32>)
  %q3 = vector.addf %p4, %k3 : vector<16xf32>
  %p3 = func.call @fmul2(%q3, %f) : (vector<16xf32>, vector<16xf32>) -> (vector<16xf32>)
  %q2 = vector.addf %p3, %k2 : vector<16xf32>
  %p2 = func.call @fmul2(%q2, %f) : (vector<16xf32>, vector<16xf32>) -> (vector<16xf32>)
  %q1 = vector.addf %p2, %k1 : vector<16xf32>
  %p1 = func.call @fmul2(%q1, %f) : (vector<16xf32>, vector<16xf32>) -> (vector<16xf32>)
  %poly = vector.addf %p1, %k0 : vector<16xf32>
  %bias = vector.splati 1262485504 : vector<16xi32>
  %biased_bits = vector.bitcast %biased : vector<16xf32> to vector<16xi32>
  %exponent = vector.subi %biased_bits, %bias : vector<16xi32>
  %shift = vector.splati 23 : vector<16xi32>
  %scale_bits = vector.shli %exponent, %shift : vector<16xi32>
  %poly_bits = vector.bitcast %poly : vector<16xf32> to vector<16xi32>
  %result_bits = vector.addi %poly_bits, %scale_bits : vector<16xi32>
  %result = vector.bitcast %result_bits : vector<16xi32> to vector<16xf32>
  %zero = vector.splatf 0.0 : vector<16xf32>
  %under = vector.cmpf olt, %x, %low : vector<16xf32> -> vector<16xi1>
  %out = vector.select %under, %zero, %result : vector<16xf32>
  func.return %out : vector<16xf32>
}

// e^x through the native exp2 (approximate-function lowering, about
// BF16 accuracy; attention probabilities only): inputs
// below -87 give 0.
func.def @exp_fast(%x: vector<16xf32>) -> (vector<16xf32>) {
  %low = vector.splatf -87.0 : vector<16xf32>
  %is_low = vector.cmpf olt, %x, %low : vector<16xf32> -> vector<16xi1>
  %xc = vector.select %is_low, %low, %x : vector<16xf32>
  %log2e = vector.splatf 1.44269504088896341 : vector<16xf32>
  %t = func.call @fmul(%xc, %log2e) : (vector<16xf32>, vector<16xf32>) -> (vector<16xf32>)
  %r = vector.exp2f<afn> %t : vector<16xf32>
  %zero = vector.splatf 0.0 : vector<16xf32>
  %under = vector.cmpf olt, %x, %low : vector<16xf32> -> vector<16xi1>
  %out = vector.select %under, %zero, %r : vector<16xf32>
  func.return %out : vector<16xf32>
}

// 1 / (1 + e^-x).
func.def @sigmoid(%x: vector<16xf32>) -> (vector<16xf32>) {
  %zero = vector.splatf 0.0 : vector<16xf32>
  %one = vector.splatf 1.0 : vector<16xf32>
  %neg = vector.subf %zero, %x : vector<16xf32>
  %e = func.call @exp(%neg) : (vector<16xf32>) -> (vector<16xf32>)
  %d = vector.addf %one, %e : vector<16xf32>
  %r = func.call @recip(%d) : (vector<16xf32>) -> (vector<16xf32>)
  func.return %r : vector<16xf32>
}
)LOOM";

namespace {
std::string S(std::uint64_t v) { return std::to_string(v); }
}  // namespace

std::string RowKernelProgram(const RowKernel& k) {
  using Layout = RowOperand::Layout;
  for (const auto* list : {&k.inputs, &k.outputs})
    for (const auto& o : *list)
      if (o.floats % 16 || o.binding >= k.bindings ||
          (o.layout != Layout::kRows && (o.floats % 8 || !o.groups)))
        throw std::invalid_argument("unsupported NPU row operand");
  std::string args, assume_in, assume_out, types;
  const std::uint32_t nb = k.bindings + (k.dynamic_rows ? 1 : 0);
  for (std::uint32_t b = 0; b < nb; ++b) {
    args += (b ? ", " : "") + std::string("%a") + S(b) + ": buffer";
    assume_in += (b ? ", " : "") + std::string("%a") + S(b);
    assume_out += (b ? ", " : "") + std::string("%b") + S(b);
    types += (b ? ", " : "") + std::string("buffer");
  }
  std::string decl, ingress, egress, acquire, release, publish;
  auto elem = [](const RowOperand& o) { return std::string(o.bf16 ? "bf16" : "f32"); };
  auto bytes = [](const RowOperand& o) { return o.bf16 ? 2u : 4u; };
  auto parts = [](const RowOperand& o) { return o.layout == Layout::kPartial ? 8u : 1u; };
  // Tile (body) view of a piece.
  auto tile_type = [&](const RowOperand& o) {
    return o.layout == Layout::kPartial ? "8x" + S(o.floats) + "x" + elem(o) : S(o.floats) + "x" + elem(o);
  };
  auto chan = [&](const RowOperand& o) { return "tile<" + tile_type(o) + ">"; };
  auto view = [&](const RowOperand& o) { return "view<" + tile_type(o) + ">"; };
  // Statements computing %<tag>_at (byte offset of the piece's first
  // element) and the global view %<tag>_global of type returned in *type,
  // plus the tile-side view %<tag>_tile_<suffix> matching it.
  auto global = [&](const RowOperand& o, const std::string& tag, const std::string& buffer,
                    std::string* type, std::string* tile_cast, const std::string& tile) {
    std::string s;
    const std::uint32_t gp = o.floats / 8;
    if (o.layout == Layout::kRows) {
      s += "      %" + tag + "_r = index.mul %row, %" + tag + "_stride : index\n"
           "      %" + tag + "_p = index.mul %piece, %" + tag + "_floats : index\n"
           "      %" + tag + "_a = index.add %" + tag + "_r, %" + tag + "_p : index\n"
           "      %" + tag + "_e = index.add %" + tag + "_a, %" + tag + "_offset : index\n";
    } else {
      // block * (parts * groups * 64) + piece * gp * 64 + (row % 8) * 8 + offset
      s += "      %" + tag + "_blk = index.div %row, %c8 : index\n"
           "      %" + tag + "_b = index.rem %row, %c8 : index\n"
           "      %" + tag + "_bs = index.mul %" + tag + "_blk, %" + tag + "_block : index\n"
           "      %" + tag + "_ps = index.mul %piece, %" + tag + "_piece : index\n"
           "      %" + tag + "_bb = index.mul %" + tag + "_b, %c8 : index\n"
           "      %" + tag + "_a0 = index.add %" + tag + "_bs, %" + tag + "_ps : index\n"
           "      %" + tag + "_a1 = index.add %" + tag + "_a0, %" + tag + "_bb : index\n"
           "      %" + tag + "_e = index.add %" + tag + "_a1, %" + tag + "_offset : index\n";
    }
    s += "      %" + tag + "_bytes = index.mul %" + tag + "_e, %" + tag + "_esize : index\n"
         "      %" + tag + "_at = index.cast %" + tag + "_bytes : index to offset\n";
    if (o.layout == Layout::kRows) {
      *type = view(o);
      *tile_cast = "";
      s += "      %" + tag + "_global = buffer.view %" + buffer + "[%" + tag + "_at] : buffer -> " + *type + "\n";
      return s;
    }
    const std::string e = elem(o);
    if (o.layout == Layout::kBlocked) {
      const std::string dense = "view<" + S(gp) + "x64x" + e + ">";
      const std::string strided = "view<" + S(gp) + "x64x" + e + ", #encoding.layout.strided<strides=[64, 1]>>";
      *type = "view<" + S(gp) + "x8x" + e + ", #encoding.layout.strided<strides=[64, 1]>>";
      s += "      %" + tag + "_dense = buffer.view %" + buffer + "[%" + tag + "_at] : buffer -> " + dense + "\n"
           "      %" + tag + "_strided = view.refine %" + tag + "_dense : " + dense + " -> " + strided + "\n"
           "      %" + tag + "_global = view.subview %" + tag + "_strided[0, 0] : " + strided + " -> " + *type + "\n";
      *tile_cast = "      %" + tag + "_shaped = view.bitcast %" + tile + " : " + view(o) + " -> view<" + S(gp) + "x8x" + e + ">\n";
    } else {
      const std::string g = S(o.groups);
      const std::string dense = "view<8x" + g + "x64x" + e + ">";
      const std::string st = "#encoding.layout.strided<strides=[" + S(std::uint64_t(o.groups) * 64) + ", 64, 1]>";
      const std::string strided = "view<8x" + g + "x64x" + e + ", " + st + ">";
      *type = "view<8x" + S(gp) + "x8x" + e + ", " + st + ">";
      s += "      %" + tag + "_dense = buffer.view %" + buffer + "[%" + tag + "_at] : buffer -> " + dense + "\n"
           "      %" + tag + "_strided = view.refine %" + tag + "_dense : " + dense + " -> " + strided + "\n"
           "      %" + tag + "_global = view.subview %" + tag + "_strided[0, 0, 0] : " + strided + " -> " + *type + "\n";
      *tile_cast = "      %" + tag + "_shaped = view.bitcast %" + tile + " : " + view(o) + " -> view<8x" + S(gp) + "x8x" + e + ">\n";
    }
    return s;
  };
  auto consts = [&](const RowOperand& o, const std::string& tag) {
    return "  %" + tag + "_stride = index.constant " + S(o.stride) + " : index\n" +
           "  %" + tag + "_floats = index.constant " + S(o.floats) + " : index\n" +
           "  %" + tag + "_offset = index.constant " + S(o.offset) + " : index\n" +
           "  %" + tag + "_esize = index.constant " + S(bytes(o)) + " : index\n" +
           "  %" + tag + "_block = index.constant " + S(std::uint64_t(parts(o)) * o.groups * 64) + " : index\n" +
           "  %" + tag + "_piece = index.constant " + S(std::uint64_t(o.floats / 8) * 64) + " : index\n";
  };
  auto storage = [&](const RowOperand& o, const std::string& tag) {
    return "  %" + tag + "_size = index.constant " + S(std::uint64_t(parts(o)) * o.floats * bytes(o)) + " : offset\n"
           "  %" + tag + "_storage = buffer.alloca<workgroup> pool(%memory) align(64) %" + tag + "_size : buffer\n"
           "  %" + tag + "_slots = buffer.view %" + tag + "_storage[%origin] : buffer -> view<1x" + tile_type(o) + ">\n"
           "  %" + tag + "_ch = channel.bind<fifo> %" + tag + "_slots, %c1 : view<1x" + tile_type(o) + ">, index -> channel<" + chan(o) + ">\n";
  };
  auto copy = [&](const std::string& tag, const std::string& from, const std::string& from_type,
                  const std::string& to, const std::string& to_type, bool in) {
    return "      %" + tag + "_cp = kernel.async.copy %" + from + " to %" + to +
           " {cache_scope = workgroup, cache_temporal = regular, direction = " +
           (in ? "global_to_workgroup" : "workgroup_to_global") + "} : " + from_type + " to " + to_type +
           " -> kernel.async.token\n"
           "      %" + tag + "_g = kernel.async.group %" + tag + "_cp : kernel.async.token -> kernel.async.group\n"
           "      kernel.async.wait %" + tag + "_g {newer_groups = 0} : kernel.async.group\n";
  };
  for (const auto& o : k.inputs) {
    const std::string tag = "in_" + o.name;
    decl += consts(o, tag) + storage(o, tag);
    std::string gtype, cast;
    std::string g = global(o, tag, "b" + S(o.binding), &gtype, &cast, tag + "_dst");
    ingress += g + "      %" + tag + "_w, %" + tag + "_dst = channel.reserve %" + tag + "_ch : channel<" + chan(o) +
               "> -> (write<" + chan(o) + ">, " + view(o) + ")\n" + cast;
    const std::string dst = cast.empty() ? tag + "_dst" : tag + "_shaped";
    const std::string dst_type = cast.empty() ? view(o) : cast.substr(cast.rfind("-> ") + 3, cast.size() - cast.rfind("-> ") - 4);
    ingress += copy(tag, tag + "_global", gtype, dst, dst_type, true) +
               "      channel.publish %" + tag + "_w : write<" + chan(o) + ">\n";
    acquire += "      %" + tag + "_rd, %" + o.name + " = channel.acquire %" + tag + "_ch : channel<" + chan(o) + "> -> (read<" + chan(o) + ">, " + view(o) + ")\n";
    release += "      channel.release %" + tag + "_rd : read<" + chan(o) + ">\n";
  }
  for (const auto& o : k.outputs) {
    const std::string tag = "out_" + o.name;
    decl += consts(o, tag) + storage(o, tag);
    acquire += "      %" + tag + "_wr, %" + o.name + " = channel.reserve %" + tag + "_ch : channel<" + chan(o) + "> -> (write<" + chan(o) + ">, " + view(o) + ")\n";
    publish += "      channel.publish %" + tag + "_wr : write<" + chan(o) + ">\n";
    std::string gtype, cast;
    std::string g = global(o, tag, "b" + S(o.binding), &gtype, &cast, tag + "_src");
    egress += g + "      %" + tag + "_rd, %" + tag + "_src = channel.acquire %" + tag + "_ch : channel<" + chan(o) +
              "> -> (read<" + chan(o) + ">, " + view(o) + ")\n" + cast;
    const std::string src = cast.empty() ? tag + "_src" : tag + "_shaped";
    const std::string src_type = cast.empty() ? view(o) : cast.substr(cast.rfind("-> ") + 3, cast.size() - cast.rfind("-> ") - 4);
    egress += copy(tag, src, src_type, tag + "_global", gtype, false) +
              "      channel.release %" + tag + "_rd : read<" + chan(o) + ">\n";
  }
  // Each strand reads the runtime row count into its own tile.
  // The compute tile has no DMA of its own: ingress delivers its count.
  auto count = [&](const std::string& tile) {
    if (!k.dynamic_rows) return std::string();
    const std::string b = "%b" + S(k.bindings);
    auto load = [&](const std::string& ch, const std::string& v) {
      return "    %" + v + "_w, %" + v + " = channel.reserve %" + ch + " : channel<tile<16xi32>> -> (write<tile<16xi32>>, view<16xi32>)\n"
             "    %" + v + "_src = buffer.view " + b + "[%origin] : buffer -> view<16xi32>\n"
             "    %" + v + "_cp = kernel.async.copy %" + v + "_src to %" + v + " {cache_scope = workgroup, cache_temporal = regular, direction = global_to_workgroup} : view<16xi32> to view<16xi32> -> kernel.async.token\n"
             "    %" + v + "_g = kernel.async.group %" + v + "_cp : kernel.async.token -> kernel.async.group\n"
             "    kernel.async.wait %" + v + "_g {newer_groups = 0} : kernel.async.group\n";
    };
    std::string s;
    if (tile == "i")
      s = load("cnt_c_ch", "cntc") + "    channel.publish %cntc_w : write<tile<16xi32>>\n" + load("cnt_o_ch", "cnto") +
          "    channel.publish %cnto_w : write<tile<16xi32>>\n" + load("cnt_i_ch", "cnt");
    else
      s = "    %cnt_r, %cnt = channel.acquire %cnt_" + tile + "_ch : channel<tile<16xi32>> -> (read<tile<16xi32>>, view<16xi32>)\n";
    return s +
           "    %cnt_raw = view.load %cnt[%c0] : view<16xi32> -> i32\n"
           "    %cnt_mask = scalar.constant 255 : i32\n"
           "    %cnt_b = scalar.andi %cnt_raw, %cnt_mask : i32\n"
           "    %col_i = index.cast %column : index to i32\n"
           "    %seven_i = scalar.constant 7 : i32\n"
           "    %three_i = scalar.constant 3 : i32\n"
           "    %seg_i = scalar.constant " + S(k.segments) + " : i32\n"
           "    %items_i = scalar.muli %cnt_b, %seg_i : i32\n"
           "    %cr0 = scalar.addi %items_i, %seven_i : i32\n"
           "    %cr1 = scalar.subi %cr0, %col_i : i32\n"
           "    %cr2 = scalar.shrui %cr1, %three_i : i32\n"
           "    %item_mask = scalar.constant 16383 : i32\n"
           "    %cr3 = scalar.andi %cr2, %item_mask : i32\n"
           "    %my_rows_d = index.cast %cr3 : i32 to index\n";
  };
  auto count_end = [&](const std::string& tile) {
    if (!k.dynamic_rows) return std::string();
    return tile != "i" ? std::string("    channel.release %cnt_r : read<tile<16xi32>>\n")
                       : std::string("    channel.publish %cnt_w : write<tile<16xi32>>\n");
  };
  const std::string bound = k.dynamic_rows ? "%my_rows_d" : "%my_rows";
  // Work items (row, piece) spread over the columns, so a single row's
  // pieces still use the whole array.
  const std::string loops_open =
      "    scf.for %local = [%c0 to " + bound + " step %c1] {\n"
      "      %stride_items = index.mul %local, %c8 : index\n"
      "      %item = index.add %stride_items, %column : index\n"
      "      %row = index.div %item, %segments : index\n"
      "      %piece = index.rem %item, %segments : index\n";
  const std::string loops_close = "    }\n";
  std::string text;
  text += "aie2p.target<array> @npu\n";
  text += kMathLibrary;
  text += "\npipeline.def<kernel> @" + k.name + "_col(%column: index) run(" + args + ") {\n";
  text += "  " + assume_out + " = buffer.assume.alignment " + assume_in + " {minimum_alignment = 64} : " + types + "\n";
  text += "  %origin = index.constant 0 : offset\n  %c0 = index.constant 0 : index\n  %c1 = index.constant 1 : index\n"
          "  %c4 = index.constant 4 : index\n  %c8 = index.constant 8 : index\n  %c16 = index.constant 16 : index\n"
          "  %segments = index.constant " + S(k.segments) + " : index\n"
          "  %rows = index.constant " + S(k.rows) + " : index\n"
          // rows of this column: ceil((rows - column) / 8)
          "  %c7 = index.constant 7 : index\n"
          "  %items = index.mul %rows, %segments : index\n"
          "  %rest = index.sub %items, %column : index\n  %rest7 = index.add %rest, %c7 : index\n"
          "  %my_rows = index.div %rest7, %c8 : index\n";
  text += "  %memory = pipeline.memory<workgroup>[%column, 3] : pool\n";
  if (k.dynamic_rows) {
    text += "  %pool_in = pipeline.memory<workgroup>[%column, 2] : pool\n  %pool_out = pipeline.memory<workgroup>[%column, 4] : pool\n";
    for (auto [tile, pool] : std::vector<std::pair<std::string, std::string>>{{"i", "pool_in"}, {"c", "memory"}, {"o", "memory"}})
      text += "  %cnt_" + tile + "_size = index.constant 64 : offset\n"
              "  %cnt_" + tile + "_storage = buffer.alloca<workgroup> pool(%" + pool + ") align(64) %cnt_" + tile + "_size : buffer\n"
              "  %cnt_" + tile + "_slots = buffer.view %cnt_" + tile + "_storage[%origin] : buffer -> view<1x16xi32>\n"
              "  %cnt_" + tile + "_ch = channel.bind<fifo> %cnt_" + tile + "_slots, %c1 : view<1x16xi32>, index -> channel<tile<16xi32>>\n";
  }
  text += decl;
  text += "  pipeline.strand workers([%column, 2], [1, 1], [1, 1]) {\n" + count("i") + loops_open + ingress + loops_close + count_end("i") + "  }\n";
  text += "  pipeline.strand workers([%column, 4], [1, 1], [1, 1]) {\n" + count("o") + loops_open + egress + loops_close + count_end("o") + "  }\n";
  text += "  pipeline.strand workers([%column, 3], [1, 1], [1, 1]) {\n" + count("c") +
          loops_open + acquire + k.body + publish + release + loops_close + count_end("c") + "  }\n}\n";
  text += "\npipeline.def<kernel> public retain target(@npu) @" + k.name + "() run(" + args + ") {\n";
  for (std::uint32_t c = 0; c < 8; ++c) text += "  %k" + S(c) + " = index.constant " + S(c) + " : index\n";
  for (std::uint32_t c = 0; c < 8; ++c)
    text += "  pipeline.compose @" + k.name + "_col[%k" + S(c) + "](" + assume_in + ") : [index](" + types + ")\n";
  text += "}\n";
  return text;
}

}  // namespace gufo::models::lfm3::npu
