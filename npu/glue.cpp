#include "glue.hpp"

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
std::string F(double v) {
  char b[64];
  std::snprintf(b, sizeof b, "%.9e", v);
  return b;
}
using Layout = RowOperand::Layout;

// %scale_<tag>: rsqrt(mean(v^2) + eps) over view %<view> of n floats.
std::string Scale(const std::string& view, const std::string& tag) {
  return Fill(R"(      %sq_@T = scf.for %i_@T = [%c0 to %n step %c16](%acc_@T = %zero : vector<16xf32>) -> (vector<16xf32>) {
        %v_@T = vector.load %@V[%i_@T] : view<@Nxf32> -> vector<16xf32>
        %p_@T = func.call @fmul(%v_@T, %v_@T) : (vector<16xf32>, vector<16xf32>) -> (vector<16xf32>)
        %s_@T = vector.addf %acc_@T, %p_@T : vector<16xf32>
        scf.yield %s_@T : vector<16xf32>
      }
      %t_@T = vector.reduce<addf> %sq_@T, %fzero : vector<16xf32>, f32
      %tv_@T = vector.splat %t_@T : vector<16xf32>
      %mean_@T = func.call @fmul(%tv_@T, %inv_n) : (vector<16xf32>, vector<16xf32>) -> (vector<16xf32>)
      %me_@T = vector.addf %mean_@T, %eps : vector<16xf32>
      %scale_@T = func.call @rsqrt(%me_@T) : (vector<16xf32>) -> (vector<16xf32>)
)",
              {{"@T", tag}, {"@V", view}});
}

}  // namespace

RowKernel NormPackKernel(std::uint32_t H, std::uint32_t rows, Residual residual, bool post, bool rows_out,
                         float eps) {
  RowKernel k;
  k.name = std::string("normpack_h") + S(H) + "_r" + S(rows) + (residual == Residual::kBlocked ? "_yb" : residual == Residual::kRows ? "_yr" : "") +
           (post ? "_post" : "") + (rows_out ? "_z" : "");
  k.rows = rows;
  std::uint32_t b = 0;
  k.inputs.push_back({"x", b++, H, 0, H});
  if (residual == Residual::kBlocked) k.inputs.push_back({"y", b++, 0, 0, H, false, Layout::kBlocked, H / 8});
  if (residual == Residual::kRows) k.inputs.push_back({"y", b++, H, 0, H});
  const std::uint32_t wfloats = (residual != Residual::kNone && post) ? 2 * H : H;
  k.inputs.push_back({"w", b++, 0, 0, wfloats});
  // The last layer's variant (rows_out) skips the residual nobody reads.
  if (residual != Residual::kNone && !rows_out) k.outputs.push_back({"x2", b++, H, 0, H});
  k.outputs.push_back({"xb", b++, 0, 0, H, true, Layout::kBlocked, H / 8});
  if (rows_out) k.outputs.push_back({"z", b++, H, 0, H});
  k.bindings = b;
  std::string body = Fill(R"(      %n = index.constant @N : index
      %zero = vector.splatf 0.0 : vector<16xf32>
      %fzero = scalar.constant 0.0 : f32
      %inv_n = vector.splatf @INV : vector<16xf32>
      %eps = vector.splatf @EPS : vector<16xf32>
      %wn_off = index.constant @WN : index
)",
                          {{"@N", S(H)}, {"@INV", F(1.0 / H)}, {"@EPS", F(eps)},
                           {"@WN", S(wfloats == 2 * H ? H : 0)}});
  std::string src = "x";
  if (residual != Residual::kNone) {
    if (post) {
      body += Scale("y", "post");
      body += R"(      scf.for %i = [%c0 to %n step %c16] {
        %xv = vector.load %x[%i] : view<@Nxf32> -> vector<16xf32>
        %yv = vector.load %y[%i] : view<@Nxf32> -> vector<16xf32>
        %wv = vector.load %w[%i] : view<@WFxf32> -> vector<16xf32>
        %ys = func.call @fmul(%yv, %scale_post) : (vector<16xf32>, vector<16xf32>) -> (vector<16xf32>)
        %yw = func.call @fmul(%ys, %wv) : (vector<16xf32>, vector<16xf32>) -> (vector<16xf32>)
        %s = vector.addf %xv, %yw : vector<16xf32>
        vector.store %s, %@RES[%i] : vector<16xf32>, view<@Nxf32>
      }
)";
    } else {
      body += R"(      scf.for %i = [%c0 to %n step %c16] {
        %xv = vector.load %x[%i] : view<@Nxf32> -> vector<16xf32>
        %yv = vector.load %y[%i] : view<@Nxf32> -> vector<16xf32>
        %s = vector.addf %xv, %yv : vector<16xf32>
        vector.store %s, %@RES[%i] : vector<16xf32>, view<@Nxf32>
      }
)";
    }
    src = rows_out ? "x" : "x2";
  }
  body += Scale(src, "norm");
  body += Fill(R"(      scf.for %i = [%c0 to %n step %c16] {
        %v = vector.load %@SRC[%i] : view<@Nxf32> -> vector<16xf32>
        %wi = index.add %i, %wn_off : index
        %g = vector.load %w[%wi] : view<@WFxf32> -> vector<16xf32>
        %t = func.call @fmul(%v, %scale_norm) : (vector<16xf32>, vector<16xf32>) -> (vector<16xf32>)
        %u = func.call @fmul(%t, %g) : (vector<16xf32>, vector<16xf32>) -> (vector<16xf32>)
        %h = vector.fptrunc %u : vector<16xf32> to vector<16xbf16>
        vector.store %h, %xb[%i] : vector<16xbf16>, view<@Nxbf16>
@Z      }
)",
               {{"@SRC", src}, {"@Z", rows_out ? "        vector.store %u, %z[%i] : vector<16xf32>, view<@Nxf32>\n" : ""}});
  // Without x2 the sum overwrites x in the tile (x is not written back).
  k.body = Fill(body, {{"@N", S(H)}, {"@WF", S(wfloats)}, {"@RES", rows_out ? "x" : "x2"}});
  return k;
}

RowKernel ReducePartialKernel(std::uint32_t m, std::uint32_t rows) {
  RowKernel k;
  k.name = "reduce_m" + S(m) + "_r" + S(rows);
  k.rows = rows;
  const std::uint32_t piece = 256;
  k.segments = m / piece;
  k.bindings = 2;
  k.inputs.push_back({"yp", 0, 0, 0, piece, false, Layout::kPartial, m / 8});
  k.outputs.push_back({"o", 1, m, 0, piece});
  k.body = Fill(R"(      %n = index.constant @P : index
      %c8i = index.constant 8 : index
      scf.for %i = [%c0 to %n step %c16] {
        %s = scf.for %c = [%c0 to %c8i step %c1](%a = %zero_r : vector<16xf32>) -> (vector<16xf32>) {
          %v = vector.load %yp[%c, %i] : view<8x@Pxf32> -> vector<16xf32>
          %b = vector.addf %a, %v : vector<16xf32>
          scf.yield %b : vector<16xf32>
        }
        vector.store %s, %o[%i] : vector<16xf32>, view<@Pxf32>
      }
)",
                {{"@P", S(piece)}});
  k.body = "      %zero_r = vector.splatf 0.0 : vector<16xf32>\n" + k.body;
  return k;
}

RowKernel SwiGluPackKernel(std::uint32_t I, std::uint32_t rows) {
  RowKernel k;
  k.name = "swiglu_i" + S(I) + "_r" + S(rows);
  k.rows = rows;
  const std::uint32_t piece = 512;
  if (I % piece) throw std::invalid_argument("SwiGLU width must be a multiple of 512");
  k.segments = I / piece;
  k.bindings = 2;
  k.inputs.push_back({"g", 0, 0, 0, piece, false, Layout::kBlocked, 2 * I / 8});
  k.inputs.push_back({"u", 0, 0, std::uint64_t(I / 8) * 64, piece, false, Layout::kBlocked, 2 * I / 8});
  k.outputs.push_back({"xb", 1, 0, 0, piece, true, Layout::kBlocked, I / 8});
  k.body = Fill(R"(      %n = index.constant @P : index
      scf.for %i = [%c0 to %n step %c16] {
        %gv = vector.load %g[%i] : view<@Pxf32> -> vector<16xf32>
        %uv = vector.load %u[%i] : view<@Pxf32> -> vector<16xf32>
        %sg = func.call @sigmoid(%gv) : (vector<16xf32>) -> (vector<16xf32>)
        %silu = func.call @fmul2(%gv, %sg) : (vector<16xf32>, vector<16xf32>) -> (vector<16xf32>)
        %o = func.call @fmul2(%silu, %uv) : (vector<16xf32>, vector<16xf32>) -> (vector<16xf32>)
        %ob = vector.fptrunc %o : vector<16xf32> to vector<16xbf16>
        vector.store %ob, %xb[%i] : vector<16xbf16>, view<@Pxbf16>
      }
)",
                {{"@P", S(piece)}});
  return k;
}

std::string ConvProgram(std::uint32_t H, std::uint32_t L, std::uint32_t rows, std::uint32_t slots,
                        const std::string& name) {
  // Column c owns channels [c * H / 8, (c + 1) * H / 8) and walks the rows in
  // order, so prompt rows of one session see each other's history.
  const std::uint32_t slice = H / 8, G = 3 * H / 8, sg = slice / 8;
  if (H % 64 || (slots & (slots - 1)) || L < 2) throw std::invalid_argument("unsupported NPU convolution");
  std::string t = "aie2p.target<array> @npu\n";
  t += kMathLibrary;
  t += Fill(R"(
pipeline.def<kernel> @@NAME_col(%column: index) run(%proj_arg: buffer, %hist_arg: buffer, %w_arg: buffer, %meta_arg: buffer, %out_arg: buffer) {
  %proj, %hist, %wts, %meta, %out = buffer.assume.alignment %proj_arg, %hist_arg, %w_arg, %meta_arg, %out_arg {minimum_alignment = 64} : buffer, buffer, buffer, buffer, buffer
  %origin = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c2 = index.constant 2 : index
  %c4 = index.constant 4 : index
  %c8 = index.constant 8 : index
  %c16 = index.constant 16 : index
  %slice = index.constant @SLICE : index
  %hidden = index.constant @H : index
  %taps = index.constant @TAPS : index
  %last_tap = index.constant @LAST : index
  %rows = index.constant @ROWS : index
  %hist_stride = index.constant @HSTRIDE : index
  %pblock = index.constant @PBLOCK : index
  %oblock = index.constant @OBLOCK : index
  %cgroup = index.constant @CGROUP : index
  %first = index.mul %column, %slice : index
  %first_group = index.mul %column, %cgroup : index
  %memory = pipeline.memory<workgroup>[%column, 2] : pool
  %w_size = index.constant @WBYTES : offset
  %w_storage = buffer.alloca<workgroup> pool(%memory) align(64) %w_size : buffer
  %w_slots = buffer.view %w_storage[%origin] : buffer -> view<1x@Lx@SLICExf32>
  %w_ch = channel.bind<fifo> %w_slots, %c1 : view<1x@Lx@SLICExf32>, index -> channel<tile<@Lx@SLICExf32>>
  %h_size = index.constant @HBYTES : offset
  %h_storage = buffer.alloca<workgroup> pool(%memory) align(64) %h_size : buffer
  %h_slots = buffer.view %h_storage[%origin] : buffer -> view<1x@TAPSx@SLICExf32>
  %h_ch = channel.bind<fifo> %h_slots, %c1 : view<1x@TAPSx@SLICExf32>, index -> channel<tile<@TAPSx@SLICExf32>>
  %p_size = index.constant @PBYTES : offset
  %p_storage = buffer.alloca<workgroup> pool(%memory) align(64) %p_size : buffer
  %p_slots = buffer.view %p_storage[%origin] : buffer -> view<1x3x@SLICExf32>
  %p_ch = channel.bind<fifo> %p_slots, %c1 : view<1x3x@SLICExf32>, index -> channel<tile<3x@SLICExf32>>
  %o_size = index.constant @OBYTES : offset
  %o_storage = buffer.alloca<workgroup> pool(%memory) align(64) %o_size : buffer
  %o_slots = buffer.view %o_storage[%origin] : buffer -> view<1x@SLICExbf16>
  %o_ch = channel.bind<fifo> %o_slots, %c1 : view<1x@SLICExbf16>, index -> channel<tile<@SLICExbf16>>
  %m_size = index.constant 64 : offset
  %m_storage = buffer.alloca<workgroup> pool(%memory) align(64) %m_size : buffer
  %m_slots = buffer.view %m_storage[%origin] : buffer -> view<1x16xi32>
  %m_ch = channel.bind<fifo> %m_slots, %c1 : view<1x16xi32>, index -> channel<tile<16xi32>>
  pipeline.strand workers([%column, 2], [1, 1], [1, 1]) {
    %ww, %w = channel.reserve %w_ch : channel<tile<@Lx@SLICExf32>> -> (write<tile<@Lx@SLICExf32>>, view<@Lx@SLICExf32>)
    %hw, %h = channel.reserve %h_ch : channel<tile<@TAPSx@SLICExf32>> -> (write<tile<@TAPSx@SLICExf32>>, view<@TAPSx@SLICExf32>)
    %pw, %p = channel.reserve %p_ch : channel<tile<3x@SLICExf32>> -> (write<tile<3x@SLICExf32>>, view<3x@SLICExf32>)
    %ow, %o = channel.reserve %o_ch : channel<tile<@SLICExbf16>> -> (write<tile<@SLICExbf16>>, view<@SLICExbf16>)
    %mw, %m = channel.reserve %m_ch : channel<tile<16xi32>> -> (write<tile<16xi32>>, view<16xi32>)
    %w_at_i = index.mul %first, %c4 : index
    %w_at = index.cast %w_at_i : index to offset
    %w_dense = buffer.view %wts[%w_at] : buffer -> view<@Lx@Hxf32>
    %w_strided = view.refine %w_dense : view<@Lx@Hxf32> -> view<@Lx@Hxf32, #encoding.layout.strided<strides=[@H, 1]>>
    %w_src = view.subview %w_strided[0, 0] : view<@Lx@Hxf32, #encoding.layout.strided<strides=[@H, 1]>> -> view<@Lx@SLICExf32, #encoding.layout.strided<strides=[@H, 1]>>
    %wc = kernel.async.copy %w_src to %w {cache_scope = workgroup, cache_temporal = regular, direction = global_to_workgroup} : view<@Lx@SLICExf32, #encoding.layout.strided<strides=[@H, 1]>> to view<@Lx@SLICExf32> -> kernel.async.token
    %wg = kernel.async.group %wc : kernel.async.token -> kernel.async.group
    kernel.async.wait %wg {newer_groups = 0} : kernel.async.group
    scf.for %row = [%c0 to %rows step %c1] {
      %m_e = index.mul %row, %c16 : index
      %m_bytes = index.mul %m_e, %c4 : index
      %m_at = index.cast %m_bytes : index to offset
      %m_src = buffer.view %meta[%m_at] : buffer -> view<16xi32>
      %mc = kernel.async.copy %m_src to %m {cache_scope = workgroup, cache_temporal = regular, direction = global_to_workgroup} : view<16xi32> to view<16xi32> -> kernel.async.token
      %mg = kernel.async.group %mc : kernel.async.token -> kernel.async.group
      kernel.async.wait %mg {newer_groups = 0} : kernel.async.group
      %slot_raw = view.load %m[%c0] : view<16xi32> -> i32
      %slot_mask = scalar.constant @SLOTM : i32
      %slot_b = scalar.andi %slot_raw, %slot_mask : i32
      %slot = index.cast %slot_b : i32 to index
      %h_base0 = index.mul %slot, %hist_stride : index
      %h_base = index.add %h_base0, %first : index
      %h_at_i = index.mul %h_base, %c4 : index
      %h_at = index.cast %h_at_i : index to offset
      %h_dense = buffer.view %hist[%h_at] : buffer -> view<@TAPSx@Hxf32>
      %h_strided = view.refine %h_dense : view<@TAPSx@Hxf32> -> view<@TAPSx@Hxf32, #encoding.layout.strided<strides=[@H, 1]>>
      %h_global = view.subview %h_strided[0, 0] : view<@TAPSx@Hxf32, #encoding.layout.strided<strides=[@H, 1]>> -> view<@TAPSx@SLICExf32, #encoding.layout.strided<strides=[@H, 1]>>
      %hc = kernel.async.copy %h_global to %h {cache_scope = workgroup, cache_temporal = regular, direction = global_to_workgroup} : view<@TAPSx@SLICExf32, #encoding.layout.strided<strides=[@H, 1]>> to view<@TAPSx@SLICExf32> -> kernel.async.token
      %hg = kernel.async.group %hc : kernel.async.token -> kernel.async.group
      kernel.async.wait %hg {newer_groups = 0} : kernel.async.group
      // B, C and x of the slice from the blocked projection output.
      %blk = index.div %row, %c8 : index
      %rb = index.rem %row, %c8 : index
      %pb0 = index.mul %blk, %pblock : index
      %pg = index.mul %first_group, %c8 : index
      %pg8 = index.mul %pg, %c8 : index
      %pr = index.mul %rb, %c8 : index
      %pa0 = index.add %pb0, %pg8 : index
      %pa = index.add %pa0, %pr : index
      scf.for %part = [%c0 to %c3p step %c1] {
        %part_off = index.mul %part, %part_stride : index
        %pe = index.add %pa, %part_off : index
        %p_bytes = index.mul %pe, %c4 : index
        %p_at = index.cast %p_bytes : index to offset
        %p_dense = buffer.view %proj[%p_at] : buffer -> view<@SGx64xf32>
        %p_strided = view.refine %p_dense : view<@SGx64xf32> -> view<@SGx64xf32, #encoding.layout.strided<strides=[64, 1]>>
        %p_global = view.subview %p_strided[0, 0] : view<@SGx64xf32, #encoding.layout.strided<strides=[64, 1]>> -> view<@SGx8xf32, #encoding.layout.strided<strides=[64, 1]>>
        %p_row = view.subview %p[%part, 0] : view<3x@SLICExf32> -> view<1x@SLICExf32>
        %p_tile = view.bitcast %p_row : view<1x@SLICExf32> -> view<@SGx8xf32>
        %pc = kernel.async.copy %p_global to %p_tile {cache_scope = workgroup, cache_temporal = regular, direction = global_to_workgroup} : view<@SGx8xf32, #encoding.layout.strided<strides=[64, 1]>> to view<@SGx8xf32> -> kernel.async.token
        %pgw = kernel.async.group %pc : kernel.async.token -> kernel.async.group
        kernel.async.wait %pgw {newer_groups = 0} : kernel.async.group
      }
      scf.for %i = [%c0 to %slice step %c16] {
        %bv = vector.load %p[%c0, %i] : view<3x@SLICExf32> -> vector<16xf32>
        %cv = vector.load %p[%c1, %i] : view<3x@SLICExf32> -> vector<16xf32>
        %xv = vector.load %p[%c2, %i] : view<3x@SLICExf32> -> vector<16xf32>
        %bx = func.call @fmul(%bv, %xv) : (vector<16xf32>, vector<16xf32>) -> (vector<16xf32>)
        %wl = vector.load %w[%taps, %i] : view<@Lx@SLICExf32> -> vector<16xf32>
        %acc0 = func.call @fmul(%wl, %bx) : (vector<16xf32>, vector<16xf32>) -> (vector<16xf32>)
        %acc = scf.for %tp = [%c0 to %taps step %c1](%a = %acc0 : vector<16xf32>) -> (vector<16xf32>) {
          %hv = vector.load %h[%tp, %i] : view<@TAPSx@SLICExf32> -> vector<16xf32>
          %wv = vector.load %w[%tp, %i] : view<@Lx@SLICExf32> -> vector<16xf32>
          %term = func.call @fmul(%wv, %hv) : (vector<16xf32>, vector<16xf32>) -> (vector<16xf32>)
          %next = vector.addf %a, %term : vector<16xf32>
          scf.yield %next : vector<16xf32>
        }
        %y = func.call @fmul(%cv, %acc) : (vector<16xf32>, vector<16xf32>) -> (vector<16xf32>)
        %yb = vector.fptrunc %y : vector<16xf32> to vector<16xbf16>
        vector.store %yb, %o[%i] : vector<16xbf16>, view<@SLICExbf16>
        scf.for %tp = [%c0 to %last_tap step %c1] {
          %tp1 = index.add %tp, %c1 : index
          %hn = vector.load %h[%tp1, %i] : view<@TAPSx@SLICExf32> -> vector<16xf32>
          vector.store %hn, %h[%tp, %i] : vector<16xf32>, view<@TAPSx@SLICExf32>
        }
        vector.store %bx, %h[%last_tap, %i] : vector<16xf32>, view<@TAPSx@SLICExf32>
      }
      %hs = kernel.async.copy %h to %h_global {cache_scope = workgroup, cache_temporal = regular, direction = workgroup_to_global} : view<@TAPSx@SLICExf32> to view<@TAPSx@SLICExf32, #encoding.layout.strided<strides=[@H, 1]>> -> kernel.async.token
      %hsg = kernel.async.group %hs : kernel.async.token -> kernel.async.group
      kernel.async.wait %hsg {newer_groups = 0} : kernel.async.group
      %ob0 = index.mul %blk, %oblock : index
      %oa0 = index.add %ob0, %pg8 : index
      %oa = index.add %oa0, %pr : index
      %o_bytes = index.mul %oa, %c2 : index
      %o_at = index.cast %o_bytes : index to offset
      %o_dense = buffer.view %out[%o_at] : buffer -> view<@SGx64xbf16>
      %o_strided = view.refine %o_dense : view<@SGx64xbf16> -> view<@SGx64xbf16, #encoding.layout.strided<strides=[64, 1]>>
      %o_global = view.subview %o_strided[0, 0] : view<@SGx64xbf16, #encoding.layout.strided<strides=[64, 1]>> -> view<@SGx8xbf16, #encoding.layout.strided<strides=[64, 1]>>
      %o_tile = view.bitcast %o : view<@SLICExbf16> -> view<@SGx8xbf16>
      %os = kernel.async.copy %o_tile to %o_global {cache_scope = workgroup, cache_temporal = regular, direction = workgroup_to_global} : view<@SGx8xbf16> to view<@SGx8xbf16, #encoding.layout.strided<strides=[64, 1]>> -> kernel.async.token
      %osg = kernel.async.group %os : kernel.async.token -> kernel.async.group
      kernel.async.wait %osg {newer_groups = 0} : kernel.async.group
    }
    channel.publish %ww : write<tile<@Lx@SLICExf32>>
    channel.publish %hw : write<tile<@TAPSx@SLICExf32>>
    channel.publish %pw : write<tile<3x@SLICExf32>>
    channel.publish %ow : write<tile<@SLICExbf16>>
    channel.publish %mw : write<tile<16xi32>>
  }
}
)",
            {{"@NAME", name}, {"@SLICE", S(slice)}, {"@TAPS", S(L - 1)}, {"@LAST", S(L - 2)}, {"@ROWS", S(rows)},
             {"@HSTRIDE", S(std::uint64_t(L - 1) * H)}, {"@PBLOCK", S(std::uint64_t(G) * 64)},
             {"@OBLOCK", S(std::uint64_t(H / 8) * 64)}, {"@CGROUP", S(sg)}, {"@WBYTES", S(std::uint64_t(L) * slice * 4)},
             {"@HBYTES", S(std::uint64_t(L - 1) * slice * 4)}, {"@PBYTES", S(3ull * slice * 4)},
             {"@OBYTES", S(slice * 2)}, {"@SLOTM", S(slots - 1)}, {"@SG", S(sg)}, {"@L", S(L)}, {"@H", S(H)}});
  // Parts B, C, x sit H / 8 groups apart in the projection's blocked row.
  t = Fill(t, {{"  %first = index.mul %column, %slice : index",
                "  %c3p = index.constant 3 : index\n  %part_stride = index.constant " + S(std::uint64_t(H / 8) * 64) +
                    " : index\n  %first = index.mul %column, %slice : index"}});
  std::string compose = "\npipeline.def<kernel> public retain target(@npu) @" + name +
                        "() run(%proj: buffer, %hist: buffer, %wts: buffer, %meta: buffer, %out: buffer) {\n";
  for (int c = 0; c < 8; ++c) compose += "  %k" + S(c) + " = index.constant " + S(c) + " : index\n";
  for (int c = 0; c < 8; ++c)
    compose += "  pipeline.compose @" + name + "_col[%k" + S(c) +
               "](%proj, %hist, %wts, %meta, %out) : [index](buffer, buffer, buffer, buffer, buffer)\n";
  return t + compose + "}\n";
}

}  // namespace gufo::models::lfm3::npu
