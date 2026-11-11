// Prototype: attention with BFP16 MMA on the XDNA NPU, against a CPU
// reference. Column c computes query blocks c, c + 8, ... (8 query vectors
// each, D = 64) over the same NK keys, with an online softmax over 32-key
// chunks. S = Q K^T and O += P V use 8x8x8 vector.mma on BFP16 fragments.
//   q   tiled BF16 Qt[block][j][r][d]: query 8 block + r, dims 8 j + d (pre-scaled)
//   k,v BF16 [NK][64]
//   out F32 [blocks * 8][64]
// Usage: npu_mma_attention_test [blocks keys]
#include <chrono>
#include <cmath>
#include <cstring>
#include <sstream>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "../rowops.hpp"
#include "bfp.hpp"
#include "npu_runtime.hpp"

namespace {
std::string Fill(std::string text, const std::vector<std::pair<std::string, std::string>>& values) {
  for (const auto& [key, value] : values)
    for (auto at = text.find(key); at != std::string::npos; at = text.find(key, at + value.size()))
      text.replace(at, key.size(), value);
  return text;
}

constexpr const char* kSchema =
    "%schema = encoding.define #encoding.operand<element_format=bfp16ebs8, payload_elements=64, "
    "payload_packing=target_fragment, payload_registers=18, rounding=flush_subnormal> : encoding<schema>";

std::string Program(std::uint32_t blocks, std::uint32_t keys, int mode) {
  std::string t = "aie2p.target<array> @npu\naie2p.target<core> @core\n"
      "low.func.def target<amd.xdna.aie2p.core>(@core) @cycles_low() -> (reg<aie2p.er>) asm {\n"
      "  %pair = mov.cycles\n  %cycles = slice %pair[0] : reg<aie2p.er x2> -> reg<aie2p.er>\n  return %cycles\n}\n"
      "func.def @cycles() -> (i32) {\n  %c = low.invoke @cycles_low() : () -> (i32)\n  func.return %c : i32\n}\n";
  t += gufo::models::lfm3::npu::kMathLibrary;
  t += R"LOOMH(
func.def @vmax(%a: vector<16xf32>, %b: vector<16xf32>) -> (vector<16xf32>) {
  %gt = vector.cmpf ogt, %a, %b : vector<16xf32> -> vector<16xi1>
  %r = vector.select %gt, %a, %b : vector<16xf32>
  func.return %r : vector<16xf32>
}
func.def @swap8(%x: vector<16xf32>) -> (vector<16xf32>) {
  %lo = vector.slice %x[0] : vector<16xf32> -> vector<8xf32>
  %hi = vector.slice %x[8] : vector<16xf32> -> vector<8xf32>
  %r = vector.concat<0> %hi, %lo : vector<8xf32>, vector<8xf32> -> vector<16xf32>
  func.return %r : vector<16xf32>
}
)LOOMH";
  t += Fill(R"(
pipeline.def<kernel> @attn_col(%column: index) run(%q_arg: buffer, %k_arg: buffer, %v_arg: buffer, %o_arg: buffer, %p_arg: buffer) {
  %qb, %kb, %vb, %ob, %pb = buffer.assume.alignment %q_arg, %k_arg, %v_arg, %o_arg, %p_arg {minimum_alignment = 64} : buffer, buffer, buffer, buffer, buffer
  %origin = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c2 = index.constant 2 : index
  %c3 = index.constant 3 : index
  %c4 = index.constant 4 : index
  %c8 = index.constant 8 : index
  %c16 = index.constant 16 : index
  %c32 = index.constant 32 : index
  %c64 = index.constant 64 : index
  %c512 = index.constant 512 : index
  %c256 = index.constant 256 : index
  %blocks = index.constant @BLOCKS : index
  %chunks = index.constant @CHUNKS : index
  %memory = pipeline.memory<workgroup>[%column, 3] : pool
  // Scratch, reserved once from one-slot channels.
  %qs_size = index.constant 1024 : offset
  %qs_storage = buffer.alloca<workgroup> pool(%memory) align(64) %qs_size : buffer
  %qs_slots = buffer.view %qs_storage[%origin] : buffer -> view<1x8x8x8xbf16>
  %qs_ch = channel.bind<fifo> %qs_slots, %c1 : view<1x8x8x8xbf16>, index -> channel<tile<8x8x8xbf16>>
  %ks_size = index.constant 4096 : offset
  %ks_storage = buffer.alloca<workgroup> pool(%memory) align(64) %ks_size : buffer
  %ks_slots = buffer.view %ks_storage[%origin] : buffer -> view<1x8x32x8xbf16>
  %ks_ch = channel.bind<fifo> %ks_slots, %c1 : view<1x8x32x8xbf16>, index -> channel<tile<8x32x8xbf16>>
  %vs_storage = buffer.alloca<workgroup> pool(%memory) align(64) %ks_size : buffer
  %vs_slots = buffer.view %vs_storage[%origin] : buffer -> view<1x8x32x8xbf16>
  %vs_ch = channel.bind<fifo> %vs_slots, %c1 : view<1x8x32x8xbf16>, index -> channel<tile<8x32x8xbf16>>
  %ss_size = index.constant 1280 : offset
  %ss_storage = buffer.alloca<workgroup> pool(%memory) align(64) %ss_size : buffer
  %ss_slots = buffer.view %ss_storage[%origin] : buffer -> view<1x320xf32>
  %ss_ch = channel.bind<fifo> %ss_slots, %c1 : view<1x320xf32>, index -> channel<tile<320xf32>>
  %ps_size = index.constant 512 : offset
  %ps_storage = buffer.alloca<workgroup> pool(%memory) align(64) %ps_size : buffer
  %ps_slots = buffer.view %ps_storage[%origin] : buffer -> view<1x256xbf16>
  %ps_ch = channel.bind<fifo> %ps_slots, %c1 : view<1x256xbf16>, index -> channel<tile<256xbf16>>
  %os_size = index.constant 2048 : offset
  %os_storage = buffer.alloca<workgroup> pool(%memory) align(64) %os_size : buffer
  %os_slots = buffer.view %os_storage[%origin] : buffer -> view<1x512xf32>
  %os_ch = channel.bind<fifo> %os_slots, %c1 : view<1x512xf32>, index -> channel<tile<512xf32>>

  pipeline.strand workers([%column, 3], [1, 1], [1, 1]) {

    %qs_w, %qs = channel.reserve %qs_ch : channel<tile<8x8x8xbf16>> -> (write<tile<8x8x8xbf16>>, view<8x8x8xbf16>)
    %ks_w, %ks = channel.reserve %ks_ch : channel<tile<8x32x8xbf16>> -> (write<tile<8x32x8xbf16>>, view<8x32x8xbf16>)
    %vs_w, %vs = channel.reserve %vs_ch : channel<tile<8x32x8xbf16>> -> (write<tile<8x32x8xbf16>>, view<8x32x8xbf16>)
    %ss_w, %ss = channel.reserve %ss_ch : channel<tile<320xf32>> -> (write<tile<320xf32>>, view<320xf32>)
    %ps_w, %ps = channel.reserve %ps_ch : channel<tile<256xbf16>> -> (write<tile<256xbf16>>, view<256xbf16>)
    %os_w, %os = channel.reserve %os_ch : channel<tile<512xf32>> -> (write<tile<512xf32>>, view<512xf32>)
    %ml = view.subview %ss[256] : view<320xf32> -> view<64xf32>
    %pz = vector.splatf 0.0 : vector<16xf32>
    %c48p = index.constant 48 : index
    vector.store %pz, %ml[%c48p] : vector<16xf32>, view<64xf32>
    %zero16 = vector.splatf 0.0 : vector<16xf32>
    %zero64 = vector.splatf 0.0 : vector<64xf32>
    %neg = vector.splatf -1.0e30 : vector<16xf32>
    %fzero = scalar.constant 0.0 : f32
    %izero_s = scalar.constant 0 : i32
    %fneg = scalar.constant -1.0e30 : f32
    @SCHEMA
    %nblk = index.sub %blocks, %column : index
    %nblk7 = index.add %nblk, %c8 : index
    %nblk8 = index.sub %nblk7, %c1 : index
    %my_blocks = index.div %nblk8, %c8 : index
    scf.for %bi = [%c0 to %my_blocks step %c1] {
      %bo = index.mul %bi, %c8 : index
      %blk = index.add %bo, %column : index
      // Q tiles of this block.
      %q_e = index.mul %blk, %c64 : index
      %q_e8 = index.mul %q_e, %c8 : index
      %q_bytes = index.mul %q_e8, %c2 : index
      %q_at = index.cast %q_bytes : index to offset
      %q_src = buffer.view %qb[%q_at] : buffer -> view<8x8x8xbf16>
      %q_cp = kernel.async.copy %q_src to %qs {cache_scope = workgroup, cache_temporal = regular, direction = global_to_workgroup} : view<8x8x8xbf16> to view<8x8x8xbf16> -> kernel.async.token
      %q_g = kernel.async.group %q_cp : kernel.async.token -> kernel.async.group
      kernel.async.wait %q_g {newer_groups = 0} : kernel.async.group
      vector.store %neg, %ml[%c0] : vector<16xf32>, view<64xf32>
      vector.store %zero16, %ml[%c16] : vector<16xf32>, view<64xf32>
      scf.for %i = [%c0 to %c512 step %c64] {
        vector.store %zero64, %os[%i] : vector<64xf32>, view<512xf32>
      }
      scf.for %ch = [%c0 to %chunks step %c1] {
        %tprev_dma = func.call @cycles() : () -> (i32)
        // K and V rows [32 x 64] reordered by DMA to [j][key][d].
        %k_row = index.mul %ch, %c32 : index
        %k_e = index.mul %k_row, %c64 : index
        %k_bytes = index.mul %k_e, %c2 : index
        %k_at = index.cast %k_bytes : index to offset
        %k_dense = buffer.view %kb[%k_at] : buffer -> view<8x32x8xbf16>
        %k_src = view.refine %k_dense : view<8x32x8xbf16> -> view<8x32x8xbf16, #encoding.layout.strided<strides=[8, 64, 1]>>
        %k_cp = kernel.async.copy %k_src to %ks {cache_scope = workgroup, cache_temporal = regular, direction = global_to_workgroup} : view<8x32x8xbf16, #encoding.layout.strided<strides=[8, 64, 1]>> to view<8x32x8xbf16> -> kernel.async.token
        %v_dense = buffer.view %vb[%k_at] : buffer -> view<8x32x8xbf16>
        %v_src = view.refine %v_dense : view<8x32x8xbf16> -> view<8x32x8xbf16, #encoding.layout.strided<strides=[8, 64, 1]>>
        %v_cp = kernel.async.copy %v_src to %vs {cache_scope = workgroup, cache_temporal = regular, direction = global_to_workgroup} : view<8x32x8xbf16, #encoding.layout.strided<strides=[8, 64, 1]>> to view<8x32x8xbf16> -> kernel.async.token
        %kv_g = kernel.async.group %k_cp, %v_cp : kernel.async.token, kernel.async.token -> kernel.async.group
        kernel.async.wait %kv_g {newer_groups = 0} : kernel.async.group
        %tm_dma = func.call @cycles() : () -> (i32)
        %tpslot_dma = index.constant 48 : index
        %tpf_dma = view.load %ml[%tpslot_dma] : view<64xf32> -> f32
        %tp_dma = scalar.bitcast %tpf_dma : f32 to i32
        %td_dma = scalar.subi %tm_dma, %tprev_dma : i32
        %ts_dma = scalar.addi %tp_dma, %td_dma : i32
        %tsf_dma = scalar.bitcast %ts_dma : i32 to f32
        view.store %tsf_dma, %ml[%tpslot_dma] : f32, view<64xf32>
        %tprev_s = func.call @cycles() : () -> (i32)
        // S^T tiles [kb][key][r]: sum over dim blocks j of K(j, kb) Q(j)^T.
        scf.for %kt = [%c0 to %c4i step %c1] {
          %s = scf.for %j = [%c0 to %c8 step %c1](%acc = %zero64 : vector<64xf32>) -> (vector<64xf32>) {
            %qm = vector.load %qs[%j, %c0, %c0] : view<8x8x8xbf16> -> vector<8x8xbf16>
            %qv = vector.bitcast %qm : vector<8x8xbf16> to vector<64xbf16>
            %qe = vector.encode %qv using %schema : vector<64xbf16>, encoding<schema> -> vector<72xi8>
            %rhs = vector.fragment<rhs> %qe shape [%c8, %c8] using {schema = %schema : encoding<schema>} : vector<72xi8>
            %kt8 = index.mul %kt, %c8 : index
            %km = vector.load %ks[%j, %kt8, %c0] : view<8x32x8xbf16> -> vector<8x8xbf16>
            %kv = vector.bitcast %km : vector<8x8xbf16> to vector<64xbf16>
            %ke = vector.encode %kv using %schema : vector<64xbf16>, encoding<schema> -> vector<72xi8>
            %lhs = vector.fragment<lhs> %ke shape [%c8, %c8] using {schema = %schema : encoding<schema>} : vector<72xi8>
            %init = vector.fragment<init> %acc shape [%c8, %c8] : vector<64xf32>
            %n = vector.mma %lhs, %rhs, %init : vector<72xi8>, vector<72xi8>, vector<64xf32>
            scf.yield %n : vector<64xf32>
          }
          %so = index.mul %kt, %c64 : index
          vector.store %s, %ss[%so] : vector<64xf32>, view<320xf32>
        }
        %tm_s = func.call @cycles() : () -> (i32)
        %tpslot_s = index.constant 49 : index
        %tpf_s = view.load %ml[%tpslot_s] : view<64xf32> -> f32
        %tp_s = scalar.bitcast %tpf_s : f32 to i32
        %td_s = scalar.subi %tm_s, %tprev_s : i32
        %ts_s = scalar.addi %tp_s, %td_s : i32
        %tsf_s = scalar.bitcast %ts_s : i32 to f32
        view.store %tsf_s, %ml[%tpslot_s] : f32, view<64xf32>
        // Online softmax for the 8 queries at once. S^T [kb][key][r] holds
        // query r in lanes r and r + 8 of every 16-lane chunk.
        %m_old = vector.load %ml[%c0] : view<64xf32> -> vector<16xf32>
        %l_old = vector.load %ml[%c16] : view<64xf32> -> vector<16xf32>
        %c256s = index.constant 256 : index
        %mx = scf.for %i = [%c0 to %c256s step %c16](%acc = %neg : vector<16xf32>) -> (vector<16xf32>) {
          %v = vector.load %ss[%i] : view<320xf32> -> vector<16xf32>
          %gt = vector.cmpf ogt, %v, %acc : vector<16xf32> -> vector<16xi1>
          %n = vector.select %gt, %v, %acc : vector<16xf32>
          scf.yield %n : vector<16xf32>
        }
        %mxs = func.call @swap8(%mx) : (vector<16xf32>) -> (vector<16xf32>)
        %gt2 = vector.cmpf ogt, %mxs, %mx : vector<16xf32> -> vector<16xi1>
        %rm = vector.select %gt2, %mxs, %mx : vector<16xf32>
        // Lazy rescale: move the running max (and rescale O, l) only when a
        // row's max grows by more than 8 (base 2); probabilities stay <= 2^8.
        %slack = vector.splatf 8.0 : vector<16xf32>
        %grow = vector.subf %rm, %m_old : vector<16xf32>
        %big = vector.cmpf ogt, %grow, %slack : vector<16xf32> -> vector<16xi1>
        %one_i = vector.splati 1 : vector<16xi32>
        %zero_i = vector.splati 0 : vector<16xi32>
        %bigi = vector.select %big, %one_i, %zero_i : vector<16xi32>
        %nbig = vector.reduce<addi> %bigi, %izero_s : vector<16xi32>, i32
        %need = scalar.cmpi ne, %nbig, %izero_s : i32
        %m_new, %alpha = scf.if %need -> (vector<16xf32>, vector<16xf32>) {
          %gt3 = vector.cmpf ogt, %rm, %m_old : vector<16xf32> -> vector<16xi1>
          %mn = vector.select %gt3, %rm, %m_old : vector<16xf32>
          %dm = vector.subf %m_old, %mn : vector<16xf32>
          %al = vector.exp2f<afn> %dm : vector<16xf32>
          scf.for %i = [%c0 to %c512 step %c16] {
            %v = vector.load %os[%i] : view<512xf32> -> vector<16xf32>
            %w = func.call @fmul2(%v, %al) : (vector<16xf32>, vector<16xf32>) -> (vector<16xf32>)
            vector.store %w, %os[%i] : vector<16xf32>, view<512xf32>
          }
          scf.yield %mn, %al : vector<16xf32>, vector<16xf32>
        } else {
          %onef = vector.splatf 1.0 : vector<16xf32>
          scf.yield %m_old, %onef : vector<16xf32>, vector<16xf32>
        }
        %sum = scf.for %i = [%c0 to %c256s step %c16](%acc = %zero16 : vector<16xf32>) -> (vector<16xf32>) {
          %v = vector.load %ss[%i] : view<320xf32> -> vector<16xf32>
          %d = vector.subf %v, %m_new : vector<16xf32>
          %e = vector.exp2f<afn> %d : vector<16xf32>
          %eb = vector.fptrunc %e : vector<16xf32> to vector<16xbf16>
          vector.store %eb, %ps[%i] : vector<16xbf16>, view<256xbf16>
          %n = vector.addf %acc, %e : vector<16xf32>
          scf.yield %n : vector<16xf32>
        }
        %sums = func.call @swap8(%sum) : (vector<16xf32>) -> (vector<16xf32>)
        %rs = vector.addf %sum, %sums : vector<16xf32>
        %la = func.call @fmul2(%l_old, %alpha) : (vector<16xf32>, vector<16xf32>) -> (vector<16xf32>)
        %l_new = vector.addf %la, %rs : vector<16xf32>
        vector.store %m_new, %ml[%c0] : vector<16xf32>, view<64xf32>
        vector.store %l_new, %ml[%c16] : vector<16xf32>, view<64xf32>
        %tm_x = func.call @cycles() : () -> (i32)
        %tpslot_x = index.constant 50 : index
        %tpf_x = view.load %ml[%tpslot_x] : view<64xf32> -> f32
        %tp_x = scalar.bitcast %tpf_x : f32 to i32
        %td_x = scalar.subi %tm_x, %tm_s : i32
        %ts_x = scalar.addi %tp_x, %td_x : i32
        %tsf_x = scalar.bitcast %ts_x : i32 to f32
        view.store %tsf_x, %ml[%tpslot_x] : f32, view<64xf32>
        // O^T tiles [j][d][r] += sum over kb of V(j, kb)^T P(kb).
        scf.for %j = [%c0 to %c8 step %c1] {
          %jo = index.mul %j, %c64 : index
          %o_in = vector.load %os[%jo] : view<512xf32> -> vector<64xf32>
          %o_new = scf.for %kt = [%c0 to %c4i step %c1](%acc = %o_in : vector<64xf32>) -> (vector<64xf32>) {
            %po = index.mul %kt, %c64 : index
            %pv = vector.load %ps[%po] : view<256xbf16> -> vector<64xbf16>
            %pm = vector.bitcast %pv : vector<64xbf16> to vector<8x8xbf16>
            %pt = vector.transpose<[1, 0]> %pm : vector<8x8xbf16> -> vector<8x8xbf16>
            %ptf = vector.bitcast %pt : vector<8x8xbf16> to vector<64xbf16>
            %pe = vector.encode %ptf using %schema : vector<64xbf16>, encoding<schema> -> vector<72xi8>
            %rhs = vector.fragment<rhs> %pe shape [%c8, %c8] using {schema = %schema : encoding<schema>} : vector<72xi8>
            %vt8 = index.mul %kt, %c8 : index
            %vm = vector.load %vs[%j, %vt8, %c0] : view<8x32x8xbf16> -> vector<8x8xbf16>
            %vt = vector.transpose<[1, 0]> %vm : vector<8x8xbf16> -> vector<8x8xbf16>
            %vtf = vector.bitcast %vt : vector<8x8xbf16> to vector<64xbf16>
            %ve = vector.encode %vtf using %schema : vector<64xbf16>, encoding<schema> -> vector<72xi8>
            %lhs = vector.fragment<lhs> %ve shape [%c8, %c8] using {schema = %schema : encoding<schema>} : vector<72xi8>
            %init = vector.fragment<init> %acc shape [%c8, %c8] : vector<64xf32>
            %n = vector.mma %lhs, %rhs, %init : vector<72xi8>, vector<72xi8>, vector<64xf32>
            scf.yield %n : vector<64xf32>
          }
          vector.store %o_new, %os[%jo] : vector<64xf32>, view<512xf32>
        }
        %tm_v = func.call @cycles() : () -> (i32)
        %tpslot_v = index.constant 51 : index
        %tpf_v = view.load %ml[%tpslot_v] : view<64xf32> -> f32
        %tp_v = scalar.bitcast %tpf_v : f32 to i32
        %td_v = scalar.subi %tm_v, %tm_x : i32
        %ts_v = scalar.addi %tp_v, %td_v : i32
        %tsf_v = scalar.bitcast %ts_v : i32 to f32
        view.store %tsf_v, %ml[%tpslot_v] : f32, view<64xf32>
      }
      // out[8 blk + r][8 j + d] = O^T[j][d][r] / l[r], BF16.
      %l_fin = vector.load %ml[%c16] : view<64xf32> -> vector<16xf32>
      %inv = func.call @recip(%l_fin) : (vector<16xf32>) -> (vector<16xf32>)
      scf.for %i = [%c0 to %c512 step %c16] {
        %v = vector.load %os[%i] : view<512xf32> -> vector<16xf32>
        %w = func.call @fmul2(%v, %inv) : (vector<16xf32>, vector<16xf32>) -> (vector<16xf32>)
        vector.store %w, %os[%i] : vector<16xf32>, view<512xf32>
      }
      scf.for %j = [%c0 to %c8 step %c1] {
        %jo = index.mul %j, %c64 : index
        %ofs = vector.load %os[%jo] : view<512xf32> -> vector<64xf32>
        %ofb = vector.fptrunc %ofs : vector<64xf32> to vector<64xbf16>
        %ofbm = vector.bitcast %ofb : vector<64xbf16> to vector<8x8xbf16>
        %oft = vector.transpose<[1, 0]> %ofbm : vector<8x8xbf16> -> vector<8x8xbf16>
        vector.store %oft, %qs[%j, %c0, %c0] : vector<8x8xbf16>, view<8x8x8xbf16>
      }
      %o_e = index.mul %blk, %c512 : index
      %o_bytes = index.mul %o_e, %c2 : index
      %o_at = index.cast %o_bytes : index to offset
      %o_dense = buffer.view %ob[%o_at] : buffer -> view<8x8x8xbf16>
      %o_dst = view.refine %o_dense : view<8x8x8xbf16> -> view<8x8x8xbf16, #encoding.layout.strided<strides=[8, 64, 1]>>
      %o_cp = kernel.async.copy %qs to %o_dst {cache_scope = workgroup, cache_temporal = regular, direction = workgroup_to_global} : view<8x8x8xbf16> to view<8x8x8xbf16, #encoding.layout.strided<strides=[8, 64, 1]>> -> kernel.async.token
      %o_g = kernel.async.group %o_cp : kernel.async.token -> kernel.async.group
      kernel.async.wait %o_g {newer_groups = 0} : kernel.async.group
    }
    %pf_e = index.mul %column, %c256 : index
    %pf_at = index.cast %pf_e : index to offset
    %pf_dst = buffer.view %pb[%pf_at] : buffer -> view<64xf32>
    %pf_cp = kernel.async.copy %ml to %pf_dst {cache_scope = workgroup, cache_temporal = regular, direction = workgroup_to_global} : view<64xf32> to view<64xf32> -> kernel.async.token
    %pf_g = kernel.async.group %pf_cp : kernel.async.token -> kernel.async.group
    kernel.async.wait %pf_g {newer_groups = 0} : kernel.async.group
    channel.publish %qs_w : write<tile<8x8x8xbf16>>
    channel.publish %ks_w : write<tile<8x32x8xbf16>>
    channel.publish %vs_w : write<tile<8x32x8xbf16>>
    channel.publish %ss_w : write<tile<320xf32>>
    channel.publish %ps_w : write<tile<256xbf16>>
    channel.publish %os_w : write<tile<512xf32>>
  }
}

pipeline.def<kernel> public retain target(@npu) @attn() run(%q: buffer, %k: buffer, %v: buffer, %o: buffer, %p: buffer) {
@COMPOSE}
)",
            {{"@SCHEMA", kSchema}, {"@BLOCKS", std::to_string(blocks)}, {"@CHUNKS", std::to_string(keys / 32)}});
  std::string compose;
  for (int c = 0; c < 8; ++c) compose += "  %k" + std::to_string(c) + " = index.constant " + std::to_string(c) + " : index\n";
  for (int c = 0; c < 8; ++c)
    compose += "  pipeline.compose @attn_col[%k" + std::to_string(c) + "](%q, %k, %v, %o, %p) : [index](buffer, buffer, buffer, buffer, buffer)\n";
  // mode 1: K/V DMA of chunk 0 only (reused), 2: no softmax/PV, 4: no S
  if (mode & 1) {
    t = Fill(t, {{"        %k_row = index.mul %ch, %c32 : index", "        %k_row = index.mul %c0, %c32 : index"}});
  }
  if (mode & 2) {
    auto a = t.find("        // Online softmax"), b = t.find("      // out[8 blk");
    t = t.substr(0, a) + "      }\n" + t.substr(b);
  }
  if (mode & 4) {
    auto a = t.find("        // S tiles"), b = t.find("        // Online softmax");
    t = t.substr(0, a) + t.substr(b);
  }
  if (mode & 8) {  // no P V
    auto a = t.find("        // O tiles"), b = t.find("      // out[8 blk");
    t = t.substr(0, a) + "      }\n" + t.substr(b);
  }
  if (mode & 16) {  // no O rescale
    auto a = t.find("          // Rescale O row r"), b = t.find("        // O tiles");
    t = t.substr(0, a) + "        }\n" + t.substr(b);
  }
  if (mode & 32) {  // timing only: exp replaced by identity
    t += "\nfunc.def @ident(%x: vector<16xf32>) -> (vector<16xf32>) {\n  func.return %x : vector<16xf32>\n}\n";
    auto a = t.find("pipeline.def<kernel> @attn_col");
    t = t.substr(0, a) + Fill(t.substr(a), {{"@exp_fast(", "@ident("}});
  }
  if (mode & 64) {  // timing only: fmul2 replaced by mulf
    t += "\nfunc.def @mulf(%a: vector<16xf32>, %b: vector<16xf32>) -> (vector<16xf32>) {\n  %r = vector.mulf %a, %b : vector<16xf32>\n  func.return %r : vector<16xf32>\n}\n";
    auto a = t.find("pipeline.def<kernel> @attn_col");
    t = t.substr(0, a) + Fill(t.substr(a), {{"@fmul2(", "@mulf("}});
  }
  if (mode & 128) {  // max via compare and select
    std::string out;
    std::istringstream in(t);
    for (std::string line; std::getline(in, line);) {
      const auto at = line.find("= vector.maxnumf ");
      if (at != std::string::npos) {
        const auto args = line.substr(at + 17, line.find(" :", at) - at - 17);
        line = line.substr(0, at) + "= func.call @vmax(" + args + ") : (vector<16xf32>, vector<16xf32>) -> (vector<16xf32>)";
      }
      out += line + "\n";
    }
    t = out;
  }
  return Fill(t, {{"@COMPOSE", compose}, {"%c4i", "%c4"}});
}
}  // namespace

int main(int argc, char** argv) {
  const std::uint32_t blocks = argc > 2 ? std::atoi(argv[1]) : 8, keys = argc > 2 ? std::atoi(argv[2]) : 256;
  const int mode = argc > 3 ? std::atoi(argv[3]) : 0;
  lfm3::npu::Runtime npu;
  std::string text = Program(blocks, keys, mode);
  if (std::getenv("DUMP")) {
    FILE* f = std::fopen("attn.loom", "w");
    std::fputs(text.c_str(), f);
    std::fclose(f);
  }
  int id;
  try {
    id = npu.Compile(text, "attn.loom", "@attn");
  } catch (const std::exception& e) {
    std::printf("%s\n", e.what());
    FILE* f = std::fopen("attn.loom", "w");
    std::fputs(text.c_str(), f);
    std::fclose(f);
    return 1;
  }
  const std::uint32_t NQ = blocks * 8, D = 64;
  std::mt19937 rng(3);
  std::normal_distribution<float> n(0, 1);
  std::vector<float> q(NQ * D), k(keys * D), v(keys * D);
  for (auto& x : q) x = n(rng) * 0.6f / 8;  // pre-scaled by 1/sqrt(D)
  // The kernel works in base 2: its Q carries an extra log2(e).
  const float kLog2e = 1.4426950408889634f;
  for (auto& x : k) x = n(rng) * 0.6f;
  for (auto& x : v) x = n(rng);
  auto bf = [](float x) { return lfm3::npu::ToBF16(x); };
  auto unbf = [](std::uint16_t h) { std::uint32_t u = std::uint32_t(h) << 16; float f; std::memcpy(&f, &u, 4); return f; };
  std::vector<std::uint16_t> qt(NQ * D), kb(keys * D), vb(keys * D);
  for (std::uint32_t i = 0; i < NQ; ++i)
    for (std::uint32_t d = 0; d < D; ++d) qt[((i / 8) * 8 + d / 8) * 64 + (i % 8) * 8 + d % 8] = bf(q[i * D + d] * kLog2e);
  for (std::size_t i = 0; i < k.size(); ++i) kb[i] = bf(k[i]), vb[i] = bf(v[i]);
  const int bq = npu.Upload(std::span<const std::uint16_t>(qt)), bk = npu.Upload(std::span<const std::uint16_t>(kb)),
            bv = npu.Upload(std::span<const std::uint16_t>(vb)), bo = npu.Alloc(NQ * D * 2), bp = npu.Alloc(8 * 256);
  const int bind[] = {bq, bk, bv, bo, bp};
  npu.Dispatch(id, bind);
  std::vector<std::uint16_t> ob(NQ * D);
  npu.Read(bo, 0, ob.data(), ob.size() * 2);
  std::vector<float> o(NQ * D);
  for (std::size_t i = 0; i < o.size(); ++i) o[i] = unbf(ob[i]);
  double max_err = 0, max_ref = 0;
  for (std::uint32_t i = 0; i < NQ; ++i) {
    std::vector<double> sc(keys);
    double mx = -1e30, den = 0;
    for (std::uint32_t j = 0; j < keys; ++j) {
      double t = 0;
      for (std::uint32_t d = 0; d < D; ++d) t += double(unbf(qt[((i / 8) * 8 + d / 8) * 64 + (i % 8) * 8 + d % 8])) / kLog2e * unbf(kb[j * D + d]);
      sc[j] = t;
      mx = std::max(mx, t);
    }
    for (auto& t : sc) den += (t = std::exp(t - mx));
    for (std::uint32_t d = 0; d < D; ++d) {
      double r = 0;
      for (std::uint32_t j = 0; j < keys; ++j) r += sc[j] * unbf(vb[j * D + d]);
      r /= den;
      max_err = std::max(max_err, std::abs(r - o[i * D + d]));
      max_ref = std::max(max_ref, std::abs(r));
    }
  }
  const int reps = 10;
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < reps; ++i) npu.Dispatch(id, bind);
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
  const double flop = 4.0 * NQ * keys * D;
  std::printf("queries %u keys %u: max err %.4f of %.3f, %.3f ms (%.1f GFLOP/s)\n", NQ, keys, max_err, max_ref, ms,
              flop / ms / 1e6);
  std::uint32_t prof[8 * 64];
  npu.Read(bp, 0, prof, sizeof prof);
  const double steps = double((blocks + 7) / 8) * (keys / 32);
  std::printf("column 0 cycles per 8q x 32k step (last run): dma %.0f, S %.0f, softmax %.0f, PV %.0f\n", prof[48] / steps,
              prof[49] / steps, prof[50] / steps, prof[51] / steps);
  return mode ? 0 : max_err > 0.05;
}

