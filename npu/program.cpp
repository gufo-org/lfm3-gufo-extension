#include "program.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

namespace gufo::models::lfm3::npu {
namespace {

constexpr std::uint32_t kPer = 8;  // tiles per aligned 576-byte load
constexpr const char* kSchema =
    "%schema = encoding.define #encoding.operand<element_format=bfp16ebs8, payload_elements=64, "
    "payload_packing=target_fragment, payload_registers=18, rounding=flush_subnormal> : encoding<schema>";

std::string Replace(std::string text, const std::vector<std::pair<std::string, std::string>>& values) {
  for (const auto& [key, value] : values) {
    for (std::size_t at = text.find(key); at != std::string::npos; at = text.find(key, at + value.size()))
      text.replace(at, key.size(), value);
  }
  return text;
}
std::string S(std::uint64_t v) { return std::to_string(v); }

// Accumulates one record of `record_tiles` tiles into acc, reading x tiles
// tile0 + j of a resident view of `x_tiles` tiles.
std::string TileLoop(const std::string& name, std::uint32_t record_tiles, std::uint32_t x_tiles) {
  std::string body, consts;
  for (std::uint32_t i = 0; i < kPer; ++i) {
    consts += "  %c" + S(i * 64) + "_x = index.constant " + S(i * 64) + " : index\n";
    body += Replace(R"(    %x_at@I = index.add %x_base, %c@X_x : index
    %x_values@I = vector.load %x[%x_at@I] : view<@XEbf16> -> vector<64xbf16>
    %x_bytes@I = vector.encode %x_values@I using %schema : vector<64xbf16>, encoding<schema> -> vector<72xi8>
    %lhs@I = vector.fragment<lhs> %x_bytes@I shape [%c8, %c8] using {schema = %schema : encoding<schema>} : vector<72xi8>
    %w_bytes@I = vector.slice %w_blocks[@W] : vector<@GBxi8> -> vector<72xi8>
    %rhs@I = vector.fragment<rhs> %w_bytes@I shape [%c8, %c8] using {schema = %schema : encoding<schema>} : vector<72xi8>
    %a@N = vector.mma %lhs@I, %rhs@I, %a@I : vector<72xi8>, vector<72xi8>, vector<64xf32>
)",
                    {{"@XE", S(x_tiles * 64) + "x"}, {"@I", S(i)}, {"@X", S(i * 64)},
                     {"@N", S(i + 1)}, {"@W", S(i * 72)}, {"@GB", S(kPer * 72)}});
  }
  return Replace(R"(
func.def inline @@NAME(%x: view<@XTxbf16>, %w: view<@RBxi8>, %tile0: index, %acc: vector<64xf32>) -> (vector<64xf32>) {
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c8 = index.constant 8 : index
  %c64 = index.constant 64 : index
  %cper = index.constant @PER : index
  %cbytes = index.constant @GB : index
  %chunks = index.constant @CHUNKS : index
@CONSTS  @SCHEMA
  %result = scf.for %chunk = [%c0 to %chunks step %c1](%a0 = %acc : vector<64xf32>) -> (vector<64xf32>) {
    %w_at = index.mul %chunk, %cbytes : index
    %w_group = vector.load %w[%w_at] : view<@RBxi8> -> vector<@GBxi8>
    %w_blocks = vector.fragment<rhs> %w_group shape [blocks(%cper), %c8, %c8] using {schema = %schema : encoding<schema>} : vector<@GBxi8>
    %tile_in = index.mul %chunk, %cper : index
    %tile = index.add %tile0, %tile_in : index
    %x_base = index.mul %tile, %c64 : index
@BODY    scf.yield %a@PER : vector<64xf32>
  }
  func.return %result : vector<64xf32>
}
)",
                 {{"@NAME", name}, {"@XT", S(x_tiles * 64)}, {"@RB", S(record_tiles * 72)},
                  {"@PER", S(kPer)}, {"@GB", S(kPer * 72)}, {"@CHUNKS", S(record_tiles / kPer)},
                  {"@CONSTS", consts}, {"@SCHEMA", kSchema}, {"@BODY", body}});
}

}  // namespace

std::string ProjectionDot(const std::string& name, std::uint32_t record_tiles, std::uint32_t x_tiles) {
  return TileLoop(name, record_tiles, x_tiles);
}

ProjectionShape PlanProjection(std::uint32_t k, std::uint32_t m, std::uint32_t blocks) {
  ProjectionShape s;
  s.blocks = blocks ? blocks : 1;
  const std::uint32_t k128 = (k + 127) / 128 * 128;
  if (k128 <= 2048) {
    s.split = Split::kRows;
    s.k = k128;
    s.m = (m + 8 * kColumns - 1) / (8 * kColumns) * (8 * kColumns);
  } else {
    s.split = Split::kColumns;
    s.k = (k + 511) / 512 * 512;
    s.m = (m + 7) / 8 * 8;
  }
  return s;
}

std::string ProjectionRoot(const ProjectionShape& s) {
  return std::string("@proj_") + (s.split == Split::kRows ? "m" : "k") + "_k" + S(s.k) + "_m" +
         S(s.m) + "_b" + S(s.blocks);
}

std::string ProjectionProgram(const ProjectionShape& s) {
  const std::uint32_t tiles = s.k / 8;
  const bool rows = s.split == Split::kRows;
  const std::uint32_t x_tiles = rows ? tiles : tiles / kColumns;
  const std::uint32_t record_tiles = rows ? tiles / 2 : tiles / kColumns;
  const std::uint32_t records_per_group = rows ? 2 : 1;
  const std::uint32_t groups = rows ? s.m / 8 / kColumns : s.m / 8;
  if (record_tiles % kPer || (rows && s.m % (8 * kColumns)) || s.m % 8)
    throw std::invalid_argument("unsupported NPU projection shape");
  const std::string rec = S(record_tiles * 72) + "xi8";
  const std::string rtile = "tile<" + rec + ">";
  const std::string xb = S(x_tiles * 64) + "xbf16";
  const std::string dot = "dot_x" + S(x_tiles) + "_r" + S(record_tiles);
  std::string acquire, acc = "%init";
  for (std::uint32_t r = 0; r < records_per_group; ++r) {
    acquire += Replace(R"(      %r@R, %w@R = channel.acquire %w_channel : channel<@RT> -> (read<@RT>, view<@REC>)
      %t@R = index.constant @T0 : index
      %acc@R = func.call @@DOT(%x_values, %w@R, %t@R, @ACC) : (view<@XB>, view<@REC>, index, vector<64xf32>) -> (vector<64xf32>)
      channel.release %r@R : read<@RT>
)",
                       {{"@RT", rtile}, {"@REC", rec}, {"@R", S(r)}, {"@T0", S(r * record_tiles)},
                        {"@DOT", dot}, {"@ACC", acc}, {"@XB", xb}});
    acc = "%acc" + S(r);
  }
  // Index expressions per split (all in index units).
  //   x tile of the column's first resident tile within one block
  //   weight record index of local record %record
  //   output tile index of local group %g in block %blk
  const std::string x_first = rows ? "%c0" : "%x_column";
  const std::string column = Replace(R"(
pipeline.def<kernel> @@COL(%column: index) run(%x_arg: buffer, %w_arg: buffer, %y_arg: buffer) {
  %x_unique, %w_unique, %y_unique = buffer.assume.noalias %x_arg, %w_arg, %y_arg : buffer, buffer, buffer
  %x_buf, %w_buf, %y_buf = buffer.assume.alignment %x_unique, %w_unique, %y_unique {minimum_alignment = 64} : buffer, buffer, buffer
  %origin = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c2 = index.constant 2 : index
  %blocks = index.constant @BLOCKS : index
  %groups = index.constant @GROUPS : index
  %records = index.constant @RECORDS : index
  %record_bytes = index.constant @RECBYTES : index
  %tile_bytes = index.constant 128 : index
  %out_bytes = index.constant 256 : index
  %block_tiles = index.constant @TILES : index
  %column_tiles = index.constant @XTILES : index
  %x_column = index.mul %column, %column_tiles : index
  %first_record = index.mul %column, %records : index
  %out_column = index.mul %column, %groups : index
  %block_outputs = index.constant @BLOCKOUT : index
  %memory = pipeline.memory<workgroup>[%column, 3] : pool
  %x_size = index.constant @XBYTES : offset
  %x_storage = buffer.alloca<workgroup> pool(%memory) align(64) %x_size : buffer
  %x_slots = buffer.view %x_storage[%origin] : buffer -> view<1x@XB>
  %x_channel = channel.bind<fifo> %x_slots, %c1 : view<1x@XB>, index -> channel<tile<@XB>>
  %ring_size = index.constant @RINGBYTES : offset
  %w_storage = buffer.alloca<workgroup> pool(%memory) align(64) %ring_size : buffer
  %w_slots = buffer.view %w_storage[%origin] : buffer -> view<2x@REC>
  %w_channel = channel.bind<fifo> %w_slots, %c2 : view<2x@REC>, index -> channel<@RT>
  %y_size = index.constant 512 : offset
  %y_storage = buffer.alloca<workgroup> pool(%memory) align(64) %y_size : buffer
  %y_slots = buffer.view %y_storage[%origin] : buffer -> view<2x8x8xf32>
  %y_channel = channel.bind<fifo> %y_slots, %c2 : view<2x8x8xf32>, index -> channel<tile<8x8xf32>>

  pipeline.strand workers([%column, 2], [1, 1], [1, 1]) {
    scf.for %blk = [%c0 to %blocks step %c1] {
      %x_block = index.mul %blk, %block_tiles : index
      %x_first = index.add %x_block, @XFIRST : index
      %x_byte = index.mul %x_first, %tile_bytes : index
      %x_offset = index.cast %x_byte : index to offset
      %x_external = buffer.view %x_buf[%x_offset] : buffer -> view<@XB>
      %x_write, %x_payload = channel.reserve %x_channel : channel<tile<@XB>> -> (write<tile<@XB>>, view<@XB>)
      %x_copy = kernel.async.copy %x_external to %x_payload {cache_scope = workgroup, cache_temporal = regular, direction = global_to_workgroup} : view<@XB> to view<@XB> -> kernel.async.token
      %x_group = kernel.async.group %x_copy : kernel.async.token -> kernel.async.group
      kernel.async.wait %x_group {newer_groups = 0} : kernel.async.group
      channel.publish %x_write : write<tile<@XB>>
      scf.for %record = [%c0 to %records step %c1] {
        %global = index.add %record, %first_record : index
        %byte_index = index.mul %global, %record_bytes : index
        %offset = index.cast %byte_index : index to offset
        %external = buffer.view %w_buf[%offset] : buffer -> view<@REC>
        %write, %payload = channel.reserve %w_channel : channel<@RT> -> (write<@RT>, view<@REC>)
        %copy = kernel.async.copy %external to %payload {cache_scope = workgroup, cache_temporal = regular, direction = global_to_workgroup} : view<@REC> to view<@REC> -> kernel.async.token
        %group = kernel.async.group %copy : kernel.async.token -> kernel.async.group
        kernel.async.wait %group {newer_groups = 0} : kernel.async.group
        channel.publish %write : write<@RT>
      }
    }
  }

  pipeline.strand workers([%column, 4], [1, 1], [1, 1]) {
    scf.for %blk = [%c0 to %blocks step %c1] {
      %block_out = index.mul %blk, %block_outputs : index
      scf.for %g = [%c0 to %groups step %c1] {
        %read, %payload = channel.acquire %y_channel : channel<tile<8x8xf32>> -> (read<tile<8x8xf32>>, view<8x8xf32>)
        %local = index.add %g, %out_column : index
        %out = index.add %local, %block_out : index
        %out_index = index.mul %out, %out_bytes : index
        %out_offset = index.cast %out_index : index to offset
        %y_external = buffer.view %y_buf[%out_offset] : buffer -> view<8x8xf32>
        %y_copy = kernel.async.copy %payload to %y_external {cache_scope = workgroup, cache_temporal = regular, direction = workgroup_to_global} : view<8x8xf32> to view<8x8xf32> -> kernel.async.token
        %y_group = kernel.async.group %y_copy : kernel.async.token -> kernel.async.group
        kernel.async.wait %y_group {newer_groups = 0} : kernel.async.group
        channel.release %read : read<tile<8x8xf32>>
      }
    }
  }

  pipeline.strand workers([%column, 3], [1, 1], [1, 1]) {
    %c8 = index.constant 8 : index
    scf.for %blk = [%c0 to %blocks step %c1] {
      %x_read, %x_values = channel.acquire %x_channel : channel<tile<@XB>> -> (read<tile<@XB>>, view<@XB>)
      scf.for %g = [%c0 to %groups step %c1] {
        %zero = vector.splatf 0.0 : vector<64xf32>
        %init = vector.fragment<init> %zero shape [%c8, %c8] : vector<64xf32>
@ACQUIRE        %y_write, %y_values = channel.reserve %y_channel : channel<tile<8x8xf32>> -> (write<tile<8x8xf32>>, view<8x8xf32>)
        vector.fragment.store<result> @FINAL, %y_values[0, 0] shape [%c8, %c8] : vector<64xf32>, view<8x8xf32>
        channel.publish %y_write : write<tile<8x8xf32>>
      }
      channel.release %x_read : read<tile<@XB>>
    }
  }
}
)",
                                     {{"@COL", ProjectionRoot(s).substr(1) + "_col"},
                                      {"@BLOCKS", S(s.blocks)},
                                      {"@GROUPS", S(groups)},
                                      {"@RECORDS", S(std::uint64_t(groups) * records_per_group)},
                                      {"@RECBYTES", S(record_tiles * 72)},
                                      {"@XTILES", S(x_tiles)},
                                      {"@TILES", S(tiles)},
                                      {"@BLOCKOUT", S(rows ? s.m / 8 : std::uint64_t(s.m / 8) * kColumns)},
                                      {"@XBYTES", S(x_tiles * 128)},
                                      {"@RINGBYTES", S(2 * record_tiles * 72)},
                                      {"@XFIRST", x_first},
                                      {"@ACQUIRE", Replace(acquire, {{"      ", "        "}})},
                                      {"@FINAL", acc},
                                      {"@XB", xb},
                                      {"@REC", rec},
                                      {"@RT", rtile}});
  std::string compose;
  for (std::uint32_t c = 0; c < kColumns; ++c) {
    compose += "  %c" + S(c) + " = index.constant " + S(c) + " : index\n";
  }
  for (std::uint32_t c = 0; c < kColumns; ++c)
    compose += "  pipeline.compose @" + ProjectionRoot(s).substr(1) + "_col[%c" + S(c) +
               "](%x, %w, %y) : [index](buffer, buffer, buffer)\n";
  return "aie2p.target<array> @npu\n" + TileLoop(dot, record_tiles, x_tiles) + column +
         "\npipeline.def<kernel> public retain target(@npu) " + ProjectionRoot(s) +
         "() run(%x: buffer, %w: buffer, %y: buffer) {\n" + compose + "}\n";
}


std::uint32_t KSplitBlocksPerPass(const ProjectionShape& s) {
  // Activations of this many blocks stay resident (about 32 KiB) while each
  // weight record streams once per pass; the largest divisor of `blocks`.
  const std::uint32_t x_bytes = s.k / 8 / kColumns * 128;
  std::uint32_t cap = std::max<std::uint32_t>(1, 32768 / x_bytes), bp = 1;
  for (std::uint32_t b = 1; b <= std::min(cap, s.blocks); ++b)
    if (s.blocks % b == 0) bp = b;
  return bp;
}

std::string ProjectionKMultiProgram(const ProjectionShape& s) {
  // K split for several row blocks: column c holds K / 8 of `bp` blocks'
  // activations and applies each weight record to all of them, so the
  // weights stream blocks / bp times instead of `blocks` times. Layouts as
  // ProjectionProgram (partial outputs Yp).
  if (s.split != Split::kColumns) throw std::invalid_argument("K-multi projection needs the K split");
  const std::uint32_t tiles = s.k / 8, x_tiles = tiles / kColumns, record_tiles = tiles / kColumns, groups = s.m / 8;
  if (record_tiles % kPer || s.m % 8) throw std::invalid_argument("unsupported NPU projection shape");
  const std::uint32_t bp = KSplitBlocksPerPass(s), passes = s.blocks / bp;
  const std::string rec = S(record_tiles * 72) + "xi8", rtile = "tile<" + rec + ">";
  const std::string xb = S(x_tiles * 64) + "xbf16", xr = S(bp) + "x" + xb, xt = "tile<" + xr + ">";
  const std::string dot = "dot_x" + S(x_tiles) + "_r" + S(record_tiles);
  const std::string name = ProjectionRoot(s).substr(1) + "_kmulti";
  const std::string column = Replace(R"(
pipeline.def<kernel> @@COL(%column: index) run(%x_arg: buffer, %w_arg: buffer, %y_arg: buffer) {
  %x_unique, %w_unique, %y_unique = buffer.assume.noalias %x_arg, %w_arg, %y_arg : buffer, buffer, buffer
  %x_buf, %w_buf, %y_buf = buffer.assume.alignment %x_unique, %w_unique, %y_unique {minimum_alignment = 64} : buffer, buffer, buffer
  %origin = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c2 = index.constant 2 : index
  %passes = index.constant @PASSES : index
  %bp = index.constant @BP : index
  %groups = index.constant @GROUPS : index
  %record_bytes = index.constant @RECBYTES : index
  %tile_bytes = index.constant 128 : index
  %out_bytes = index.constant 256 : index
  %block_tiles = index.constant @TILES : index
  %column_tiles = index.constant @XTILES : index
  %x_column = index.mul %column, %column_tiles : index
  %first_record = index.mul %column, %groups : index
  %out_column = index.mul %column, %groups : index
  %block_outputs = index.constant @BLOCKOUT : index
  %memory = pipeline.memory<workgroup>[%column, 3] : pool
  %x_size = index.constant @XBYTES : offset
  %x_storage = buffer.alloca<workgroup> pool(%memory) align(64) %x_size : buffer
  %x_slots = buffer.view %x_storage[%origin] : buffer -> view<1x@XR>
  %x_channel = channel.bind<fifo> %x_slots, %c1 : view<1x@XR>, index -> channel<@XT>
  %ring_size = index.constant @RINGBYTES : offset
  %w_storage = buffer.alloca<workgroup> pool(%memory) align(64) %ring_size : buffer
  %w_slots = buffer.view %w_storage[%origin] : buffer -> view<2x@REC>
  %w_channel = channel.bind<fifo> %w_slots, %c2 : view<2x@REC>, index -> channel<@RT>
  %y_size = index.constant 512 : offset
  %y_storage = buffer.alloca<workgroup> pool(%memory) align(64) %y_size : buffer
  %y_slots = buffer.view %y_storage[%origin] : buffer -> view<2x8x8xf32>
  %y_channel = channel.bind<fifo> %y_slots, %c2 : view<2x8x8xf32>, index -> channel<tile<8x8xf32>>

  pipeline.strand workers([%column, 2], [1, 1], [1, 1]) {
    scf.for %pass = [%c0 to %passes step %c1] {
      // bp blocks' slices, block stride TILES tiles.
      %blk0 = index.mul %pass, %bp : index
      %x_block = index.mul %blk0, %block_tiles : index
      %x_first = index.add %x_block, %x_column : index
      %x_byte = index.mul %x_first, %tile_bytes : index
      %x_offset = index.cast %x_byte : index to offset
      // Three DMA dimensions: block, tile, element (a block slice alone
      // exceeds one dimension's wrap).
      %x_dense = buffer.view %x_buf[%x_offset] : buffer -> view<@X3>
      %x_external = view.refine %x_dense : view<@X3> -> view<@X3, #encoding.layout.strided<strides=[@BSTRIDE, 64, 1]>>
      %x_write, %x_payload = channel.reserve %x_channel : channel<@XT> -> (write<@XT>, view<@XR>)
      %x_local = view.bitcast %x_payload : view<@XR> -> view<@X3>
      %x_copy = kernel.async.copy %x_external to %x_local {cache_scope = workgroup, cache_temporal = regular, direction = global_to_workgroup} : view<@X3, #encoding.layout.strided<strides=[@BSTRIDE, 64, 1]>> to view<@X3> -> kernel.async.token
      %x_group = kernel.async.group %x_copy : kernel.async.token -> kernel.async.group
      kernel.async.wait %x_group {newer_groups = 0} : kernel.async.group
      channel.publish %x_write : write<@XT>
      scf.for %record = [%c0 to %groups step %c1] {
        %global = index.add %record, %first_record : index
        %byte_index = index.mul %global, %record_bytes : index
        %offset = index.cast %byte_index : index to offset
        %external = buffer.view %w_buf[%offset] : buffer -> view<@REC>
        %write, %payload = channel.reserve %w_channel : channel<@RT> -> (write<@RT>, view<@REC>)
        %copy = kernel.async.copy %external to %payload {cache_scope = workgroup, cache_temporal = regular, direction = global_to_workgroup} : view<@REC> to view<@REC> -> kernel.async.token
        %group = kernel.async.group %copy : kernel.async.token -> kernel.async.group
        kernel.async.wait %group {newer_groups = 0} : kernel.async.group
        channel.publish %write : write<@RT>
      }
    }
  }

  pipeline.strand workers([%column, 4], [1, 1], [1, 1]) {
    scf.for %pass = [%c0 to %passes step %c1] {
      %blk0 = index.mul %pass, %bp : index
      scf.for %g = [%c0 to %groups step %c1] {
        scf.for %b = [%c0 to %bp step %c1] {
          %blk = index.add %blk0, %b : index
          %read, %payload = channel.acquire %y_channel : channel<tile<8x8xf32>> -> (read<tile<8x8xf32>>, view<8x8xf32>)
          %block_out = index.mul %blk, %block_outputs : index
          %local = index.add %g, %out_column : index
          %out = index.add %local, %block_out : index
          %out_index = index.mul %out, %out_bytes : index
          %out_offset = index.cast %out_index : index to offset
          %y_external = buffer.view %y_buf[%out_offset] : buffer -> view<8x8xf32>
          %y_copy = kernel.async.copy %payload to %y_external {cache_scope = workgroup, cache_temporal = regular, direction = workgroup_to_global} : view<8x8xf32> to view<8x8xf32> -> kernel.async.token
          %y_group = kernel.async.group %y_copy : kernel.async.token -> kernel.async.group
          kernel.async.wait %y_group {newer_groups = 0} : kernel.async.group
          channel.release %read : read<tile<8x8xf32>>
        }
      }
    }
  }

  pipeline.strand workers([%column, 3], [1, 1], [1, 1]) {
    %c8 = index.constant 8 : index
    %c0t = index.constant 0 : index
    scf.for %pass = [%c0 to %passes step %c1] {
      %x_read, %x_all = channel.acquire %x_channel : channel<@XT> -> (read<@XT>, view<@XR>)
      %x_flat = view.bitcast %x_all : view<@XR> -> view<@XBALL>
      scf.for %g = [%c0 to %groups step %c1] {
        %w_read, %w_values = channel.acquire %w_channel : channel<@RT> -> (read<@RT>, view<@REC>)
        scf.for %b = [%c0 to %bp step %c1] {
          %zero = vector.splatf 0.0 : vector<64xf32>
          %init = vector.fragment<init> %zero shape [%c8, %c8] : vector<64xf32>
          %xo = index.mul %b, %column_tiles : index
          %acc = func.call @@DOT(%x_flat, %w_values, %xo, %init) : (view<@XBALL>, view<@REC>, index, vector<64xf32>) -> (vector<64xf32>)
          %y_write, %y_values = channel.reserve %y_channel : channel<tile<8x8xf32>> -> (write<tile<8x8xf32>>, view<8x8xf32>)
          vector.fragment.store<result> %acc, %y_values[0, 0] shape [%c8, %c8] : vector<64xf32>, view<8x8xf32>
          channel.publish %y_write : write<tile<8x8xf32>>
        }
        channel.release %w_read : read<@RT>
      }
      channel.release %x_read : read<@XT>
    }
  }
}
)",
                                     {{"@COL", name + "_col"}, {"@PASSES", S(passes)}, {"@BP", S(bp)},
                                      {"@GROUPS", S(groups)}, {"@RECBYTES", S(record_tiles * 72)},
                                      {"@XTILES", S(x_tiles)}, {"@TILES", S(tiles)},
                                      {"@BLOCKOUT", S(std::uint64_t(groups) * kColumns)},
                                      {"@XBYTES", S(std::uint64_t(bp) * x_tiles * 128)},
                                      {"@RINGBYTES", S(2 * record_tiles * 72)},
                                      {"@BSTRIDE", S(std::uint64_t(tiles) * 64)},
                                      {"@XBALL", S(std::uint64_t(bp) * x_tiles * 64) + "xbf16"},
                                      {"@X3", S(bp) + "x" + S(x_tiles) + "x64xbf16"}, {"@XR", xr}, {"@XT", xt}, {"@XB", xb}, {"@REC", rec}, {"@RT", rtile},
                                      {"@DOT", dot}});
  std::string compose;
  for (std::uint32_t c = 0; c < kColumns; ++c) compose += "  %c" + S(c) + " = index.constant " + S(c) + " : index\n";
  for (std::uint32_t c = 0; c < kColumns; ++c)
    compose += "  pipeline.compose @" + name + "_col[%c" + S(c) + "](%x, %w, %y) : [index](buffer, buffer, buffer)\n";
  return "aie2p.target<array> @npu\n" + TileLoop(dot, record_tiles, bp * x_tiles) + column +
         "\npipeline.def<kernel> public retain target(@npu) @" + name + "() run(%x: buffer, %w: buffer, %y: buffer) {\n" +
         compose + "}\n";
}

}  // namespace gufo::models::lfm3::npu
