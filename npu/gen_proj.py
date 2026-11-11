"""Generates npu/lfm3_proj.loom: LFM3 projections on the XDNA NPU.

y_b = W x_b for up to eight sequences b through AIE2P bfp16ebs8 matrix
multiplies (see docs/EXPERIMENTS.md, "NPU GEMV with block-floating weights").

Layouts (host side packs them, see npu/host/bfp.hpp):
  x  blocked BF16, Xb[j][b][c] = x_b[8j + c]          (K / 8 tiles of 64)
  W  pre-encoded 8 x 8 tiles, 72 bytes each: 8 rows of [exponent][8 signed
     mantissas], row n = output row 8g + n, column = k within the tile.
  Row split ("m" kernels, K = 2048): column c owns groups [c G, (c + 1) G);
     record R = 2g + h holds half h of K for group g: tile (R, jl) at byte
     (R * HALF + jl) * 72.
     y blocked F32, Yb[g][b][n] = y_b[8g + n].
  K split ("k" kernels): column c owns tiles [c T, (c + 1) T) of every group;
     record (c, g) at byte ((c * groups + g) * T) * 72.
     y partial F32 per column, Yp[c][g][b][n]; the host sums the eight.
"""

import sys

SCHEMA = ('%schema = encoding.define #encoding.operand<element_format=bfp16ebs8, payload_elements=64, '
          'payload_packing=target_fragment, payload_registers=18, rounding=flush_subnormal> : encoding<schema>')
COLUMNS = 8
PER = 8  # tiles per aligned 576-byte load


def tile_loop(name, tiles_per_record, x_tiles):
    """Accumulates one record of `tiles_per_record` tiles into acc. x tile
    index = tile0 + j, in a resident x view of `x_tiles` tiles."""
    body = []
    for i in range(PER):
        body.append(f"""    %x_at{i} = index.add %x_base, %c{i * 64}_x : index
    %x_values{i} = vector.load %x[%x_at{i}] : view<{x_tiles * 64}xbf16> -> vector<64xbf16>
    %x_bytes{i} = vector.encode %x_values{i} using %schema : vector<64xbf16>, encoding<schema> -> vector<72xi8>
    %lhs{i} = vector.fragment<lhs> %x_bytes{i} shape [%c8, %c8] using {{schema = %schema : encoding<schema>}} : vector<72xi8>
    %w_bytes{i} = vector.slice %w_blocks[{i * 72}] : vector<{PER * 72}xi8> -> vector<72xi8>
    %rhs{i} = vector.fragment<rhs> %w_bytes{i} shape [%c8, %c8] using {{schema = %schema : encoding<schema>}} : vector<72xi8>
    %a{i + 1} = vector.mma %lhs{i}, %rhs{i}, %a{i} : vector<72xi8>, vector<72xi8>, vector<64xf32>""")
    consts = '\n'.join(f'  %c{i * 64}_x = index.constant {i * 64} : index' for i in range(PER))
    nl = '\n'
    rec = f'{tiles_per_record * 72}xi8'
    return f"""
func.def inline @{name}(%x: view<{x_tiles * 64}xbf16>, %w: view<{rec}>, %tile0: index, %acc: vector<64xf32>) -> (vector<64xf32>) {{
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c8 = index.constant 8 : index
  %c64 = index.constant 64 : index
  %cper = index.constant {PER} : index
  %cbytes = index.constant {PER * 72} : index
  %chunks = index.constant {tiles_per_record // PER} : index
{consts}
  {SCHEMA}
  %result = scf.for %chunk = [%c0 to %chunks step %c1](%a0 = %acc : vector<64xf32>) -> (vector<64xf32>) {{
    %w_at = index.mul %chunk, %cbytes : index
    %w_group = vector.load %w[%w_at] : view<{rec}> -> vector<{PER * 72}xi8>
    %w_blocks = vector.fragment<rhs> %w_group shape [blocks(%cper), %c8, %c8] using {{schema = %schema : encoding<schema>}} : vector<{PER * 72}xi8>
    %tile_in = index.mul %chunk, %cper : index
    %tile = index.add %tile0, %tile_in : index
    %x_base = index.mul %tile, %c64 : index
{nl.join(body)}
    scf.yield %a{PER} : vector<64xf32>
  }}
  func.return %result : vector<64xf32>
}}
"""


def column(name, dot, x_tiles, x_first_tile_expr, tiles_per_record, records_per_group, groups,
           record_index_expr, out_index_expr):
    """One column's ingress, compute and egress strands."""
    rec = f'{tiles_per_record * 72}xi8'
    rtile = f'tile<{rec}>'
    xb = f'{x_tiles * 64}xbf16'
    acquire = []
    acc = '%init'
    for r in range(records_per_group):
        acquire.append(f"""      %r{r}, %w{r} = channel.acquire %w_channel : channel<{rtile}> -> (read<{rtile}>, view<{rec}>)
      %t{r} = index.constant {r * tiles_per_record} : index
      %acc{r} = func.call @{dot}(%x_values, %w{r}, %t{r}, {acc}) : (view<{xb}>, view<{rec}>, index, vector<64xf32>) -> (vector<64xf32>)
      channel.release %r{r} : read<{rtile}>""")
        acc = f'%acc{r}'
    nl = '\n'
    return f"""
pipeline.def<kernel> @{name}(%column: index) run(%x_arg: buffer, %w_arg: buffer, %y_arg: buffer) {{
  %x_unique, %w_unique, %y_unique = buffer.assume.noalias %x_arg, %w_arg, %y_arg : buffer, buffer, buffer
  %x_buf, %w_buf, %y_buf = buffer.assume.alignment %x_unique, %w_unique, %y_unique {{minimum_alignment = 64}} : buffer, buffer, buffer
  %origin = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c2 = index.constant 2 : index
  %groups = index.constant {groups} : index
  %records_per_group = index.constant {records_per_group} : index
  %records = index.mul %groups, %records_per_group : index
  %record_bytes = index.constant {tiles_per_record * 72} : index
  %tile_bytes = index.constant 128 : index
  %out_bytes = index.constant 256 : index
  %memory = pipeline.memory<workgroup>[%column, 3] : pool
  %x_size = index.constant {x_tiles * 128} : offset
  %x_storage = buffer.alloca<workgroup> pool(%memory) align(64) %x_size : buffer
  %x_slots = buffer.view %x_storage[%origin] : buffer -> view<1x{xb}>
  %x_channel = channel.bind<fifo> %x_slots, %c1 : view<1x{xb}>, index -> channel<tile<{xb}>>
  %ring_size = index.constant {2 * tiles_per_record * 72} : offset
  %w_storage = buffer.alloca<workgroup> pool(%memory) align(64) %ring_size : buffer
  %w_slots = buffer.view %w_storage[%origin] : buffer -> view<2x{rec}>
  %w_channel = channel.bind<fifo> %w_slots, %c2 : view<2x{rec}>, index -> channel<{rtile}>
  %y_size = index.constant 512 : offset
  %y_storage = buffer.alloca<workgroup> pool(%memory) align(64) %y_size : buffer
  %y_slots = buffer.view %y_storage[%origin] : buffer -> view<2x8x8xf32>
  %y_channel = channel.bind<fifo> %y_slots, %c2 : view<2x8x8xf32>, index -> channel<tile<8x8xf32>>

  pipeline.strand workers([%column, 2], [1, 1], [1, 1]) {{
    %x_first = {x_first_tile_expr}
    %x_byte = index.mul %x_first, %tile_bytes : index
    %x_offset = index.cast %x_byte : index to offset
    %x_external = buffer.view %x_buf[%x_offset] : buffer -> view<{xb}>
    %x_write, %x_payload = channel.reserve %x_channel : channel<tile<{xb}>> -> (write<tile<{xb}>>, view<{xb}>)
    %x_copy = kernel.async.copy %x_external to %x_payload {{cache_scope = workgroup, cache_temporal = regular, direction = global_to_workgroup}} : view<{xb}> to view<{xb}> -> kernel.async.token
    %x_group = kernel.async.group %x_copy : kernel.async.token -> kernel.async.group
    kernel.async.wait %x_group {{newer_groups = 0}} : kernel.async.group
    channel.publish %x_write : write<tile<{xb}>>
    scf.for %record = [%c0 to %records step %c1] {{
      %global = {record_index_expr}
      %byte_index = index.mul %global, %record_bytes : index
      %offset = index.cast %byte_index : index to offset
      %external = buffer.view %w_buf[%offset] : buffer -> view<{rec}>
      %write, %payload = channel.reserve %w_channel : channel<{rtile}> -> (write<{rtile}>, view<{rec}>)
      %copy = kernel.async.copy %external to %payload {{cache_scope = workgroup, cache_temporal = regular, direction = global_to_workgroup}} : view<{rec}> to view<{rec}> -> kernel.async.token
      %group = kernel.async.group %copy : kernel.async.token -> kernel.async.group
      kernel.async.wait %group {{newer_groups = 0}} : kernel.async.group
      channel.publish %write : write<{rtile}>
    }}
  }}

  pipeline.strand workers([%column, 4], [1, 1], [1, 1]) {{
    scf.for %g = [%c0 to %groups step %c1] {{
      %read, %payload = channel.acquire %y_channel : channel<tile<8x8xf32>> -> (read<tile<8x8xf32>>, view<8x8xf32>)
      %out = {out_index_expr}
      %out_index = index.mul %out, %out_bytes : index
      %out_offset = index.cast %out_index : index to offset
      %y_external = buffer.view %y_buf[%out_offset] : buffer -> view<8x8xf32>
      %y_copy = kernel.async.copy %payload to %y_external {{cache_scope = workgroup, cache_temporal = regular, direction = workgroup_to_global}} : view<8x8xf32> to view<8x8xf32> -> kernel.async.token
      %y_group = kernel.async.group %y_copy : kernel.async.token -> kernel.async.group
      kernel.async.wait %y_group {{newer_groups = 0}} : kernel.async.group
      channel.release %read : read<tile<8x8xf32>>
    }}
  }}

  pipeline.strand workers([%column, 3], [1, 1], [1, 1]) {{
    %c8 = index.constant 8 : index
    %x_read, %x_values = channel.acquire %x_channel : channel<tile<{xb}>> -> (read<tile<{xb}>>, view<{xb}>)
    scf.for %g = [%c0 to %groups step %c1] {{
      %zero = vector.splatf 0.0 : vector<64xf32>
      %init = vector.fragment<init> %zero shape [%c8, %c8] : vector<64xf32>
{nl.join(acquire)}
      %y_write, %y_values = channel.reserve %y_channel : channel<tile<8x8xf32>> -> (write<tile<8x8xf32>>, view<8x8xf32>)
      vector.fragment.store<result> {acc}, %y_values[0, 0] shape [%c8, %c8] : vector<64xf32>, view<8x8xf32>
      channel.publish %y_write : write<tile<8x8xf32>>
    }}
    channel.release %x_read : read<tile<{xb}>>
  }}
}}
"""


def compose(public, col, extra=''):
    consts = '\n'.join(f'  %c{i} = index.constant {i} : index' for i in range(COLUMNS))
    comp = '\n'.join(f'  pipeline.compose @{col}[%c{i}](%x, %w, %y) : [index](buffer, buffer, buffer)'
                     for i in range(COLUMNS))
    return f"""
pipeline.def<kernel> public retain target(@npu) @{public}() run(%x: buffer, %w: buffer, %y: buffer) {{
{consts}
{comp}
}}
"""


def m_kernel(k, m):
    """Row split, x resident: K tiles split into two records per group."""
    tiles = k // 8
    assert tiles % 2 == 0 and (tiles // 2) % PER == 0
    half = tiles // 2
    assert m % (8 * COLUMNS) == 0, m
    groups = m // 8 // COLUMNS
    col = f'proj_m_k{k}_m{m}_col'
    dot = f'dot_x{tiles}_r{half}'
    return dot, (half, tiles), column(
        col, dot, tiles, 'index.constant 0 : index', half, 2, groups,
        f'index.add %record, %first_record : index\n      %first_record = index.mul %column, %records : index' if False else
        'index.add %record, %first_record : index',
        'index.add %g, %first_group : index') .replace(
        '  %memory = pipeline.memory',
        '  %first_group = index.mul %column, %groups : index\n'
        '  %first_record = index.mul %first_group, %c2 : index\n'
        '  %memory = pipeline.memory') + compose(f'proj_m_k{k}_m{m}', col)


def k_kernel(k, m):
    """K split: column c owns K / 8 of the tiles of every group."""
    tiles = k // 8
    assert tiles % COLUMNS == 0
    t = tiles // COLUMNS
    assert t % PER == 0, t
    assert m % 8 == 0
    groups = m // 8
    col = f'proj_k_k{k}_m{m}_col'
    dot = f'dot_x{t}_r{t}'
    return dot, (t, t), column(
        col, dot, t, f'index.mul %column, %c{t}_tiles : index', t, 1, groups,
        'index.add %record, %first_record : index',
        'index.add %g, %first_record : index').replace(
        '  %memory = pipeline.memory',
        f'  %c{t}_tiles = index.constant {t} : index\n'
        '  %first_record = index.mul %column, %groups : index\n'
        '  %memory = pipeline.memory') + compose(f'proj_k_k{k}_m{m}', col)


# LFM3 1.7B shapes: QKV + attention gate fused and padded (3136 rows),
# conv in_proj, operator out projections, fused gate/up, head; down by K split.
M_SHAPES = [(2048, 3136), (2048, 6144), (2048, 2048), (2048, 15360), (2048, 128000), (2048, 256)]
K_SHAPES = [(7680, 2048)]

out = ['// Generated by gen_proj.py; do not edit.\n\naie2p.target<array> @npu\n']
dots = {}
bodies = []
for k, m in M_SHAPES:
    dot, args, text = m_kernel(k, m)
    dots[dot] = args
    bodies.append(text)
for k, m in K_SHAPES:
    dot, args, text = k_kernel(k, m)
    dots[dot] = args
    bodies.append(text)
for dot, (rec_tiles, x_tiles) in dots.items():
    out.append(tile_loop(dot, rec_tiles, x_tiles))
out += bodies
text = ''.join(out)
path = sys.argv[1] if len(sys.argv) > 1 else 'lfm3_proj.loom'
open(path, 'w').write(text)
# The engine embeds the program; keep each raw-string piece small.
pieces = [text[i:i + 16000] for i in range(0, len(text), 16000)]
open(path.rsplit('.', 1)[0] + '.inc', 'w').write(
    '// Generated by gen_proj.py; do not edit.\n' +
    '\n'.join(f'R"LOOM({p})LOOM"' for p in pieces) + '\n')
