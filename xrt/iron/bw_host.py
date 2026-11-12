"""Times bw.py's xclbin: python3 bw_host.py <prefix>"""
import sys, time
import numpy as np
import pyxrt
x = sys.argv[1]
dev = pyxrt.device(0); xb = pyxrt.xclbin(x + ".xclbin"); dev.register_xclbin(xb)
ctx = pyxrt.hw_context(dev, xb.get_uuid())
k = pyxrt.kernel(ctx, [n.get_name() for n in xb.get_kernels() if n.get_name().startswith("MLIR_AIE")][0])
ins = np.fromfile(x + ".bin", np.uint32)
bi = pyxrt.bo(dev, ins.nbytes, pyxrt.bo.cacheable, k.group_id(1)); bi.write(ins.tobytes(), 0); bi.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
ba = pyxrt.bo(dev, 1 << 30, pyxrt.bo.host_only, k.group_id(3))
ts = []
for _ in range(5):
    t0 = time.perf_counter(); k(3, bi, ins.size, ba).wait(); ts.append(time.perf_counter() - t0)
print(f"{x}: {min(ts) * 1e3:.1f} ms, {(1 << 30) / min(ts) / 1e9:.1f} GB/s")
