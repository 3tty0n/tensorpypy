import os
import sys

if sys.argv[1] == "--cc":
    import torch
    major, minor = torch.cuda.get_device_capability()
    open(sys.argv[2], "w").write(str(major * 10 + minor))
    sys.exit(0)

import triton
from triton.backends.compiler import GPUTarget

ttir, ptx, meta = sys.argv[1:4]
cc = sys.argv[4] if len(sys.argv) > 4 else "auto"
if cc == "auto":
    import torch
    major, minor = torch.cuda.get_device_capability()
    cc = major * 10 + minor
cc = int(cc)
num_warps = int(sys.argv[5]) if len(sys.argv) > 5 else 4
num_stages = int(sys.argv[6]) if len(sys.argv) > 6 else 3
c = triton.compile(ttir, target=GPUTarget("cuda", cc, 32),
                   options={"num_warps": num_warps, "num_stages": num_stages})
md = c.metadata
open(ptx, "w").write(c.asm["ptx"])
# Older drivers refuse to JIT Triton's PTX (CUDA_ERROR_UNSUPPORTED_PTX_VERSION),
# so prefer the prebuilt cubin and fall back to PTX only when it is missing.
cubin_path = ptx[:-4] + ".cubin" if ptx.endswith(".ptx") else ptx + ".cubin"
if os.path.exists(cubin_path):
    os.unlink(cubin_path)
cubin = c.asm.get("cubin")
if cubin:
    open(cubin_path, "wb").write(cubin)
extra = 2
open(meta, "w").write("%d %d %d %d\n" % (md.num_warps * 32, md.shared, extra, cc))
