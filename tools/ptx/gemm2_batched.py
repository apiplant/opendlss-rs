"""A batched variant of upstream's gemm2_e4m3 (scripts/ptx/gemm2_e4m3.py): grid z selects a batch, whose input
columns, output columns and weight matrix are offset by per-batch strides. Nothing else changes - the generator
is upstream's, with three parameters appended and the three loads they offset rewritten - so every batch computes
exactly the bytes a separate gemm2 launch on that slice would.

Used for the 512-channel stage's branch MLP (eight independent 64 -> 256 -> 64 branches), which upstream runs as a
GLSL kernel with no PTX twin: two batched launches per block instead of sixteen.

python gemm2_batched.py <upstream scripts/ptx> <K> <flags> out.ptx
"""
import sys

ptx_dir, K, flags, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
sys.path.insert(0, ptx_dir)
import ptxgen  # noqa: E402
import gemm2_e4m3  # noqa: E402

STRIDES = {"inputColumnBase": "batchInputStride", "outputColumnOffset": "batchOutputStride", "pW": "batchWeightStride"}

original_entry = ptxgen.Ptx.entry
original_u32 = ptxgen.Ptx.load_param_u32
original_u64 = ptxgen.Ptx.load_param_u64


def entry(self, name, params, *args, **kwargs):
    params = list(params) + [("u32", stride) for stride in STRIDES.values()]
    return original_entry(self, name.replace("gemm2_", "gemm2b_"), params, *args, **kwargs)


def batch(self):
    if not hasattr(self, "_batch"):
        self._batch = self.special("ctaid.z")
    return self._batch


def load_u32(self, name):
    value = original_u32(self, name)
    if name in STRIDES:
        stride = original_u32(self, STRIDES[name])
        offset = self.reg("b32")
        self.emit(f"mad.lo.u32 {offset}, {batch(self)}, {stride}, {value};")
        return offset
    return value


def load_u64(self, name):
    value = original_u64(self, name)
    if name in STRIDES:
        stride = original_u32(self, STRIDES[name])
        bytes32 = self.reg("b32")
        self.emit(f"mul.lo.u32 {bytes32}, {batch(self)}, {stride};")
        wide = self.reg("b64")
        self.emit(f"cvt.u64.u32 {wide}, {bytes32};")
        offset = self.reg("b64")
        self.emit(f"add.u64 {offset}, {value}, {wide};")
        return offset
    return value


ptxgen.Ptx.entry = entry
ptxgen.Ptx.load_param_u32 = load_u32
ptxgen.Ptx.load_param_u64 = load_u64

name, text = gemm2_e4m3.generate(K, flags)
assert "ctaid.z" in text and name.startswith("gemm2_")
open(out, "w").write(text)
print(name.replace("gemm2_", "gemm2b_"))
