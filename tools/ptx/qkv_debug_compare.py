"""Decodes tools/ptx/qkv_debug.py's words for one query and compares each stage with the reference operands
tools/verify_block0/verify_expert_block.cpp dumped for it (NR_DUMP_OPERANDS).

python qkv_debug_compare.py <debug.bin> <operands.bin> <query local index>
"""
import struct, sys
import numpy as np

debug = np.fromfile(sys.argv[1], dtype=np.uint32).reshape(400, 128)
ops = np.fromfile(sys.argv[2], dtype=np.float32)
query = int(sys.argv[3])
q_ref, keys_ref = ops[:32], ops[32:32 + 2048].reshape(64, 32)
at = 32 + 2048
prior, scores_ref = ops[at:at + 64], ops[at + 64:at + 128]
at += 128
values_ref = ops[at:at + 2048].reshape(64, 32)
at += 2048
weights_ref, attended_ref = ops[at:at + 64], ops[at + 64:at + 96]

halves = lambda word: np.array([word & 0xffff, word >> 16], dtype=np.uint16).view(np.float16).astype(np.float32)


def e4(value):
    """The E4M3 publication of a half (RNE, saturating at 448, NaN -> 0), as the reference's fp8Domain."""
    if value != value: return 0.0
    m = min(abs(value), 448.0)
    if m == 0: return 0.0 * np.sign(value)
    if m < 2 ** -6: q = round(m * 512) / 512
    else:
        bits = struct.unpack("<I", struct.pack("<f", m))[0]
        bits = (bits + 0x7ffff + ((bits >> 20) & 1)) & 0xfff00000
        q = struct.unpack("<f", struct.pack("<I", bits))[0]
    return -q if value < 0 else q


def row_values(slot_of, tiles, row):
    """Row `row` (0..63) of a 16 x (8 tiles) C fragment set, per warp: warp row // 16, g = row % 8, h = (row % 16) // 8."""
    warp, g, h = row // 16, row % 8, (row % 16) // 8
    out = np.zeros(8 * tiles, dtype=np.float32)
    for j in range(tiles):
        for t in range(4):
            lo, hi = halves(debug[slot_of(j, h)][warp * 32 + g * 4 + t])
            out[8 * j + 2 * t], out[8 * j + 2 * t + 1] = lo, hi
    return out


def tiled(token):
    x, y = token & 7, token >> 3
    return (y >> 2) * 32 + (x >> 2) * 16 + (y & 3) * 4 + (x & 3)


def report(name, got, want):
    bad = np.nonzero(got != want)[0]
    print(f"{name:34s} {'equal' if not len(bad) else f'{len(bad)} differ, at {bad[:8].tolist()}'}")
    for i in bad[:4]:
        print(f"    [{i}] kernel {got[i]!r} reference {want[i]!r}")
    return len(bad)


q = np.array([e4(v) for v in row_values(lambda j, h: 2 * j + h, 4, query)])
report("q normalized (E4)", q, q_ref)
k_bad = v_bad = 0
for token in range(64):
    k = np.array([e4(v) for v in row_values(lambda j, h: 8 + 2 * (4 * h + j), 4, token)])
    v = np.array([e4(v) for v in row_values(lambda j, h: 9 + 2 * (4 * h + j), 4, token)])
    k_bad += (k != keys_ref[tiled(token)]).sum()
    v_bad += (v != values_ref[tiled(token)]).sum()
print(f"{'k normalized (E4), 64 keys':34s} {'equal' if not k_bad else f'{k_bad} values differ'}")
print(f"{'v (E4), 64 keys':34s} {'equal' if not v_bad else f'{v_bad} values differ'}")
report("scores (physical key order)", row_values(lambda m, h: 100 + 2 * m + h, 8, query), scores_ref)
w = np.array([e4(v) for v in row_values(lambda m, h: 24 + 2 * m + h, 8, query)])
report("softmax weights (E4)", w, weights_ref)
report("attended (f16, before E4)", row_values(lambda j, h: 40 + 2 * j + h, 4, query), attended_ref)

# ---- the first divergence in detail: the differing K value, its raw row and its norm
f16 = lambda x: float(np.float16(x))
for token in range(64):
    kq = row_values(lambda j, h: 8 + 2 * (4 * h + j), 4, token)
    k = np.array([e4(v) for v in kq])
    bad = np.nonzero(k != keys_ref[tiled(token)])[0]
    if not len(bad):
        continue
    raw = row_values(lambda j, h: 208 + 2 * j + h, 4, token)
    c = bad[0]
    print(f"\nkey token {token} (physical {tiled(token)}) channel {c}: normalized half {kq[c]!r} -> E4 {k[c]}, reference E4 {keys_ref[tiled(token)][c]}")
    # the norm, recomputed in the specified order from the kernel's own raw K (f16 steps)
    r = [f16(f16(raw[i] * raw[i]) if False else 0) for i in range(16)]
    r = [f16(np.float64(raw[i]) * raw[i] + f16(raw[i + 16] * raw[i + 16])) for i in range(16)]
    for stride in (8, 4, 2, 1):
        r = [f16(r[i] + r[i + stride]) for i in range(stride)] + r[stride:]
    s1 = r[0]
    exact = np.float32(1.0) / np.sqrt(np.float32(s1))
    norm_exact = f16(exact)
    implied = kq[c] / raw[c]
    print(f"  raw k[{c}] = {raw[c]!r}, row square sum {s1!r}")
    print(f"  1/sqrt(sum) = {float(exact)!r}: correctly rounded to half {norm_exact!r}")
    bits = np.float32(exact).view(np.uint32)
    print(f"  distance to the half rounding boundary: {abs(int(bits & 0x1fff) - 0x1000)} f32 ulps")
    print(f"  the kernel's normalized value implies a norm of about {implied!r}; raw * correct norm = {f16(raw[c] * norm_exact)!r}")

# ---- optional: the reference's raw QKV row of a token, against the kernel's raw K of it
if len(sys.argv) > 5:
    token, raw_ref = int(sys.argv[4]), np.fromfile(sys.argv[5], dtype=np.float32)
    head = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    kernel_q = row_values(lambda j, h: 200 + 2 * j + h, 4, token)
    kernel_k = row_values(lambda j, h: 208 + 2 * j + h, 4, token)
    ref_q, ref_k = raw_ref[head * 96:head * 96 + 32], raw_ref[head * 96 + 32:head * 96 + 64]
    print(f"\nraw q of token {token}: {int((kernel_q != ref_q).sum())} of 32 differ; raw k: {int((kernel_k != ref_k).sum())} of 32 differ")
    for c in np.nonzero(kernel_k != ref_k)[0][:6]:
        print(f"  k[{c}]: kernel {kernel_k[c]!r} reference {ref_k[c]!r}")
