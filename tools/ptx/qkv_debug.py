"""An instrumented build of upstream's qkv_e4m3 (scripts/ptx/qkv_e4m3.py) for finding where it departs from the
reference: for one item (the CTA whose index equals `debugItem`; launch it with grid = itemCount = item + 1, so
every CTA runs exactly one item), every f16x2 register that is published to E4M3, the raw Q/K accumulators and the
scores before the softmax are stored to `pDebug` as [slot][128 threads] words. The arithmetic is upstream's,
untouched: the stores only read registers.

Slots: 0..7 normalized q (tile j, half h: 2j + h); 8..23 k and v, interleaved (8 + 2 (4h + j) + {0 k, 1 v});
24..39 softmax weights (24 + 2m + h, m the 8-key tile); 40..47 attended output (40 + 2j + h);
100..115 scores (100 + 2m + h); 200..215 raw q then k accumulators before normalization (200 + 8 call + 2j + h).

python qkv_debug.py <upstream scripts/ptx> <C> out.ptx
"""
import sys

ptx_dir, C, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
sys.path.insert(0, ptx_dir)
import ptxgen  # noqa: E402
import swin  # noqa: E402
import qkv_e4m3  # noqa: E402

original_entry = ptxgen.Ptx.entry
state = {"cvt": 0, "normalize": 0}


def entry(self, name, params, *args, **kwargs):
    return original_entry(self, name.replace("qkv_", "qkv_debug_"), list(params) + [("u64", "pDebug"), ("u32", "debugItem")], *args, **kwargs)


def store(p, slot, reg):
    # recomputed at every store (cheap, and valid wherever the store lands in the item loop)
    tid = p.special("tid.x"); cta = p.special("ctaid.x")
    item = p.reg("b32"); p.emit(f"ld.param.u32 {item}, [debugItem];")
    base = p.reg("b64"); p.emit(f"ld.param.u64 {base}, [pDebug];")
    on = p.setp("eq.u32", cta, item)
    off = p.reg("b32"); p.emit(f"mad.lo.u32 {off}, {tid}, 4, {slot * 512};")
    wide = p.reg("b64"); p.emit(f"cvt.u64.u32 {wide}, {off};")
    addr = p.reg("b64"); p.emit(f"add.u64 {addr}, {base}, {wide};")
    p.emit(f"@{on} st.global.b32 [{addr}], {reg};")


original_cvt = swin.cvt_e4x2


def cvt_e4x2(p, v):
    store(p, state["cvt"], v)
    state["cvt"] += 1
    return original_cvt(p, v)


original_softmax = qkv_e4m3.softmax


def softmax(p, s_tiles, li, k):
    for m in range(8):
        for h in range(2):
            store(p, 100 + 2 * m + h, s_tiles[m][h])
    return original_softmax(p, s_tiles, li, k)


original_normalize = qkv_e4m3.normalize


def normalize(p, tiles, li, zero, scale2=None, guarded=True):
    call = state["normalize"]
    state["normalize"] += 1
    for j in range(4):
        for h in range(2):
            store(p, 200 + 8 * call + 2 * j + h, tiles[j][h])
    return original_normalize(p, tiles, li, zero, scale2, guarded)


ptxgen.Ptx.entry = entry
swin.cvt_e4x2 = cvt_e4x2
qkv_e4m3.cvt_e4x2 = cvt_e4x2
qkv_e4m3.softmax = softmax
qkv_e4m3.normalize = normalize

name, text = qkv_e4m3.generate(C, 80)
assert state["cvt"] == 48 and state["normalize"] == 2, state
open(out, "w").write(text)
print(name.replace("qkv_", "qkv_debug_"))
