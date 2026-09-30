// The reference port's arithmetic self-test, run through this crate's wgpu/naga stack instead of a browser.
//
// web/fixtures/numerics.bin holds what the Vulkan implementation's CPU reference produced for every scalar
// primitive the kernels are built from, exhaustively over all 65 536 halves where the domain allows it. This
// draws the same inputs as the port does, has `opendlss-nr numerics-selftest` evaluate shaders/selftest.wgsl
// on the GPU, and compares with the port's own `compare`:
//
//   node tools/check_numerics.mjs [path/to/browser-webgpu] [path/to/opendlss-nr binary]
import { readFileSync, writeFileSync, mkdtempSync, rmSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const port = resolve(process.argv[2] ?? '../OpenDLSS-NR/ports/browser-webgpu');
const binary = resolve(process.argv[3] ?? 'target/release/opendlss-nr');
const { readFixture, numericsCases, compare } =
  await import(pathToFileURL(join(port, 'src/numerics_cases.js')).href);

const cases = numericsCases(readFixture(new Uint8Array(readFileSync(join(port, 'web/fixtures/numerics.bin')))));
// The same f32 -> half reference, through WGSL's native f16() conversion (see src/selftest.rs).
const f16Case = cases.find((entry) => entry.entryPoint === 'case_f16_bits');
cases.push({ ...f16Case, name: 'native f16() over arbitrary f32 patterns', entryPoint: 'case_native_f16_bits' });
// Random patterns almost never land on a rounding boundary, which is where a conversion can be wrong: every
// half, and the f32s at, just below and just above the midpoint to the next half, against the JS oracle.
{
  const num = await import(pathToFileURL(join(port, 'src/numerics.js')).href);
  const inputs = [];
  for (let bits = 0; bits < 0x7c00; ++bits) {
    for (const sign of [0, 0x8000]) {
      const low = num.f32Bits(num.f16ToNumber(bits | sign));
      const high = num.f32Bits(num.f16ToNumber((bits + 1) | sign));
      const mid = num.f32Bits((num.f16ToNumber(bits | sign) + num.f16ToNumber((bits + 1) | sign)) / 2);
      inputs.push(low, mid - 1, mid, mid + 1);
      if (high === undefined) throw new Error('unreachable');
    }
  }
  const gpuInputs = Uint32Array.from(inputs);
  const expected = gpuInputs.map((word) => num.f16Bits(num.f32FromBits(word)));
  cases.push({ name: 'native f16() at every rounding boundary (JS oracle)', count: gpuInputs.length, expected,
               kind: 'half', gpuInputs, entryPoint: 'case_native_f16_bits', oracle: 'js' });
  // The window norm's half fma, fma(x, x, f16(y^2)), against an exact oracle: every value is a multiple of 2^-48.
  {
    const rng = new (await import(pathToFileURL(join(port, 'src/numerics_cases.js')).href)).Xorshift(0x5eed);
    const half = (bits) => { const e = (bits >> 10) & 31, m = bits & 1023, s = bits & 0x8000 ? -1n : 1n;
      return e ? s * BigInt(1024 + m) << BigInt(e - 1) : s * BigInt(m); };            // value * 2^24
    const toHalf = (n) => {   // n * 2^-48 -> the nearest-even half's bits
      const sign = n < 0n ? 0x8000 : 0; let m = n < 0n ? -n : n;
      if (m === 0n) return sign;
      let e = m.toString(2).length - 1 - 48;                                            // floor(log2 value)
      e = Math.max(e, -14);
      const shift = BigInt(48 + e - 10);                                                // units of the half's last bit
      let q = m >> shift; const r = m - (q << shift), halfway = 1n << (shift - 1n);
      if (r > halfway || (r === halfway && (q & 1n))) q += 1n;
      if (q >= 2048n) { q >>= 1n; e += 1; }
      if (e > 15) return sign | 0x7c00;
      return q < 1024n ? sign | Number(q) : sign | ((e + 15) << 10) | Number(q - 1024n);
    };
    const n = 1 << 17, words = new Uint32Array(n + 1), expected = new Uint16Array(n);
    for (let i = 0; i < n; ++i) {
      let x, y;
      do { x = rng.next() & 0xffff; } while ((x & 0x7c00) === 0x7c00 || ((x >> 10) & 31) > 20);
      do { y = rng.next() & 0xffff; } while ((y & 0x7c00) === 0x7c00);
      // c = f16(y^2), computed exactly then rounded, as the kernel's highSquare
      const c = toHalf(half(y) * half(y));
      if ((c & 0x7c00) === 0x7c00) { --i; continue; }
      words[i] = x | (c << 16);
      expected[i] = toHalf(half(x) * half(x) + (half(c) << 24n));
    }
    cases.push({ name: 'window norm half fma, correctly rounded (exact oracle)', count: n, expected, kind: 'half',
                 gpuInputs: words, entryPoint: 'case_native_norm_fma', oracle: 'js' });
  }
  cases.push({ name: 'f16Bits at every rounding boundary (JS oracle)', count: gpuInputs.length, expected,
               kind: 'half', gpuInputs, entryPoint: 'case_f16_bits', oracle: 'js' });
}
const work = mkdtempSync(join(tmpdir(), 'nr-numerics-'));
const list = cases.map((entry, i) => {
  const inputs = entry.gpuInputs ?? new Uint32Array(1);
  writeFileSync(join(work, `${i}.in`), Buffer.from(inputs.buffer, inputs.byteOffset, inputs.byteLength));
  return `${i} ${entry.entryPoint} ${entry.count}`;
});
writeFileSync(join(work, 'cases.txt'), list.join('\n') + '\n');
execFileSync(binary, ['numerics-selftest', '--dir', work], { stdio: 'inherit' });

let failures = 0;
cases.forEach((entry, i) => {
  // Copied out: a small file read by Node can sit at an offset inside a shared pool.
  const words = new Uint32Array(Uint8Array.from(readFileSync(join(work, `${i}.out`))).buffer);
  const produced = (j) => (entry.kind === 'byte' ? words[j] & 0xff : entry.kind === 'half' ? words[j] & 0xffff : words[j]);
  const result = compare(entry, produced);
  if (result.mismatches) failures += 1;
  console.log(`${result.mismatches ? 'FAIL' : 'pass'}  ${entry.name.padEnd(48)} ` +
              `${entry.count - result.skipped - result.mismatches}/${entry.count - result.skipped} exact` +
              (result.mismatches ? `, first mismatches: ${JSON.stringify(result.first)}` : ''));
});
// The GEMM's SiLU tables, built on the device: per half input, the activation as a half and as its E4M3
// publication (as a half), and the same publication as the packed byte.
{
  const num = await import(pathToFileURL(join(port, 'src/numerics.js')).href);
  const silu = new Uint16Array(Uint8Array.from(readFileSync(join(work, 'silu.bin'))).buffer);
  const packed = readFileSync(join(work, 'packed_silu.bin'));
  const oracle = num.siluTable();
  let checked = 0, bad = 0, first = null;
  for (let bits = 0; bits < 65536; ++bits) {
    if ((bits & 0x7c00) === 0x7c00) continue;   // non-finite inputs never reach an activation
    checked += 1;
    const code = num.e4m3FromF16Bits(oracle[bits]);
    const want = [oracle[bits], num.f16Bits(num.e4m3ToNumber(code)), code];
    const got = [silu[bits * 2], silu[bits * 2 + 1], packed[bits]];
    // The port's table builder publishes a zero activation as +0 whatever its sign, and the port is bit-exact
    // against native with that table, so the sign of a zero is not held against it.
    const zero = [0x7fff, 0x7fff, 0x7f];
    if (want.some((w, k) => w !== got[k] && ((w | got[k]) & zero[k]) !== 0)) { bad += 1; first ??= { bits, want, got }; }
  }
  if (bad) failures += 1;
  console.log(`${bad ? 'FAIL' : 'pass'}  ${'GEMM SiLU tables over every finite half'.padEnd(48)} ` +
              `${checked - bad}/${checked} exact${bad ? `, first: ${JSON.stringify(first)}` : ''}`);
}
rmSync(work, { recursive: true, force: true });
console.log(failures ? `NUMERICS FAIL: ${failures} case(s)` : 'NUMERICS PASS');
process.exit(failures ? 1 : 0);
