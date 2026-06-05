# Analytical GPU-vs-CPU crossover model for the seq-len sweep.
#
# Models per-generate transformer latency as a quadratic in sequence length M:
#
#     T(M) = c  +  a*M  +  b*M^2
#            |      |       |
#            |      |       attention (scores + A.V): 4*L*d*steps FLOP per M^2
#            |      FC (QKV/O proj + FFN + out-proj): 2*N_lin*steps FLOP per token
#            launch/sync overhead per generate
#
# Coefficients tie back to hardware via effective throughputs:
#     a = 2*N_lin*steps / P_fc        (int8 GEMM path)
#     b = 4*L*d*steps   / P_attn      (f32 NEON on CPU, fp16 flash-attn on GPU)
#
# Fit (a, b, c) per backend by exact solve through three measured sweep points,
# hold out the remaining point for validation, then derive:
#   - effective P_fc / P_attn (and % of the roofline.py peak estimates)
#   - crossover M* where the GPU's quadratic advantage beats its linear
#     handicap + overhead:  (b_c - b_g)*M^2 = (a_g - a_c)*M + (c_g - c_c)
#   - gain(M) = T_cpu/T_gpu table, with asymptote b_c/b_g as M -> inf
#
# Stdlib-only, like roofline.py. Data: README.md tables in this directory
# (CPU = Pixel 9 XNNPACK Q8 SMMLA; GPU = Pixel 9 Mali-G715 OpenCL GQ8,
# int8-dot FC + fp16 flash-attention + LN-affine fusion, transformer-only).

import math

# ---- model dims (MaskGIT-256, fixed across the sweep) ----
L, D, FFN, VOCAB, STEPS = 24, 768, 3072, 2025, 8

N_LIN = L * (4 * D * D + 2 * D * FFN) + D * VOCAB        # MACs/token/forward
LIN_FLOP_PER_TOKEN = 2 * N_LIN * STEPS                   # FLOP per token per generate
ATTN_FLOP_PER_M2 = 4 * L * D * STEPS                     # FLOP per M^2 per generate

# ---- hardware peak estimates (same order-of-magnitude numbers as roofline.py) ----
PEAK = {
    "cpu_fc":   0.6e12,   # Cortex-X3+A715 SMMLA int8, ~0.6 TOP/s effective ceiling
    "cpu_attn": 0.2e12,   # big-core f32 NEON FMA, ~0.2 TFLOP/s
    "gpu_fc":   1.3e12,   # Mali-G715 arm_dot int8 ~ fp32 FMA rate, ~1.3 TOP/s
    "gpu_attn": 2.6e12,   # Mali-G715 fp16 ALU rate, ~2.6 TFLOP/s
}

# ---- measured transformer-only latency, seconds per generate (x8 steps) ----
# CPU: fit on (65, 257, 1025); 4097 held out (5+ min sustained -> thermal throttle).
# GPU: fit on (257, 1025, 4097); 65 held out (measured with fp32-FA kernel).
CPU_MEAS = {65: 0.773, 257: 2.985, 1025: 22.198, 4097: 319.657}
GPU_MEAS = {65: 1.697, 257: 4.653, 1025: 19.103, 4097: 109.737}
CPU_FIT_PTS, CPU_HELD = (65, 257, 1025), 4097
GPU_FIT_PTS, GPU_HELD = (257, 1025, 4097), 65


def fit_quadratic(points, meas):
    """Exact solve of c + a*M + b*M^2 through three (M, T) points."""
    rows = [[1.0, m, float(m) * m, meas[m]] for m in points]
    # Gaussian elimination, 3x3
    for i in range(3):
        piv = max(range(i, 3), key=lambda r: abs(rows[r][i]))
        rows[i], rows[piv] = rows[piv], rows[i]
        for r in range(i + 1, 3):
            f = rows[r][i] / rows[i][i]
            for k in range(i, 4):
                rows[r][k] -= f * rows[i][k]
    x = [0.0] * 3
    for i in (2, 1, 0):
        x[i] = (rows[i][3] - sum(rows[i][k] * x[k] for k in range(i + 1, 3))) / rows[i][i]
    c, a, b = x
    return a, b, c


def T(a, b, c, m):
    return c + a * m + b * m * m


a_c, b_c, c_c = fit_quadratic(CPU_FIT_PTS, CPU_MEAS)
a_g, b_g, c_g = fit_quadratic(GPU_FIT_PTS, GPU_MEAS)

print("=== FITTED COEFFICIENTS  T(M) = c + a*M + b*M^2  (s per generate, x8 steps) ===")
print(f"            {'a (ms/token)':>14}  {'b (us/M^2)':>12}  {'c overhead (s)':>15}")
print(f" CPU XNN Q8 {a_c*1e3:14.3f}  {b_c*1e6:12.3f}  {c_c:15.3f}")
print(f" GPU gq8 FA {a_g*1e3:14.3f}  {b_g*1e6:12.3f}  {c_g:15.3f}")

print("\n=== EFFECTIVE THROUGHPUTS (backed out of a, b) ===")
for name, coef, flop, peak_key in (
    ("CPU FC   (int8 SMMLA)", a_c, LIN_FLOP_PER_TOKEN, "cpu_fc"),
    ("CPU attn (f32 NEON)  ", b_c, ATTN_FLOP_PER_M2,   "cpu_attn"),
    ("GPU FC   (int8 dot)  ", a_g, LIN_FLOP_PER_TOKEN, "gpu_fc"),
    ("GPU attn (fp16 FA)   ", b_g, ATTN_FLOP_PER_M2,   "gpu_attn"),
):
    p_eff = flop / coef
    print(f" {name}: {p_eff/1e9:6.0f} GFLOP/s effective"
          f"   (~{100*p_eff/PEAK[peak_key]:.0f}% of ~{PEAK[peak_key]/1e12:.1f}T peak est.)")
print(f"   FC ratio GPU/CPU   = {(LIN_FLOP_PER_TOKEN/a_g)/(LIN_FLOP_PER_TOKEN/a_c):.2f}x"
      f"  -> no GPU headroom on the linear term (int8 ceiling parity)")
print(f"   attn ratio GPU/CPU = {(ATTN_FLOP_PER_M2/b_g)/(ATTN_FLOP_PER_M2/b_c):.2f}x"
      f"  -> all GPU gain lives in the quadratic term")

# ---- crossover: (b_c - b_g) M^2 = (a_g - a_c) M + (c_g - c_c) ----
db, da, dc = b_c - b_g, a_g - a_c, c_g - c_c
m_star = (da + math.sqrt(da * da + 4 * db * dc)) / (2 * db)
# overhead-free intuition: where attention starts to dominate the CPU's clock
m_eq_cpu = (a_c / b_c)

print("\n=== CROSSOVER ===")
print(f" predicted M* = {m_star:.0f}   (measured: GPU loses at 257, wins at 1025)")
print(f" CPU clock becomes attention-dominated at M = a_c/b_c = {m_eq_cpu:.0f}")
print(f"   (vs FLOP-equality at M = N_lin/(2*L*d) = {N_LIN/(2*L*D):.0f} -- time-equality"
      f" lands {N_LIN/(2*L*D)/m_eq_cpu:.0f}x earlier because f32 attn is the CPU's slow path)")

print("\n=== GAIN TABLE  gain(M) = T_cpu / T_gpu ===")
print(f" {'M':>6}  {'T_cpu pred':>11}  {'T_gpu pred':>11}  {'gain':>6}   measured")
sweep = sorted(set([65, 257, 512, round(m_star), 1025, 2048, 4097, 8192, 16384]))
for m in sweep:
    tc, tg = T(a_c, b_c, c_c, m), T(a_g, b_g, c_g, m)
    meas = ""
    if m in CPU_MEAS and m in GPU_MEAS:
        held = []
        if m == CPU_HELD:
            held.append("CPU held-out, thermal")
        if m == GPU_HELD:
            held.append("GPU held-out, fp32 FA")
        note = f"  ({'; '.join(held)})" if held else ""
        meas = f"{CPU_MEAS[m]/GPU_MEAS[m]:.2f}x{note}"
    print(f" {m:>6}  {tc:>10.2f}s  {tg:>10.2f}s  {tc/tg:>5.2f}x   {meas}")
print(f" {'inf':>6}  {'':>11}  {'':>11}  {b_c/b_g:>5.2f}x   (asymptote = b_cpu/b_gpu"
      f" = P_attn_gpu/P_attn_cpu)")

print("\n=== HELD-OUT VALIDATION ===")
tc4 = T(a_c, b_c, c_c, CPU_HELD)
tg65 = T(a_g, b_g, c_g, GPU_HELD)
print(f" CPU M={CPU_HELD}: predicted {tc4:6.1f} s   measured {CPU_MEAS[CPU_HELD]:6.1f} s"
      f"  ({100*(CPU_MEAS[CPU_HELD]-tc4)/tc4:+.0f}%; thermal throttle -- M1 Max host: 207.2 s)")
print(f" GPU M={GPU_HELD}:   predicted {tg65:6.2f} s   measured {GPU_MEAS[GPU_HELD]:6.2f} s"
      f"  ({100*(GPU_MEAS[GPU_HELD]-tg65)/tg65:+.0f}%; measured kernel was fp32 FA)")
