# Roofline estimate: MaskGIT-256 on Pixel 9 / Mali-G715 (Tensor G4).
S=257; E=768; FFN=3072; L=24; H=16; HD=48; VOCAB=2025; STEPS=8

# ---- Mali-G715 (Tensor G4) hardware estimates (order-of-magnitude; vendor-undocumented) ----
TFLOPS_F32=1.3e12      # ~1.3 TFLOP/s FP32 (MP7 @ ~0.94 GHz)
TFLOPS_F16=2.6e12      # ~2x for fp16
BW=60e9                # ~60 GB/s shared LPDDR5X (theoretical; effective lower)

def t_comp(flop, tflops): return flop/tflops
def t_mem(bytes_, bw): return bytes_/bw

# ---- Transformer, per forward ----
# FC matmul MACs: QKV+O proj = 4*E*E ; FFN = 2*E*FFN ; output proj = E*VOCAB ; all *S
fc_mac = S*(4*E*E + 2*E*FFN + E*VOCAB)
attn_mac = L*0  # added below
# attention scores + (A·V): per layer  H*S*S*HD * 2
attn_mac = H*S*S*HD*2
fc_flop = 2*fc_mac*L
attn_flop = 2*attn_mac*L
tf_flop = fc_flop + attn_flop                       # per forward
# weights (Q8_0 ~1.06 B/param): attn+ffn+outproj params
params_fc = 4*E*E + 2*E*FFN + E*VOCAB
w_bytes_q8 = params_fc*L*1.0625                      # Q8_0 bytes (read >=1x/forward)
mtiles = -(-S//64)                                   # ceil(S/64) tiled weight re-reads
print("=== TRANSFORMER (per forward) ===")
print(f" FC FLOP   = {fc_flop/1e9:6.1f} G   attn FLOP = {attn_flop/1e9:5.1f} G   total = {tf_flop/1e9:6.1f} GFLOP")
print(f" Q8_0 weights = {w_bytes_q8/1e6:6.0f} MB  (tiled re-read x{mtiles} = {w_bytes_q8*mtiles/1e6:.0f} MB)")
print(f" roofline/forward:  compute(fp16) {t_comp(tf_flop,TFLOPS_F16)*1e3:5.1f} ms   mem(reread) {t_mem(w_bytes_q8*mtiles,BW)*1e3:5.1f} ms")
print(f" x{STEPS} steps IDEAL  =  compute {t_comp(tf_flop,TFLOPS_F16)*STEPS:5.2f} s   mem {t_mem(w_bytes_q8*mtiles,BW)*STEPS:5.2f} s")
print(f" MEASURED transformer (8 fwd) on Mali = 15.0 s")
print(f" -> achieved efficiency ~= {100*t_comp(tf_flop,TFLOPS_F16)*STEPS/15.0:.1f}% of fp16 compute roofline")

# ---- VQGAN decoder (once): conv MACs over the upsampling pyramid ----
# levels (res, channels), each ~ (num_res_blocks+1)=3 resblocks * 2 convs(3x3, C->C)
levels=[(16,512),(32,256),(64,256),(128,128),(256,128)]
vq_mac=0
for R,C in levels:
    vq_mac += 3*2 * (R*R*C*C*9)
vq_flop=2*vq_mac
vq_w_bytes=72e6*4   # ~72M params, F32 conv weights
print("\n=== VQGAN DECODER (once) ===")
print(f" conv FLOP = {vq_flop/1e9:6.1f} GFLOP   weights(F32) = {vq_w_bytes/1e6:.0f} MB")
print(f" roofline IDEAL: compute(fp32) {t_comp(vq_flop,TFLOPS_F32):5.2f} s   weight-mem {t_mem(vq_w_bytes,BW)*1e3:.0f} ms")
print(f" MEASURED VQGAN on Mali = 6.3 s  -> efficiency ~= {100*t_comp(vq_flop,TFLOPS_F32)/6.3:.1f}%")

# ---- CPU (XNNPACK int8) for contrast ----
# Tensor G4 big core(s) with i8mm: assume ~0.4 TOP/s effective int8 (conservative, 1-2 cores)
TOPS_I8=0.6e12
print("\n=== CPU (XNNPACK int8, i8mm) contrast ===")
print(f" transformer int8 MAC x8 = {fc_mac*L*STEPS/1e9:.0f} G   @ ~{TOPS_I8/1e12:.1f} TOP/s -> {fc_mac*L*STEPS/TOPS_I8:.1f} s ideal  (measured ~3 s)")
print(" -> the CPU runs near its int8 roofline; the GPU runs at single-digit % of its fp roofline.")
