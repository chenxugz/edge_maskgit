// mg-opencl.cpp — OpenCL graph executor + F32 kernels (foundation).
//
// Implemented ops: MulMat (2D contiguous, mg layout), Add/Mul (broadcast),
// Scale, Gelu, Silu, Reshape (buffer alias). Other ops throw "not implemented"
// so the backend grows op-by-op, each validated against the reference oracle.
#define CL_SILENCE_DEPRECATION 1
#if defined(__APPLE__)
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "mg-opencl.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace mg {
namespace {

void ck(cl_int e, const char* what) {
    if (e != CL_SUCCESS) throw std::runtime_error(std::string("opencl: ") + what + " (" + std::to_string(e) + ")");
}

// All kernels share mg's layout: ne[0] is the innermost/contiguous dimension.
const char* kKernels = R"CLC(
// fused matmul-epilogue activation: 0=none, 1=gelu (exact erf), 2=silu (swish).
inline float mg_act(float v, int a) {
    if (a == 1) return 0.5f*v*(1.0f + erf(v*0.70710678f));
    if (a == 2) return v/(1.0f + exp(-v));
    return v;
}
// Stride-aware batched matmul: w[K,N,wb2,wb3], x[K,M,b2,b3] -> out[N,M,b2,b3]
// (out contiguous). strides are in ELEMENTS; w broadcasts over batch dims of size 1.
// out[n,m,p2,p3] = sum_k w[k,n,*] * x[k,m,p2,p3].
__kernel void k_mul_mat(__global const float* w, __global const float* x, __global float* o,
                        int K, int N, int M, int B2,
                        int ws0,int ws1,int ws2,int ws3, int xs0,int xs1,int xs2,int xs3,
                        int wb2,int wb3) {
    int n = get_global_id(0), m = get_global_id(1), pq = get_global_id(2);
    int p2 = pq % B2, p3 = pq / B2;
    int wp2 = (wb2==1?0:p2), wp3 = (wb3==1?0:p3);
    long wo = (long)n*ws1 + (long)wp2*ws2 + (long)wp3*ws3;
    long xo = (long)m*xs1 + (long)p2*xs2 + (long)p3*xs3;
    float acc = 0.0f;
    for (int k = 0; k < K; k++) acc += w[wo + (long)k*ws0] * x[xo + (long)k*xs0];
    o[(((long)p3*B2 + p2)*M + m)*N + n] = acc;
}
// Tiled (local-memory) F32 matmul: same contract, batch on dim2, strides applied in
// the cooperative load. TSxTS workgroup stages one K-slab of each operand in local
// memory -> each operand read ~TS x fewer times. dim0=m (col), dim1=n (row).
__kernel void k_mul_mat_t(__global const float* w, __global const float* x, __global float* o,
                          int K, int N, int M, int B2,
                          int ws0,int ws1,int ws2,int ws3, int xs0,int xs1,int xs2,int xs3,
                          int wb2,int wb3,
                          __global const float* bias, int hasbias,
                          __global const float* resid, int hasresid, int eact) {
    // TSK=16. We also tried TSK=32 (halving barriers) but it didn't help the attention
    // path — see DEEP_DIVE §13.3 Step 9 for the negative-result analysis.
    __local float As[16][16];   // x slab: As[k_local][m_local]
    __local float Bs[16][16];   // w slab: Bs[n_local][k_local]
    int tx = get_local_id(0), ty = get_local_id(1), pq = get_global_id(2);
    int p2 = pq % B2, p3 = pq / B2;
    int wp2 = (wb2==1?0:p2), wp3 = (wb3==1?0:p3);
    long wbase = (long)wp2*ws2 + (long)wp3*ws3;
    long xbase = (long)p2*xs2 + (long)p3*xs3;
    int m = get_group_id(0)*16 + tx, n = get_group_id(1)*16 + ty;
    float acc = 0.0f;
    for (int k0 = 0; k0 < K; k0 += 16) {
        int ml = get_group_id(0)*16 + tx, kA = k0 + ty;
        As[ty][tx] = (ml < M && kA < K) ? x[xbase + (long)ml*xs1 + (long)kA*xs0] : 0.0f;
        int nl = get_group_id(1)*16 + ty, kB = k0 + tx;
        Bs[ty][tx] = (nl < N && kB < K) ? w[wbase + (long)nl*ws1 + (long)kB*ws0] : 0.0f;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int kk = 0; kk < 16; kk++) acc += Bs[ty][kk] * As[kk][tx];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (n < N && m < M) {
        long idx = (((long)p3*B2 + p2)*M + m)*N + n;
        float v = acc;
        if (hasbias)  v += bias[n];
        v = mg_act(v, eact);
        if (hasresid) v += resid[idx];
        o[idx] = v;
    }
}
// fp16 variant of k_mul_mat_t (same contract; F32 operand loads cast to half in local
// memory; fp16 multiply, fp32 accumulate). Used by the attention path (Q.K^T and A.V),
// which is now the dominant F32 cost. Mali fp16 ALU runs ~2x fp32; the activation
// magnitudes here are small enough that fp16 multiply with fp32 accumulate is precise.
#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void k_mul_mat_t_h(__global const float* w, __global const float* x, __global float* o,
                            int K, int N, int M, int B2,
                            int ws0,int ws1,int ws2,int ws3, int xs0,int xs1,int xs2,int xs3,
                            int wb2,int wb3,
                            __global const float* bias, int hasbias,
                            __global const float* resid, int hasresid, int eact) {
    __local half As[16][16];
    __local half Bs[16][16];
    int tx = get_local_id(0), ty = get_local_id(1), pq = get_global_id(2);
    int p2 = pq % B2, p3 = pq / B2;
    int wp2 = (wb2==1?0:p2), wp3 = (wb3==1?0:p3);
    long wbase = (long)wp2*ws2 + (long)wp3*ws3;
    long xbase = (long)p2*xs2 + (long)p3*xs3;
    int m = get_group_id(0)*16 + tx, n = get_group_id(1)*16 + ty;
    float acc = 0.0f;
    for (int k0 = 0; k0 < K; k0 += 16) {
        int ml = get_group_id(0)*16 + tx, kA = k0 + ty;
        As[ty][tx] = (ml < M && kA < K) ? (half)x[xbase + (long)ml*xs1 + (long)kA*xs0] : (half)0;
        int nl = get_group_id(1)*16 + ty, kB = k0 + tx;
        Bs[ty][tx] = (nl < N && kB < K) ? (half)w[wbase + (long)nl*ws1 + (long)kB*ws0] : (half)0;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int kk = 0; kk < 16; kk++) acc += (float)(Bs[ty][kk] * As[kk][tx]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (n < N && m < M) {
        long idx = (((long)p3*B2 + p2)*M + m)*N + n;
        float v = acc;
        if (hasbias)  v += bias[n];
        v = mg_act(v, eact);
        if (hasresid) v += resid[idx];
        o[idx] = v;
    }
}
#endif
// ggml Q8_0 dequant-fused matmul. w = N rows x (K/32) blocks, each block = 34 bytes
// (fp16 scale + 32 int8). x[K,M] f32 -> out[N,M]. out[n,m]=sum_b d_b*sum_i q[i]*x[m*K+b*32+i].
// One work-item -> one output o[n,m]. Scalar accumulator stays in a register; this
// maximizes parallelism and is the fast path on high-thread desktop GPUs (M1 Max 0.54s).
__kernel void k_mul_mat_q8(__global const uchar* w, __global const float* x, __global float* o,
                           int K, int N, int M) {
    int n = get_global_id(0), m = get_global_id(1);
    if (n >= N || m >= M) return;
    int nb = K / 32; float acc = 0.0f;
    for (int b = 0; b < nb; b++) {
        __global const uchar* blk = w + (long)(n*nb + b) * 34;
        float d = vload_half(0, (__global const half*)blk);
        __global const char* qs = (__global const char*)(blk + 2);
        int kk = b*32;
        for (int i = 0; i < 32; i++) acc += d * (float)qs[i] * x[(long)m*K + kk + i];
    }
    o[(long)m*N + n] = acc;
}
// M register-blocked variant: each work-item computes MR=4 columns for one row n,
// dequantizing each weight block once and reusing it across the 4 x columns -> 1/4 the
// weight-byte traffic. MR is compile-time so acc[MR] register-allocates and loops unroll.
// Best on bandwidth-bound mobile GPUs (Mali-G715 18.7->14.4s); loses on desktop (less
// parallelism), so the host selects this only for Mali/Adreno.
#define MRB 4
__kernel void k_mul_mat_q8_b4(__global const uchar* w, __global const float* x, __global float* o,
                              int K, int N, int M) {
    int n = get_global_id(0), m0 = get_global_id(1) * MRB;
    if (n >= N || m0 >= M) return;
    int mr = min(MRB, M - m0), nb = K / 32;
    float acc[MRB]; for (int r = 0; r < MRB; r++) acc[r] = 0.0f;
    for (int b = 0; b < nb; b++) {
        __global const uchar* blk = w + (long)(n*nb + b) * 34;
        float d = vload_half(0, (__global const half*)blk);
        __global const char* qs = (__global const char*)(blk + 2);
        int kk = b*32;
        for (int i = 0; i < 32; i++) {
            float wv = d * (float)qs[i];
            for (int r = 0; r < mr; r++) acc[r] += wv * x[(long)(m0+r)*K + kk + i];
        }
    }
    for (int r = 0; r < mr; r++) o[(long)(m0+r)*N + n] = acc[r];
}
// ggml Q4_K get_scale_min_k4: unpack 6-bit (scale,min) for sub-block j from 12 bytes.
inline uchar2 q4k_sm(__global const uchar* sc, int j) {
    uchar d, m;
    if (j < 4) { d = sc[j] & 63; m = sc[j+4] & 63; }
    else { d = (sc[j+4] & 0xF) | ((sc[j-4] >> 6) << 4);
           m = (sc[j+4] >>  4) | ((sc[j]   >> 6) << 4); }
    return (uchar2)(d, m);
}
// ggml Q4_K dequant-fused matmul. w = N rows x (K/256) super-blocks (144 B each):
// fp16 d, fp16 dmin, 12 B packed scales/mins, 128 B 4-bit quants. y = d*sc*q - dmin*m.
__kernel void k_mul_mat_q4k(__global const uchar* w, __global const float* x, __global float* o,
                            int K, int N, int M) {
    int n = get_global_id(0), m = get_global_id(1);
    if (n>=N || m>=M) return;
    int nsb = K / 256; float acc = 0.0f;
    for (int sb = 0; sb < nsb; sb++) {
        __global const uchar* blk = w + (long)(n*nsb + sb) * 144;
        float d    = vload_half(0, (__global const half*)(blk));
        float dmin = vload_half(0, (__global const half*)(blk + 2));
        __global const uchar* sc = blk + 4;
        __global const uchar* qs = blk + 16;
        int xs = sb*256;
        for (int g = 0; g < 4; g++) {
            uchar2 a = q4k_sm(sc, 2*g), b = q4k_sm(sc, 2*g+1);
            float da=d*a.x, mna=dmin*a.y, db=d*b.x, mnb=dmin*b.y;
            __global const uchar* qg = qs + g*32;
            for (int l = 0; l < 32; l++) {
                uchar qb = qg[l];
                float wa = da*(qb & 0xF) - mna, wb = db*(qb >> 4) - mnb;
                int ka = xs + (2*g)*32 + l, kb = xs + (2*g+1)*32 + l;
                acc += wa * x[(long)m*K + ka] + wb * x[(long)m*K + kb];
            }
        }
    }
    o[(long)m*N + n] = acc;
}
// M register-blocked Q4_K (MR=4), mobile path — see k_mul_mat_q8_b4 rationale.
__kernel void k_mul_mat_q4k_b4(__global const uchar* w, __global const float* x, __global float* o,
                               int K, int N, int M) {
    int n = get_global_id(0), m0 = get_global_id(1) * MRB;
    if (n>=N || m0>=M) return;
    int mr = min(MRB, M - m0), nsb = K / 256;
    float acc[MRB]; for (int r = 0; r < MRB; r++) acc[r] = 0.0f;
    for (int sb = 0; sb < nsb; sb++) {
        __global const uchar* blk = w + (long)(n*nsb + sb) * 144;
        float d    = vload_half(0, (__global const half*)(blk));
        float dmin = vload_half(0, (__global const half*)(blk + 2));
        __global const uchar* sc = blk + 4;
        __global const uchar* qs = blk + 16;
        int xs = sb*256;
        for (int g = 0; g < 4; g++) {
            uchar2 a = q4k_sm(sc, 2*g), b = q4k_sm(sc, 2*g+1);
            float da=d*a.x, mna=dmin*a.y, db=d*b.x, mnb=dmin*b.y;
            __global const uchar* qg = qs + g*32;
            for (int l = 0; l < 32; l++) {
                uchar qb = qg[l];
                float wa = da*(qb & 0xF) - mna, wb = db*(qb >> 4) - mnb;
                int ka = xs + (2*g)*32 + l, kb = xs + (2*g+1)*32 + l;
                for (int r = 0; r < mr; r++)
                    acc[r] += wa * x[(long)(m0+r)*K + ka] + wb * x[(long)(m0+r)*K + kb];
            }
        }
    }
    for (int r = 0; r < mr; r++) o[(long)(m0+r)*N + n] = acc[r];
}

// ---- Tiled (local-memory) quantized matmul -------------------------------------
// The scalar/b4 kernels are memory-bound: each weight row is re-read M times and each
// activation column N times (~2*N*K*M global reads). These stage TSxTS tiles of both
// operands in local memory, so each is read ~TS x fewer times -- the standard GEMM
// blocking. One work-item -> one output o[n,m]; a workgroup of TS x TS cooperatively
// loads (and dequantizes) one K-slab of each operand per step. Bounds-checked so the
// non-multiple-of-TS dims (M=257) are safe.
#define TS 16
// single-element Q4_K dequant: weight (n,k) from N-row x (K/256)-superblock storage.
inline float deq_q4k_elem(__global const uchar* w, int n, int k, int nsb) {
    int sb = k >> 8, r = k & 255;            // superblock, offset within (0..255)
    __global const uchar* blk = w + (long)(n*nsb + sb) * 144;
    float d    = vload_half(0, (__global const half*)(blk));
    float dmin = vload_half(0, (__global const half*)(blk + 2));
    int sub = r >> 5, j = r & 31;            // sub-block (0..7), index within (0..31)
    uchar2 sm = q4k_sm(blk + 4, sub);
    uchar qb = (blk + 16)[(sub >> 1) * 32 + j];
    float q = (sub & 1) ? (qb >> 4) : (qb & 0xF);
    return d * sm.x * q - dmin * sm.y;
}
__kernel void k_mul_mat_q8_t(__global const uchar* w, __global const float* x, __global float* o,
                             int K, int N, int M) {
    __local float As[TS][TS];   // x slab:  As[k_local][m_local]
    __local float Bs[TS][TS];   // w slab:  Bs[n_local][k_local] (dequantized)
    int tx = get_local_id(0), ty = get_local_id(1);
    int m = get_group_id(0)*TS + tx;        // activation column
    int n = get_group_id(1)*TS + ty;        // weight row
    int nb = K / 32;
    float acc = 0.0f;
    for (int k0 = 0; k0 < K; k0 += TS) {
        int ml = get_group_id(0)*TS + tx, kA = k0 + ty;     // As[ty][tx] = x[ml, kA]
        As[ty][tx] = (ml < M && kA < K) ? x[(long)ml*K + kA] : 0.0f;
        int nl = get_group_id(1)*TS + ty, kB = k0 + tx;     // Bs[ty][tx] = dequant w[nl, kB]
        float wv = 0.0f;
        if (nl < N && kB < K) {
            __global const uchar* blk = w + (long)(nl*nb + (kB>>5)) * 34;
            wv = vload_half(0, (__global const half*)blk) * (float)((__global const char*)(blk+2))[kB & 31];
        }
        Bs[ty][tx] = wv;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int kk = 0; kk < TS; kk++) acc += Bs[ty][kk] * As[kk][tx];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (n < N && m < M) o[(long)m*N + n] = acc;
}
__kernel void k_mul_mat_q4k_t(__global const uchar* w, __global const float* x, __global float* o,
                              int K, int N, int M) {
    __local float As[TS][TS];
    __local float Bs[TS][TS];
    int tx = get_local_id(0), ty = get_local_id(1);
    int m = get_group_id(0)*TS + tx;
    int n = get_group_id(1)*TS + ty;
    int nsb = K / 256;
    float acc = 0.0f;
    for (int k0 = 0; k0 < K; k0 += TS) {
        int ml = get_group_id(0)*TS + tx, kA = k0 + ty;
        As[ty][tx] = (ml < M && kA < K) ? x[(long)ml*K + kA] : 0.0f;
        int nl = get_group_id(1)*TS + ty, kB = k0 + tx;
        Bs[ty][tx] = (nl < N && kB < K) ? deq_q4k_elem(w, nl, kB, nsb) : 0.0f;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int kk = 0; kk < TS; kk++) acc += Bs[ty][kk] * As[kk][tx];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (n < N && m < M) o[(long)m*N + n] = acc;
}

// ---- 2D register-tiled quantized matmul ----------------------------------------
// The _t kernels above compute 1 output/work-item: the inner loop does 1 FMA per 2
// local loads (low arithmetic intensity). These compute a WPTM x WPTN (4x4) micro-tile
// per work-item: a step loads WPTM A-values + WPTN B-values into registers and does
// WPTM*WPTN=16 FMAs -> 8 loads per 16 FMA, ~4x the intensity. GEMM terms: A=x[M,K]
// (row m), B[k,n]=dequant w[n,k], C[m,n]=o[m*N+n]. A TSMxTSN (64x64) output tile per
// workgroup of RTSM x RTSN = 16x16 work-items; one TSK(16)-deep K-slab staged in local
// memory per step (weights dequantized once on load). (myGEMM-style register blocking.)
// Tile config: WPTM x WPTN work-per-thread micro-tile, fixed 16x16 (=RTS^2) workgroup.
// Injected at clBuildProgram time (-DGEMM_WPTM=.. -DGEMM_WPTN=..) so the host dispatch
// and the kernel share one source of truth; defaults here keep standalone builds valid.
#ifndef GEMM_WPTM
#define GEMM_WPTM 4
#endif
#ifndef GEMM_WPTN
#define GEMM_WPTN 4
#endif
#define RTS  16                 // work-items per tile dim (workgroup = RTS*RTS = 256)
#define WPTM GEMM_WPTM
#define WPTN GEMM_WPTN
#define TSM  (RTS*WPTM)         // output-tile rows (pixels/m)
#define TSN  (RTS*WPTN)         // output-tile cols (n)
#define TSK  16                 // K-slab depth
#define RTSM RTS
#define RTSN RTS
#define LPTA WPTM               // (TSK*TSM)/(RTS*RTS) = WPTM
#define LPTB WPTN               // (TSK*TSN)/(RTS*RTS) = WPTN
// LT = local-slab element type. float = exact; half (with cl_khr_fp16) halves local
// storage and runs the micro-tile multiply on Mali's 2x-rate fp16 ALU, accumulating in
// float so the K-length sum stays accurate.
#define K_GEMM2D_BODY(LT, DEQ_B)                                                      \
    int tidm = get_local_id(0), tidn = get_local_id(1);                               \
    int offM = TSM * get_group_id(0), offN = TSN * get_group_id(1);                   \
    int tid  = tidn * RTSM + tidm;                                                    \
    __local LT Asub[TSK][TSM];                                                        \
    __local LT Bsub[TSK][TSN];                                                        \
    float acc[WPTM][WPTN];                                                            \
    for (int a=0;a<WPTM;a++) for (int b=0;b<WPTN;b++) acc[a][b] = 0.0f;               \
    int nt = (K + TSK - 1) / TSK;                                                      \
    for (int t = 0; t < nt; t++) {                                                    \
        for (int la = 0; la < LPTA; la++) {                                           \
            int id = la*RTSM*RTSN + tid, row = id % TSM, col = id / TSM;              \
            int gm = offM + row, gk = t*TSK + col;                                    \
            Asub[col][row] = (gm < M && gk < K) ? (LT)x[(long)gm*K + gk] : (LT)0;     \
        }                                                                             \
        for (int lb = 0; lb < LPTB; lb++) {                                           \
            int id = lb*RTSM*RTSN + tid, row = id % TSN, col = id / TSN;              \
            int gn = offN + row, gk = t*TSK + col;                                    \
            Bsub[col][row] = (gn < N && gk < K) ? (LT)(DEQ_B) : (LT)0;                \
        }                                                                             \
        barrier(CLK_LOCAL_MEM_FENCE);                                                 \
        for (int k = 0; k < TSK; k++) {                                               \
            LT areg[WPTM], breg[WPTN];                                                \
            for (int wm = 0; wm < WPTM; wm++) areg[wm] = Asub[k][tidm + wm*RTSM];     \
            for (int wn = 0; wn < WPTN; wn++) breg[wn] = Bsub[k][tidn + wn*RTSN];     \
            for (int wm = 0; wm < WPTM; wm++)                                         \
                for (int wn = 0; wn < WPTN; wn++)                                     \
                    acc[wm][wn] += (float)(areg[wm]*breg[wn]);                        \
        }                                                                             \
        barrier(CLK_LOCAL_MEM_FENCE);                                                 \
    }                                                                                 \
    for (int wm = 0; wm < WPTM; wm++) {                                               \
        int gm = offM + tidm + wm*RTSM;                                               \
        for (int wn = 0; wn < WPTN; wn++) {                                           \
            int gn = offN + tidn + wn*RTSN;                                           \
            if (gm < M && gn < N) {                                                   \
                float v = acc[wm][wn];                                                \
                if (hasbias)  v += bias[gn];                                          \
                v = mg_act(v, eact);                                                  \
                if (hasresid) v += resid[(long)gm*N + gn];                            \
                o[(long)gm*N + gn] = v;                                               \
            }                                                                         \
        }                                                                             \
    }
// fused epilogue params shared by the *_t2 kernels (out = act(w.x + bias) + resid)
#define EPI __global const float* bias, int hasbias, __global const float* resid, int hasresid, int eact
// dequant one Q8_0 weight (gn,gk): block gk/32 of row gn -> fp16 scale * int8.
#define DEQ_Q8 (vload_half(0,(__global const half*)(w+(long)(gn*nb+(gk>>5))*34)) * \
                (float)((__global const char*)(w+(long)(gn*nb+(gk>>5))*34+2))[gk&31])
__kernel void k_mul_mat_q8_t2(__global const uchar* w, __global const float* x, __global float* o,
                              int K, int N, int M, EPI) {
    int nb = K / 32;
    K_GEMM2D_BODY(float, DEQ_Q8)
}
__kernel void k_mul_mat_q4k_t2(__global const uchar* w, __global const float* x, __global float* o,
                               int K, int N, int M, EPI) {
    int nsb = K / 256;
    K_GEMM2D_BODY(float, deq_q4k_elem(w, gn, gk, nsb))
}
// fp16 variants (Mali): half local slabs + fp16 micro-tile multiply, fp32 accumulate.
// Guarded so the program still builds on devices without cl_khr_fp16 (e.g. M1).
#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void k_mul_mat_q8_t2_h(__global const uchar* w, __global const float* x, __global float* o,
                                int K, int N, int M, EPI) {
    int nb = K / 32;
    K_GEMM2D_BODY(half, DEQ_Q8)
}
__kernel void k_mul_mat_q4k_t2_h(__global const uchar* w, __global const float* x, __global float* o,
                                 int K, int N, int M, EPI) {
    int nsb = K / 256;
    K_GEMM2D_BODY(half, deq_q4k_elem(w, gn, gk, nsb))
}
#endif

// ---- int8 matmul via ARM dot-product (the GPU analog of the CPU's i8mm) -----------
// Mali exposes arm_dot_acc(char4,char4,int) = acc + sum a[i]*b[i] in one op. Instead of
// dequantizing Q8_0 weights to float, we quantize the ACTIVATION to Q8_0 too and do an
// int8 x int8 -> int32 dot per 32-block, then scale by (d_w * d_x) -- exactly ggml's
// Q8_0xQ8_0 vec_dot. Adds int8 activation-quantization error (cosine ~0.999) but uses
// the native int8 datapath. Guarded: only compiles where the extension exists.
#ifdef cl_arm_integer_dot_product_accumulate_int8
#pragma OPENCL EXTENSION cl_arm_integer_dot_product_accumulate_int8 : enable
// quantize activation x[K,M] (f32, contiguous x[k+K*m]) -> Q8_0: qx int8[K,M] + dx[M*nb]
// (symmetric amax/127 per column m, 32-block along K; matches ggml quantize_row_q8_0).
__kernel void k_quantize_q8(__global const float* x, __global char* qx, __global float* dx,
                            int K, int M) {
    int m = get_global_id(0), b = get_global_id(1), nb = K/32;
    if (m >= M || b >= nb) return;
    long base = (long)m*K + b*32;
    float amax = 0.0f;
    for (int i = 0; i < 32; i++) amax = fmax(amax, fabs(x[base+i]));
    float d = amax/127.0f, id = d > 0.0f ? 1.0f/d : 0.0f;
    dx[(long)m*nb + b] = d;
    for (int i = 0; i < 32; i++) qx[base+i] = (char)round(x[base+i]*id);
}
// tiled int8 GEMM: w = Q8_0 (34B blocks: fp16 d + 32 int8), qx/dx = quantized activation.
// 64x64 output tile / 16x16 workgroup / 4x4 micro-tile, one 32-block K-slab per step.
#define I8_RTS 16
#define I8_TSM 64
#define I8_TSN 64
__kernel void k_mul_mat_q8_i8(__global const uchar* w, __global const char* qx,
                              __global const float* dx, __global float* o,
                              int K, int N, int M, EPI) {
    __local char  As[I8_TSM][32];   // activation int8 slab (m-tile x 32)
    __local char  Bs[I8_TSN][32];   // weight     int8 slab (n-tile x 32)
    __local float dxl[I8_TSM], dwl[I8_TSN];   // per-block scales for the tile
    int tidm = get_local_id(0), tidn = get_local_id(1), tid = tidn*I8_RTS + tidm;
    int offM = I8_TSM*get_group_id(0), offN = I8_TSN*get_group_id(1), nb = K/32;
    float acc[4][4]; for (int a=0;a<4;a++) for (int b=0;b<4;b++) acc[a][b]=0.0f;
    for (int t = 0; t < nb; t++) {
        for (int l = tid; l < I8_TSN*32; l += I8_RTS*I8_RTS) {   // weight int8 slab
            int nL = l>>5, kk = l&31, gn = offN+nL;
            Bs[nL][kk] = (gn<N) ? ((__global const char*)(w+(long)(gn*nb+t)*34+2))[kk] : (char)0;
        }
        for (int l = tid; l < I8_TSM*32; l += I8_RTS*I8_RTS) {   // activation int8 slab
            int mL = l>>5, kk = l&31, gm = offM+mL;
            As[mL][kk] = (gm<M) ? qx[(long)gm*K + t*32 + kk] : (char)0;
        }
        for (int l = tid; l < I8_TSN; l += I8_RTS*I8_RTS) {      // weight scales
            int gn = offN+l;
            dwl[l] = (gn<N) ? vload_half(0,(__global const half*)(w+(long)(gn*nb+t)*34)) : 0.0f;
        }
        for (int l = tid; l < I8_TSM; l += I8_RTS*I8_RTS) {      // activation scales
            int gm = offM+l;
            dxl[l] = (gm<M) ? dx[(long)gm*nb + t] : 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int wm = 0; wm < 4; wm++) {
            int mL = tidm + wm*I8_RTS; float dxm = dxl[mL];
            char4 av[8];                              // hoist activation slab into regs:
            for (int i = 0; i < 8; i++) av[i] = vload4(i, As[mL]);   // loaded once, reused over wn
            for (int wn = 0; wn < 4; wn++) {
                int nL = tidn + wn*I8_RTS;
                int idot = 0;
                for (int i = 0; i < 8; i++)
                    idot = arm_dot_acc(av[i], vload4(i, Bs[nL]), idot);
                acc[wm][wn] += dwl[nL]*dxm*(float)idot;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int wm = 0; wm < 4; wm++) {
        int gm = offM + tidm + wm*I8_RTS;
        for (int wn = 0; wn < 4; wn++) {
            int gn = offN + tidn + wn*I8_RTS;
            if (gm < M && gn < N) {
                float v = acc[wm][wn];
                if (hasbias)  v += bias[gn];
                v = mg_act(v, eact);
                if (hasresid) v += resid[(long)gm*N + gn];
                o[(long)gm*N + gn] = v;
            }
        }
    }
}

// ---- int8 VQGAN conv (arm_dot) -----------------------------------------------------
// Same idea as the FC int8 path, applied to the implicit-GEMM conv. Pre-quantize conv
// weights to int8 per (oc, 32-block along k=ic*KH*KW); the conv kernel gathers the F32
// im2col column into local memory and quantizes each pixel's 32-element K-block with its
// OWN scale (per-block, gather-time) -- matching the FC's per-32-block activation quant,
// which keeps cosine ~1.0 (a per-tensor scale clipped the wide-magnitude blocks).
// quantize conv weights ker[oc*K + k] (k contiguous per oc) -> int8 + per-block scales.
__kernel void k_quantize_conv_w(__global const float* ker, __global char* qker,
                                __global float* dker, int K, int OC) {
    int oc = get_global_id(0), b = get_global_id(1), nb = (K+31)/32;
    if (oc >= OC || b >= nb) return;
    int k0 = b*32, n = min(32, K-k0); long base = (long)oc*K + k0;
    float amax = 0.0f;
    for (int i=0;i<n;i++) amax = fmax(amax, fabs(ker[base+i]));
    float d = amax/127.0f, id = d>0.0f?1.0f/d:0.0f;
    dker[(long)oc*nb + b] = d;
    for (int i=0;i<32;i++) qker[base+i] = (i<n) ? (char)round(ker[base+i]*id) : (char)0;
}
// tiled int8 conv: 64x64 output (pixel x oc) / 16x16 wg / 4x4 micro, 32-block K-slab.
// reads the F32 input directly (no pre-quantization pass); quantizes the gathered column
// per (pixel, 32-block) on the fly.
__kernel void k_conv2d_i8(__global const float* in,
                          __global const char* qker, __global const float* dker,
                          __global float* o,
                          int IW,int IH,int IC,int OW,int OH,int OC,int KW,int KH,int stride,int pad) {
    int M=OW*OH, N=OC, K=IC*KW*KH, nb=(K+31)/32, nimg=get_global_id(2);
    __local float Af[64][32];   // gathered im2col column, F32
    __local char  As[64][32];   // ... quantized per (pixel,block)
    __local char  Bs[64][32];   // weights, int8
    __local float dcl[64], dwl[64];   // per-pixel activation scale, per-oc weight scale
    int tidm=get_local_id(0), tidn=get_local_id(1), tid=tidn*16+tidm;
    int offM=64*get_group_id(0), offN=64*get_group_id(1);
    float acc[4][4]; for(int a=0;a<4;a++)for(int b=0;b<4;b++)acc[a][b]=0.0f;
    for (int t=0;t<nb;t++){
        for (int l=tid; l<64*32; l+=256){          // gather col F32
            int mL=l>>5, kk=l&31, p=offM+mL, gk=t*32+kk; float v=0.0f;
            if (p<M && gk<K){
                int ow=p%OW, oh=p/OW, kw=gk%KW, tmp=gk/KW, kh=tmp%KH, ic=tmp/KH;
                int iw=ow*stride-pad+kw, ih=oh*stride-pad+kh;
                if (iw>=0&&iw<IW&&ih>=0&&ih<IH) v=in[(long)iw + IW*((long)ih + IH*((long)ic + IC*nimg))];
            }
            Af[mL][kk]=v;
        }
        for (int l=tid; l<64*32; l+=256){          // weights int8 (qker[oc*K+gk])
            int nL=l>>5, kk=l&31, oc=offN+nL, gk=t*32+kk;
            Bs[nL][kk] = (oc<N && gk<K) ? qker[(long)oc*K + gk] : (char)0;
        }
        for (int l=tid; l<64; l+=256){ int oc=offN+l; dwl[l]=(oc<N)?dker[(long)oc*nb+t]:0.0f; }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid<64){                                // per-pixel per-block quantize the col
            float amax=0.0f; for(int kk=0;kk<32;kk++) amax=fmax(amax,fabs(Af[tid][kk]));
            float d=amax/127.0f, id=d>0.0f?1.0f/d:0.0f; dcl[tid]=d;
            for(int kk=0;kk<32;kk++) As[tid][kk]=(char)round(Af[tid][kk]*id);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int wm=0;wm<4;wm++){
            char4 av[8]; int mL=tidm+wm*16; for(int i=0;i<8;i++) av[i]=vload4(i,As[mL]);
            float dcm=dcl[mL];
            for (int wn=0;wn<4;wn++){
                int nL=tidn+wn*16, idot=0;
                for(int i=0;i<8;i++) idot=arm_dot_acc(av[i], vload4(i,Bs[nL]), idot);
                acc[wm][wn]+=dcm*dwl[nL]*(float)idot;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int wm=0;wm<4;wm++){
        int p=offM+tidm+wm*16, ow=p%OW, oh=p/OW;
        for (int wn=0;wn<4;wn++){
            int oc=offN+tidn+wn*16;
            if (p<M && oc<N) o[(long)ow + OW*((long)oh + OH*((long)oc + OC*nimg))]=acc[wm][wn];
        }
    }
}
#endif
// gather rows of a[E,R] by I32 ids -> out[E,n]; negative id wraps (mask token).
__kernel void k_get_rows(__global const float* a, __global const int* ids, __global float* o,
                         int E, int R) {
    int e = get_global_id(0), i = get_global_id(1);
    int r = ids[i]; if (r < 0) r += R;
    o[(long)i*E + e] = a[(long)r*E + e];
}
// LayerNorm over ne0 (no affine).
// LayerNorm (no affine) over ne[0]=D. Workgroup-parallel: one workgroup of NORM_TG
// threads per row, with local-memory tree reduction for mean and variance. The original
// kernel was one work-item per row reading D=768 elements 3 times sequentially.
#define NORM_TG 64
__kernel __attribute__((reqd_work_group_size(NORM_TG,1,1)))
void k_norm(__global const float* a, __global float* o, int D, float eps) {
    int tid = get_local_id(0); int row = get_group_id(0);
    __global const float* r = a + (long)row*D;
    __global float* w = o + (long)row*D;
    __local float lp[NORM_TG];
    float s = 0.0f;
    for (int d = tid; d < D; d += NORM_TG) s += r[d];
    lp[tid] = s; barrier(CLK_LOCAL_MEM_FENCE);
    for (int k = NORM_TG>>1; k > 0; k >>= 1) {
        if (tid < k) lp[tid] += lp[tid + k]; barrier(CLK_LOCAL_MEM_FENCE);
    }
    float mean = lp[0] / D;
    float v = 0.0f;
    for (int d = tid; d < D; d += NORM_TG) { float t = r[d] - mean; v += t*t; }
    lp[tid] = v; barrier(CLK_LOCAL_MEM_FENCE);
    for (int k = NORM_TG>>1; k > 0; k >>= 1) {
        if (tid < k) lp[tid] += lp[tid + k]; barrier(CLK_LOCAL_MEM_FENCE);
    }
    float inv = rsqrt(lp[0] / D + eps);
    for (int d = tid; d < D; d += NORM_TG) w[d] = (r[d] - mean) * inv;
}
// Softmax over ne[0]=D with scale. Workgroup-parallel: max-reduce then exp+sum-reduce
// (numerically stable via -mx). One workgroup of SM_TG threads per row.
#define SM_TG 64
__kernel __attribute__((reqd_work_group_size(SM_TG,1,1)))
void k_soft_max(__global const float* a, __global float* o, int D, float scale) {
    int tid = get_local_id(0); int row = get_group_id(0);
    __global const float* r = a + (long)row*D;
    __global float* w = o + (long)row*D;
    __local float lp[SM_TG];
    float m = -INFINITY;
    for (int d = tid; d < D; d += SM_TG) m = fmax(m, r[d] * scale);
    lp[tid] = m; barrier(CLK_LOCAL_MEM_FENCE);
    for (int k = SM_TG>>1; k > 0; k >>= 1) {
        if (tid < k) lp[tid] = fmax(lp[tid], lp[tid + k]); barrier(CLK_LOCAL_MEM_FENCE);
    }
    float mx = lp[0];
    float s = 0.0f;
    for (int d = tid; d < D; d += SM_TG) { float e = exp(r[d]*scale - mx); w[d] = e; s += e; }
    lp[tid] = s; barrier(CLK_LOCAL_MEM_FENCE);
    for (int k = SM_TG>>1; k > 0; k >>= 1) {
        if (tid < k) lp[tid] += lp[tid + k]; barrier(CLK_LOCAL_MEM_FENCE);
    }
    float inv = 1.0f / lp[0];
    for (int d = tid; d < D; d += SM_TG) w[d] *= inv;
}
// Flash-attention v2 (forward). Replaces the QK·softmax·V chain with a single fused
// kernel that never materializes the M×M scores tensor — saves O(S²·H·B·4B) of DRAM
// traffic per layer compared with the naive path. Inputs Q, K, V are contiguous
// {D, S, H, B} (head dim innermost). One workgroup per (Q tile of BR rows, head,
// batch); each thread handles one Q row, holds Q row + O row + running (m, l) in
// registers, and tiles K/V into local memory across BC-wide chunks.
//
// Compile-time tiling assumes MaskGIT head_dim = 48. BR = BC = 32 → local mem =
// 2 × BC × D × 4 = 12 KB per workgroup, well under typical 32-64 KB Mali limit.
#define FA_D  48
#define FA_BR 32
#define FA_BC 32
__kernel __attribute__((reqd_work_group_size(FA_BR, 1, 1)))
void k_flash_attention(__global const float* Q, __global const float* K, __global const float* V,
                       __global float* O, int S, float scale) {
    int tid = get_local_id(0);
    int qt  = get_group_id(0);   // Q tile index 0..ceil(S/BR)-1
    int h   = get_group_id(1);   // head
    int b   = get_group_id(2);   // batch
    int H   = get_num_groups(1);

    int row = qt * FA_BR + tid;
    int active = (row < S) ? 1 : 0;
    long base = (long)((b * H + h) * S) * FA_D;

    // Q row into registers. Inactive lanes load zeros (we don't use the value).
    float q_row[FA_D];
    if (active) {
        for (int d = 0; d < FA_D; d++) q_row[d] = Q[base + (long)row * FA_D + d];
    }

    float m_i = -INFINITY;
    float l_i = 0.f;
    float o_row[FA_D];
    for (int d = 0; d < FA_D; d++) o_row[d] = 0.f;

    __local float K_tile[FA_BC * FA_D];
    __local float V_tile[FA_BC * FA_D];

    int n_kt = (S + FA_BC - 1) / FA_BC;
    for (int kt = 0; kt < n_kt; kt++) {
        // Collaboratively load BC × D K + V values. With FA_BR=FA_BC=32 and FA_D=48,
        // total = 1536; each thread loads 48 = FA_D contiguous elements.
        int k_base = kt * FA_BC;
        for (int idx = tid; idx < FA_BC * FA_D; idx += FA_BR) {
            int kc = idx / FA_D;
            int dd = idx % FA_D;
            int k_row = k_base + kc;
            int in_range = (k_row < S) ? 1 : 0;
            K_tile[idx] = in_range ? K[base + (long)k_row * FA_D + dd] : 0.f;
            V_tile[idx] = in_range ? V[base + (long)k_row * FA_D + dd] : 0.f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (active) {
            // BC scores for this Q row against the loaded K tile.
            float s_kc[FA_BC];
            float m_new = m_i;
            for (int kc = 0; kc < FA_BC; kc++) {
                int k_row = k_base + kc;
                if (k_row >= S) { s_kc[kc] = -INFINITY; continue; }
                float dot = 0.f;
                for (int d = 0; d < FA_D; d++) dot += q_row[d] * K_tile[kc * FA_D + d];
                dot *= scale;
                s_kc[kc] = dot;
                m_new = fmax(m_new, dot);
            }
            // Online softmax rescale + P = exp(S - m_new); l_i carries the running sum.
            float alpha = exp(m_i - m_new);
            float l_new = alpha * l_i;
            for (int kc = 0; kc < FA_BC; kc++) {
                float p = (s_kc[kc] == -INFINITY) ? 0.f : exp(s_kc[kc] - m_new);
                s_kc[kc] = p;
                l_new += p;
            }
            // O_new = alpha · O_old + P · V_tile  (rescale old accumulator, accumulate new tile)
            for (int d = 0; d < FA_D; d++) {
                float new_o = alpha * o_row[d];
                for (int kc = 0; kc < FA_BC; kc++) new_o += s_kc[kc] * V_tile[kc * FA_D + d];
                o_row[d] = new_o;
            }
            m_i = m_new;
            l_i = l_new;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (active) {
        float inv = 1.f / l_i;
        for (int d = 0; d < FA_D; d++) O[base + (long)row * FA_D + d] = o_row[d] * inv;
    }
}
// materialize a (possibly strided) view into a contiguous buffer. s = src element strides.
__kernel void k_cont(__global const float* a, __global float* o,
                     int n0,int n1,int n2,int n3, int s0,int s1,int s2,int s3) {
    int i = get_global_id(0); int tot = n0*n1*n2*n3; if (i >= tot) return;
    int i0=i%n0, t=i/n0; int i1=t%n1; t/=n1; int i2=t%n2; t/=n2; int i3=t;
    o[i] = a[(long)i0*s0 + (long)i1*s1 + (long)i2*s2 + (long)i3*s3];
}
// elementwise a OP b, b broadcast over dims whose size is 1 (a contiguous).
__kernel void k_add(__global const float* a, __global const float* b, __global float* o,
                    int n0,int n1,int n2,int n3, int b0,int b1,int b2,int b3) {
    int i = get_global_id(0); int tot = n0*n1*n2*n3; if (i >= tot) return;
    int i0=i%n0, t=i/n0; int i1=t%n1; t/=n1; int i2=t%n2; t/=n2; int i3=t;
    int j0=(b0==1?0:i0), j1=(b1==1?0:i1), j2=(b2==1?0:i2), j3=(b3==1?0:i3);
    o[i] = a[i] + b[j0 + b0*(j1 + b1*(j2 + b2*j3))];
}
__kernel void k_mul(__global const float* a, __global const float* b, __global float* o,
                    int n0,int n1,int n2,int n3, int b0,int b1,int b2,int b3) {
    int i = get_global_id(0); int tot = n0*n1*n2*n3; if (i >= tot) return;
    int i0=i%n0, t=i/n0; int i1=t%n1; t/=n1; int i2=t%n2; t/=n2; int i3=t;
    int j0=(b0==1?0:i0), j1=(b1==1?0:i1), j2=(b2==1?0:i2), j3=(b3==1?0:i3);
    o[i] = a[i] * b[j0 + b0*(j1 + b1*(j2 + b2*j3))];
}
__kernel void k_scale(__global const float* a, __global float* o, const float s) {
    int i = get_global_id(0); o[i] = a[i] * s;
}
__kernel void k_gelu(__global const float* a, __global float* o) {   // exact erf GELU
    int i = get_global_id(0); float x = a[i];
    o[i] = 0.5f * x * (1.0f + erf(x * 0.70710678118654752440f));
}
__kernel void k_silu(__global const float* a, __global float* o) {   // swish
    int i = get_global_id(0); float x = a[i];
    o[i] = x / (1.0f + exp(-x));
}
// direct conv2d, mg layout: in{IW,IH,IC,N}, ker{KW,KH,IC,OC} -> out{OW,OH,OC,N}.
__kernel void k_conv2d(__global const float* in, __global const float* ker, __global float* o,
                       int IW,int IH,int IC,int OW,int OH,int OC,int KW,int KH,int stride,int pad) {
    int ow=get_global_id(0), oh=get_global_id(1), ocn=get_global_id(2);
    int oc=ocn%OC, n=ocn/OC; if (ow>=OW||oh>=OH) return;
    float acc=0.0f;
    for (int ic=0; ic<IC; ic++)
      for (int kh=0; kh<KH; kh++) { int ih=oh*stride-pad+kh; if (ih<0||ih>=IH) continue;
        for (int kw=0; kw<KW; kw++) { int iw=ow*stride-pad+kw; if (iw<0||iw>=IW) continue;
          acc += in[(long)iw + IW*((long)ih + IH*((long)ic + IC*(long)n))]
               * ker[(long)kw + KW*((long)kh + KH*((long)ic + IC*(long)oc))]; } }
    o[(long)ow + OW*((long)oh + OH*((long)oc + OC*(long)n))] = acc;
}
// Tiled conv2d as an implicit-GEMM: out[oc,p] = sum_k ker[oc,k] * col[k,p], where
// p=(ow,oh) is an output pixel and k=(ic,kh,kw). It's the tiled local-memory GEMM
// (M=OW*OH pixels, N=OC, K=IC*KH*KW) with the im2col column GATHERED ON THE FLY into
// local memory (no materialized im2col buffer -- that would be ~300 MB at 256x256).
// A CTSxCTS workgroup loads one CTS-deep K-slab of each operand per step, so the
// kernel weights are read ~CTS x fewer times than the naive one-thread-per-pixel conv.
#define CTS 16
__kernel void k_conv2d_t(__global const float* in, __global const float* ker, __global float* o,
                         int IW,int IH,int IC,int OW,int OH,int OC,int KW,int KH,int stride,int pad) {
    int M = OW*OH, N = OC, K = IC*KH*KW, nimg = get_global_id(2);
    __local float As[CTS][CTS];   // As[k_local][p_local] = col (gathered)
    __local float Bs[CTS][CTS];   // Bs[oc_local][k_local] = weights
    int tx = get_local_id(0), ty = get_local_id(1);
    int p  = get_group_id(0)*CTS + tx;    // output pixel (row-major ow + OW*oh)
    int oc = get_group_id(1)*CTS + ty;    // output channel
    int ow = (p < M) ? p % OW : 0, oh = (p < M) ? p / OW : 0;
    float acc = 0.0f;
    int nt = (K + CTS - 1) / CTS;
    for (int t = 0; t < nt; t++) {
        int kA = t*CTS + ty;              // As[ty][tx] = col[k=kA, pixel=p]
        float av = 0.0f;
        if (p < M && kA < K) {
            int kw = kA % KW, tmp = kA / KW, kh = tmp % KH, ic = tmp / KH;
            int iw = ow*stride - pad + kw, ih = oh*stride - pad + kh;
            if (iw >= 0 && iw < IW && ih >= 0 && ih < IH)
                av = in[(long)iw + IW*((long)ih + IH*((long)ic + IC*(long)nimg))];
        }
        As[ty][tx] = av;
        int kB = t*CTS + tx, ocB = get_group_id(1)*CTS + ty;   // Bs[ty][tx] = ker[oc=ocB][k=kB]
        float bv = 0.0f;
        if (ocB < N && kB < K) {
            int kw = kB % KW, tmp = kB / KW, kh = tmp % KH, ic = tmp / KH;
            bv = ker[(long)kw + KW*((long)kh + KH*((long)ic + IC*(long)ocB))];
        }
        Bs[ty][tx] = bv;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int kk = 0; kk < CTS; kk++) acc += Bs[ty][kk] * As[kk][tx];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (p < M && oc < N)
        o[(long)ow + OW*((long)oh + OH*((long)oc + OC*(long)nimg))] = acc;
}
// GroupNorm (no affine) over {W,H,C,N}: per (n,group) across W*H*(C/G).
// GroupNorm (no affine). Workgroup-parallel: one workgroup of GN_TG threads per (n,g),
// striding over the W·H·(C/G) group elements with a local-memory tree reduction.
// The original kernel had ONE thread handle up to ~262k elements sequentially at the
// largest VQGAN resolutions (256x256, cpg=4). cnt is computed in F32 (cpg<=16 here).
#define GN_TG 256
__kernel __attribute__((reqd_work_group_size(GN_TG,1,1)))
void k_group_norm(__global const float* a, __global float* o,
                  int W,int H,int C,int N,int G,float eps) {
    int tid = get_local_id(0); int ng = get_group_id(0);
    int g = ng % G, n = ng / G, cpg = C / G; long cnt = (long)W*H*cpg;
    int c0 = g*cpg; long base_n = (long)C*W*H*(long)n + (long)c0*W*H;
    __local float lp[GN_TG];
    // sum
    float s = 0.0f;
    for (long i = tid; i < cnt; i += GN_TG) s += a[base_n + i];
    lp[tid] = s; barrier(CLK_LOCAL_MEM_FENCE);
    for (int k = GN_TG>>1; k > 0; k >>= 1) {
        if (tid < k) lp[tid] += lp[tid + k]; barrier(CLK_LOCAL_MEM_FENCE);
    }
    float mean = lp[0] / (float)cnt;
    // var
    float v = 0.0f;
    for (long i = tid; i < cnt; i += GN_TG) { float t = a[base_n + i] - mean; v += t*t; }
    lp[tid] = v; barrier(CLK_LOCAL_MEM_FENCE);
    for (int k = GN_TG>>1; k > 0; k >>= 1) {
        if (tid < k) lp[tid] += lp[tid + k]; barrier(CLK_LOCAL_MEM_FENCE);
    }
    float inv = rsqrt(lp[0] / (float)cnt + eps);
    // apply
    for (long i = tid; i < cnt; i += GN_TG) o[base_n + i] = (a[base_n + i] - mean) * inv;
}
// nearest-neighbour upscale by f over {W,H,C,N}.
__kernel void k_upscale(__global const float* a, __global float* o, int OW,int OH,int C,int N,int f) {
    int ow=get_global_id(0), oh=get_global_id(1), cn=get_global_id(2);
    int c=cn%C, n=cn/C; if (ow>=OW||oh>=OH) return;
    int IW=OW/f, IH=OH/f;
    o[(long)ow + OW*((long)oh + OH*((long)c + C*(long)n))]
      = a[(long)(ow/f) + IW*((long)(oh/f) + IH*((long)c + C*(long)n))];
}
)CLC";

} // namespace

struct OpenCLRuntime::Impl {
    cl_context ctx = nullptr;
    cl_command_queue q = nullptr;
    cl_program prog = nullptr;
    cl_device_id dev = nullptr;
    std::string dev_name;
    int mr = 1;   // quantized-matmul M register-blocking factor, tuned per device (set in init)
    bool has_fp16 = false;   // cl_khr_fp16 (Mali yes; M1 OpenCL typically no)
    bool has_arm_dot = false;// cl_arm_integer_dot_product_accumulate_int8 (int8 matmul path)
    bool has_arm_conv = false;// int8 VQGAN conv (separate gate; per-tensor act quant is lossy)
    int wptm = 4, wptn = 4;  // quantized-FC micro-tile (matches kernel build -DGEMM_WPT*)
    cl_mem qx_buf = nullptr; size_t qx_sz = 0;   // int8 quantized-activation scratch (FC)
    cl_mem dx_buf = nullptr; size_t dx_sz = 0;   // activation per-block scales scratch (FC)
    // int8 conv: cached int8 conv weights (per ker Tensor*). The activation is gathered +
    // quantized per-(pixel,32-block) inside k_conv2d_i8, so no pre-pass scratch is needed.
    std::unordered_map<Tensor*, std::pair<cl_mem,cl_mem>> qconv;   // ker -> (qker, dker)
    cl_mem ensure(cl_mem& buf, size_t& sz, size_t need) {   // grow-on-demand scratch
        if (sz < need) { if (buf) clReleaseMemObject(buf); cl_int e;
            buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE, need, nullptr, &e); ck(e, "scratch"); sz = need; }
        return buf;
    }
    // quantize+cache conv weights ker[OC*K] (F32) -> int8 + per-(oc,32-block) scales.
    std::pair<cl_mem,cl_mem> conv_weights(cl_mem kerbuf, Tensor* kerT, int K, int OC) {
        auto it = qconv.find(kerT); if (it != qconv.end()) return it->second;
        int nb = (K+31)/32; cl_int e;
        cl_mem qk = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (size_t)K*OC, nullptr, &e); ck(e,"qker");
        cl_mem dk = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (size_t)OC*nb*sizeof(float), nullptr, &e); ck(e,"dker");
        cl_kernel kr = k("k_quantize_conv_w");
        clSetKernelArg(kr,0,sizeof(cl_mem),&kerbuf); clSetKernelArg(kr,1,sizeof(cl_mem),&qk);
        clSetKernelArg(kr,2,sizeof(cl_mem),&dk); clSetKernelArg(kr,3,sizeof(int),&K); clSetKernelArg(kr,4,sizeof(int),&OC);
        size_t g[2] = {(size_t)OC,(size_t)nb};
        ck(clEnqueueNDRangeKernel(q,kr,2,nullptr,g,nullptr,0,nullptr,nullptr),"qconv_w");
        qconv[kerT] = {qk,dk}; return {qk,dk};
    }
    bool prof_on = false, prof_print = false;          // per-op profiling (see header)
    std::unordered_map<int, double> prof_ms;           // keyed by op (MulMat split 400/401)
    std::unordered_map<int, int>    prof_n;
    std::unordered_map<std::string, cl_kernel> kernels;
    struct Buf { cl_mem mem; size_t sz; };
    std::unordered_map<Tensor*, Buf> bufs;

    cl_kernel k(const char* name) {
        auto it = kernels.find(name);
        if (it != kernels.end()) return it->second;
        cl_int e; cl_kernel ker = clCreateKernel(prog, name, &e); ck(e, name);
        kernels[name] = ker; return ker;
    }

    // Views (reshape/permute/view) carry their own strides but share the source
    // tensor's storage — resolve to the underlying buffer.
    //
    // Buffers persist across compute() calls (keyed by Tensor*). Iterative decoding
    // rebuilds the graph each step into the same reset arena, so scratch tensors reuse
    // host addresses AND sizes — the cached buffer is reused and simply overwritten by
    // its kernel that step (no per-step alloc/free churn). The size check recreates a
    // buffer only if the same address is later reused by a differently-sized tensor (a
    // different graph), so reuse is always safe. Leaf inputs whose *contents* change
    // (the decode token tensor) are refreshed via invalidate(); weights upload once.
    cl_mem buf(Tensor* t) {
        if (t->op == Op::Reshape || t->op == Op::Permute || t->op == Op::View)
            return buf(t->src[0]);
        size_t need = t->nbytes();
        auto it = bufs.find(t);
        if (it != bufs.end()) {
            if (it->second.sz == need) return it->second.mem;
            clReleaseMemObject(it->second.mem); bufs.erase(it);   // size changed: recreate
        }
        cl_int e;
        cl_mem m = clCreateBuffer(ctx, CL_MEM_READ_WRITE, need, nullptr, &e);
        ck(e, "clCreateBuffer");
        if (t->op == Op::None && t->data)   // leaf: upload host data once
            ck(clEnqueueWriteBuffer(q, m, CL_TRUE, 0, need, t->data, 0, nullptr, nullptr), "write leaf");
        bufs[t] = {m, need}; return m;
    }
    static int es(const Tensor* t, int d) { return (int)(t->nb[d] / sizeof(float)); }  // element stride

    void run1(cl_kernel ker, size_t global) {
        ck(clEnqueueNDRangeKernel(q, ker, 1, nullptr, &global, nullptr, 0, nullptr, nullptr), "enqueue1d");
    }
};

OpenCLRuntime::OpenCLRuntime() : p_(new Impl) {
    cl_uint np = 0; ck(clGetPlatformIDs(0, nullptr, &np), "platforms");
    std::vector<cl_platform_id> plats(np); ck(clGetPlatformIDs(np, plats.data(), nullptr), "platforms2");
    for (auto pl : plats) {
        cl_uint nd = 0;
        if (clGetDeviceIDs(pl, CL_DEVICE_TYPE_GPU, 1, &p_->dev, &nd) == CL_SUCCESS && nd) break;
        p_->dev = nullptr;
    }
    if (!p_->dev) throw std::runtime_error("opencl: no GPU device");
    char nm[256] = {0}; clGetDeviceInfo(p_->dev, CL_DEVICE_NAME, sizeof(nm), nm, nullptr);
    p_->dev_name = nm;
    // Quantized-matmul register-blocking is a win on bandwidth-bound mobile GPUs
    // (Mali: 18.7->14.4s) but loses on high-thread desktop GPUs that prefer raw
    // parallelism (M1 Max: 0.54->0.83s). Pick per device; <=MAXMR(8).
    p_->mr = (p_->dev_name.find("Mali") != std::string::npos ||
              p_->dev_name.find("Adreno") != std::string::npos) ? 4 : 1;
    {   // fp16 matmul path requires cl_khr_fp16 (Mali/Adreno yes; M1 OpenCL no)
        size_t n = 0; clGetDeviceInfo(p_->dev, CL_DEVICE_EXTENSIONS, 0, nullptr, &n);
        std::string ext(n, '\0'); clGetDeviceInfo(p_->dev, CL_DEVICE_EXTENSIONS, n, &ext[0], nullptr);
        p_->has_fp16 = ext.find("cl_khr_fp16") != std::string::npos;
        // int8-dot path (Q8_0): quantize activation to int8 + arm_dot_acc. ~14% faster
        // than the fp16 dequant path on Mali-G715 (native int8 datapath), cosine
        // 0.99999929. On by default where available; MG_NO_ARM_DOT=1 opts out.
        p_->has_arm_dot = ext.find("cl_arm_integer_dot_product_accumulate_int8") != std::string::npos
                          && std::getenv("MG_NO_ARM_DOT") == nullptr;
        // int8 VQGAN conv: per-(pixel,32-block) activation quant (gather-time) keeps VQGAN
        // cosine ~0.99997 (vs 0.9984 with a per-tensor scale; cf. F32 1.0). ~21% faster
        // end-to-end. On by default where arm_dot is available; MG_NO_ARM_CONV=1 opts out.
        p_->has_arm_conv = p_->has_arm_dot && std::getenv("MG_NO_ARM_CONV") == nullptr;
    }
    if (std::getenv("MG_OCL_PROF")) { p_->prof_on = true; p_->prof_print = true; }
    // Quantized-FC micro-tile config (work-per-thread). Default 4x4; override via env
    // for autotuning (MG_GEMM_WPTM / MG_GEMM_WPTN). The dispatch reads the same fields.
    if (const char* s = std::getenv("MG_GEMM_WPTM")) { int v = std::atoi(s); if (v>=1 && v<=8) p_->wptm = v; }
    if (const char* s = std::getenv("MG_GEMM_WPTN")) { int v = std::atoi(s); if (v>=1 && v<=8) p_->wptn = v; }
    cl_int e;
    p_->ctx = clCreateContext(nullptr, 1, &p_->dev, nullptr, nullptr, &e); ck(e, "context");
    p_->q = clCreateCommandQueue(p_->ctx, p_->dev, 0, &e); ck(e, "queue");
    p_->prog = clCreateProgramWithSource(p_->ctx, 1, &kKernels, nullptr, &e); ck(e, "program");
    char opts[64]; std::snprintf(opts, sizeof(opts), "-DGEMM_WPTM=%d -DGEMM_WPTN=%d", p_->wptm, p_->wptn);
    if (clBuildProgram(p_->prog, 1, &p_->dev, opts, nullptr, nullptr) != CL_SUCCESS) {
        char log[8192] = {0};
        clGetProgramBuildInfo(p_->prog, p_->dev, CL_PROGRAM_BUILD_LOG, sizeof(log), log, nullptr);
        throw std::runtime_error(std::string("opencl build failed:\n") + log);
    }
}

OpenCLRuntime::~OpenCLRuntime() {
    for (auto& kv : p_->bufs) clReleaseMemObject(kv.second.mem);
    if (p_->qx_buf) clReleaseMemObject(p_->qx_buf);
    if (p_->dx_buf) clReleaseMemObject(p_->dx_buf);
    for (auto& kv : p_->qconv) { clReleaseMemObject(kv.second.first); clReleaseMemObject(kv.second.second); }
    for (auto& kv : p_->kernels) clReleaseKernel(kv.second);
    if (p_->prog) clReleaseProgram(p_->prog);
    if (p_->q) clReleaseCommandQueue(p_->q);
    if (p_->ctx) clReleaseContext(p_->ctx);
}

const std::string& OpenCLRuntime::device_name() const { return p_->dev_name; }

void OpenCLRuntime::invalidate(Tensor* t) {
    auto it = p_->bufs.find(t);
    if (it != p_->bufs.end()) { clReleaseMemObject(it->second.mem); p_->bufs.erase(it); }
}

void OpenCLRuntime::profile_enable(bool on) { p_->prof_on = on; }
void OpenCLRuntime::profile_reset() { p_->prof_ms.clear(); p_->prof_n.clear(); }

std::vector<OpProfile> OpenCLRuntime::profile_report() const {
    auto name = [](int op) -> std::string {
        switch (op) {
            case 400: return "MulMat(q)";    // quantized FC
            case 401: return "MulMat(f32)";  // attention / conv-matmul
            case (int)Op::Add:       return "Add";
            case (int)Op::Mul:       return "Mul";
            case (int)Op::Scale:     return "Scale";
            case (int)Op::GetRows:   return "GetRows";
            case (int)Op::SoftMax:   return "SoftMax";
            case (int)Op::Norm:      return "Norm";
            case (int)Op::GroupNorm: return "GroupNorm";
            case (int)Op::Gelu:      return "Gelu";
            case (int)Op::Silu:      return "Silu";
            case (int)Op::Conv2D:    return "Conv2D";
            case (int)Op::Upscale:   return "Upscale";
            case (int)Op::FlashAttention: return "FlashAttn";
            default: return "op" + std::to_string(op);
        }
    };
    std::vector<OpProfile> r;
    for (auto& kv : p_->prof_ms) r.push_back({name(kv.first), p_->prof_n[kv.first], kv.second});
    std::sort(r.begin(), r.end(), [](const OpProfile& a, const OpProfile& b){ return a.ms > b.ms; });
    return r;
}

void OpenCLRuntime::compute(Graph& g) {
    Impl& I = *p_;
    auto setI = [](cl_kernel kr, cl_uint idx, int v) { ck(clSetKernelArg(kr, idx, sizeof(int), &v), "argi"); };
    auto setM = [](cl_kernel kr, cl_uint idx, cl_mem m) { ck(clSetKernelArg(kr, idx, sizeof(cl_mem), &m), "argm"); };

    const bool prof = I.prof_on;   // accumulates into I.prof_ms/n (see header / profile_*)
    using clk = std::chrono::steady_clock;

    for (Tensor* t : g.nodes) {
        clk::time_point pt0; if (prof) pt0 = clk::now();
        switch (t->op) {
            case Op::Reshape: case Op::Permute: case Op::View:
                break;                          // views: resolved lazily via buf()
            case Op::MulMat: {
                Tensor* w = t->src[0]; Tensor* x = t->src[1];
                cl_mem bw = I.buf(w), bx = I.buf(x), o = I.buf(t);
                int K=(int)w->ne[0], N=(int)w->ne[1], M=(int)x->ne[1], B2=(int)x->ne[2], B3=(int)x->ne[3];
                // fused-epilogue args (mul_mat_ex): bias=src[2], residual=src[3], act=iparam[0].
                // Bind a valid placeholder buffer (o) when absent; the has* flag gates the read.
                auto setEpi = [&](cl_kernel kr, int base) {
                    setM(kr, base+0, t->src[2] ? I.buf(t->src[2]) : o); setI(kr, base+1, t->src[2] ? 1 : 0);
                    setM(kr, base+2, t->src[3] ? I.buf(t->src[3]) : o); setI(kr, base+3, t->src[3] ? 1 : 0);
                    setI(kr, base+4, t->iparam[0]);
                };
                // int8 matmul (ARM dot product): Q8_0 weight, device has arm_dot, x
                // contiguous [K,M]. Quantize x->Q8_0 then int8xint8 dot (native datapath).
                if (w->type == Type::Q8_0 && I.has_arm_dot &&
                    x->nb[0]==sizeof(float) && x->nb[1]==(size_t)K*sizeof(float)) {
                    int nb = K/32;
                    cl_mem qxb = I.ensure(I.qx_buf, I.qx_sz, (size_t)K*M);
                    cl_mem dxb = I.ensure(I.dx_buf, I.dx_sz, (size_t)M*nb*sizeof(float));
                    cl_kernel kq = I.k("k_quantize_q8");
                    setM(kq,0,bx); setM(kq,1,qxb); setM(kq,2,dxb); setI(kq,3,K); setI(kq,4,M);
                    size_t qg[2] = {(size_t)M, (size_t)nb};
                    ck(clEnqueueNDRangeKernel(I.q, kq, 2, nullptr, qg, nullptr, 0, nullptr, nullptr), "quantize_q8");
                    cl_kernel kr = I.k("k_mul_mat_q8_i8");
                    setM(kr,0,bw); setM(kr,1,qxb); setM(kr,2,dxb); setM(kr,3,o);
                    setI(kr,4,K); setI(kr,5,N); setI(kr,6,M); setEpi(kr,7);
                    const int TS=64, RTS=16;
                    size_t gws[2]={(size_t)((M+TS-1)/TS)*RTS,(size_t)((N+TS-1)/TS)*RTS}, lws[2]={RTS,RTS};
                    ck(clEnqueueNDRangeKernel(I.q, kr, 2, nullptr, gws, lws, 0, nullptr, nullptr), "mulmat_i8");
                    break;
                }
                if (w->type == Type::Q8_0 || w->type == Type::Q4_K) {  // ggml dequant-fused (2D FC)
                    // 2D register-tiled GEMM: TSM x TSN (64x64) output tile per workgroup
                    // of RTSM x RTSN (16x16) work-items, each computing a WPTM x WPTN micro-
                    // tile. dim0 = m, dim1 = n. global = (#tiles)*RTS per dim.
                    const int RTSM=16, RTSN=16;
                    const int TSM=RTSM*I.wptm, TSN=RTSN*I.wptn;   // match kernel build -DGEMM_WPT*
                    // fp16 helps Q8_0 (cheap dequant -> ALU/local-mem bound) but not
                    // Q4_K (dequant-bound: unpacking 6-bit scales dominates, so the
                    // extra float->half cast only adds overhead). So Q8_0 only.
                    const char* kn = w->type == Type::Q8_0
                        ? (I.has_fp16 ? "k_mul_mat_q8_t2_h" : "k_mul_mat_q8_t2")
                        : "k_mul_mat_q4k_t2";
                    cl_kernel kr = I.k(kn);
                    setM(kr,0,bw); setM(kr,1,bx); setM(kr,2,o);
                    setI(kr,3,K); setI(kr,4,N); setI(kr,5,M); setEpi(kr,6);
                    size_t gws[2] = {(size_t)((M + TSM-1)/TSM)*RTSM, (size_t)((N + TSN-1)/TSN)*RTSN};
                    size_t lws[2] = {(size_t)RTSM, (size_t)RTSN};
                    ck(clEnqueueNDRangeKernel(I.q, kr, 2, nullptr, gws, lws, 0, nullptr, nullptr), "mulmat_qk");
                    break;
                }
                // Attention/F32 matmul. We tried an fp16 variant (k_mul_mat_t_h, kept in
                // source) but it REGRESSED on Mali: the attention matmuls are too small
                // (K=48 / K=257) to benefit from fp16's 2x ALU rate -- they're launch /
                // load-overhead-bound, and the per-load F32->half cast costs more than
                // fp16 saves. F32 kernel stays the default.
                cl_kernel kr = I.k("k_mul_mat_t");
                setM(kr,0,bw); setM(kr,1,bx); setM(kr,2,o);
                setI(kr,3,K); setI(kr,4,N); setI(kr,5,M); setI(kr,6,B2);
                for (int d=0;d<4;d++) setI(kr,7+d, Impl::es(w,d));
                for (int d=0;d<4;d++) setI(kr,11+d, Impl::es(x,d));
                setI(kr,15,(int)w->ne[2]); setI(kr,16,(int)w->ne[3]); setEpi(kr,17);
                const int TS = 16;
                auto up = [&](int v){ return (size_t)((v + TS - 1) / TS * TS); };
                size_t gws[3] = {up(M),up(N),(size_t)(B2*B3)}, lws[3] = {(size_t)TS,(size_t)TS,1};
                ck(clEnqueueNDRangeKernel(I.q, kr, 3, nullptr, gws, lws, 0, nullptr, nullptr), "mulmat");
                break;
            }
            case Op::GetRows: {
                cl_mem a=I.buf(t->src[0]), ids=I.buf(t->src[1]), o=I.buf(t);
                cl_kernel kr=I.k("k_get_rows");
                setM(kr,0,a); setM(kr,1,ids); setM(kr,2,o);
                setI(kr,3,(int)t->src[0]->ne[0]); setI(kr,4,(int)t->src[0]->ne[1]);
                size_t gws[2]={(size_t)t->src[0]->ne[0], (size_t)(t->src[1]->nelements())};
                ck(clEnqueueNDRangeKernel(I.q, kr, 2, nullptr, gws, nullptr, 0, nullptr, nullptr), "getrows");
                break;
            }
            case Op::Norm: case Op::SoftMax: {
                cl_mem a=I.buf(t->src[0]), o=I.buf(t);
                cl_kernel kr=I.k(t->op==Op::Norm ? "k_norm" : "k_soft_max");
                int D=(int)t->ne[0]; float fp=t->fparam[0];
                setM(kr,0,a); setM(kr,1,o); setI(kr,2,D);
                ck(clSetKernelArg(kr,3,sizeof(float),&fp),"argf");
                // Workgroup-parallel: one workgroup of TG threads per row (tree reduction).
                const size_t TG = 64;
                size_t rows = (size_t)(t->nelements()/D);
                size_t gws = rows * TG, lws = TG;
                ck(clEnqueueNDRangeKernel(I.q, kr, 1, nullptr, &gws, &lws, 0, nullptr, nullptr),
                   t->op==Op::Norm ? "norm" : "softmax");
                break;
            }
            case Op::Cont: {
                Tensor* s=t->src[0]; cl_mem a=I.buf(s), o=I.buf(t);
                cl_kernel kr=I.k("k_cont");
                setM(kr,0,a); setM(kr,1,o);
                for (int d=0;d<4;d++) setI(kr,2+d,(int)t->ne[d]);
                for (int d=0;d<4;d++) setI(kr,6+d, Impl::es(s,d));
                I.run1(kr, (size_t)t->nelements());
                break;
            }
            case Op::FlashAttention: {
                // Inputs are contiguous {D, S, H, B} (head dim innermost). Output same shape.
                Tensor* q=t->src[0]; Tensor* k=t->src[1]; Tensor* v=t->src[2];
                cl_mem bq=I.buf(q), bk=I.buf(k), bv=I.buf(v), o=I.buf(t);
                int S=(int)q->ne[1], H=(int)q->ne[2], B=(int)q->ne[3];
                float scale = t->fparam[0];
                cl_kernel kr = I.k("k_flash_attention");
                setM(kr,0,bq); setM(kr,1,bk); setM(kr,2,bv); setM(kr,3,o);
                setI(kr,4,S); ck(clSetKernelArg(kr,5,sizeof(float),&scale),"argf-scale");
                const int BR = 32;
                int n_qt = (S + BR - 1) / BR;
                size_t gws[3] = {(size_t)n_qt * BR, (size_t)H, (size_t)B};
                size_t lws[3] = {BR, 1, 1};
                ck(clEnqueueNDRangeKernel(I.q, kr, 3, nullptr, gws, lws, 0, nullptr, nullptr),
                   "flash_attention");
                break;
            }
            case Op::Add: case Op::Mul: {
                cl_mem a = I.buf(t->src[0]), b = I.buf(t->src[1]), o = I.buf(t);
                cl_kernel kr = I.k(t->op == Op::Add ? "k_add" : "k_mul");
                setM(kr,0,a); setM(kr,1,b); setM(kr,2,o);
                for (int d = 0; d < 4; d++) setI(kr, 3+d, (int)t->ne[d]);
                for (int d = 0; d < 4; d++) setI(kr, 7+d, (int)t->src[1]->ne[d]);
                I.run1(kr, (size_t)t->nelements());
                break;
            }
            case Op::Scale: {
                cl_mem a = I.buf(t->src[0]), o = I.buf(t);
                cl_kernel kr = I.k("k_scale"); float s = t->fparam[0];
                setM(kr,0,a); setM(kr,1,o); ck(clSetKernelArg(kr,2,sizeof(float),&s), "args");
                I.run1(kr, (size_t)t->nelements());
                break;
            }
            case Op::Gelu: case Op::Silu: {
                cl_mem a = I.buf(t->src[0]), o = I.buf(t);
                cl_kernel kr = I.k(t->op == Op::Gelu ? "k_gelu" : "k_silu");
                setM(kr,0,a); setM(kr,1,o);
                I.run1(kr, (size_t)t->nelements());
                break;
            }
            case Op::Conv2D: {
                Tensor* ker=t->src[0]; Tensor* in=t->src[1];
                cl_mem bin=I.buf(in), bk=I.buf(ker), o=I.buf(t);
                int IW=(int)in->ne[0], IH=(int)in->ne[1], IC=(int)in->ne[2];
                int KW=(int)ker->ne[0], KH=(int)ker->ne[1], OC=(int)t->ne[2];
                int OW=(int)t->ne[0], OH=(int)t->ne[1], Kc=KW*KH*IC, Mc=OW*OH;
                // int8 conv (arm_dot): pre-quantize weights (cached) + per-tensor-quantize
                // input, then implicit-GEMM int8 conv. Same flag as the FC int8 path.
                if (I.has_arm_conv && (long)IH*IC > 0) {
                    auto qkdk = I.conv_weights(bk, ker, Kc, OC);   // int8 weights (cached)
                    cl_kernel kc=I.k("k_conv2d_i8");               // F32 input gathered + quantized per-block in-kernel
                    setM(kc,0,bin); setM(kc,1,qkdk.first); setM(kc,2,qkdk.second); setM(kc,3,o);
                    setI(kc,4,IW); setI(kc,5,IH); setI(kc,6,IC); setI(kc,7,OW); setI(kc,8,OH); setI(kc,9,OC);
                    setI(kc,10,KW); setI(kc,11,KH); setI(kc,12,t->iparam[0]); setI(kc,13,t->iparam[1]);
                    const int TS=64, RTS=16;
                    size_t gws[3]={(size_t)((Mc+TS-1)/TS)*RTS,(size_t)((OC+TS-1)/TS)*RTS,(size_t)t->ne[3]}, lws[3]={RTS,RTS,1};
                    ck(clEnqueueNDRangeKernel(I.q,kc,3,nullptr,gws,lws,0,nullptr,nullptr),"conv_i8");
                    break;
                }
                cl_kernel kr=I.k("k_conv2d_t");   // tiled implicit-GEMM conv (F32 fallback)
                setM(kr,0,bin); setM(kr,1,bk); setM(kr,2,o);
                setI(kr,3,(int)in->ne[0]); setI(kr,4,(int)in->ne[1]); setI(kr,5,(int)in->ne[2]);
                setI(kr,6,(int)t->ne[0]); setI(kr,7,(int)t->ne[1]); setI(kr,8,(int)t->ne[2]);
                setI(kr,9,(int)ker->ne[0]); setI(kr,10,(int)ker->ne[1]);
                setI(kr,11,t->iparam[0]); setI(kr,12,t->iparam[1]);
                // implicit GEMM: M=OW*OH pixels (dim0), N=OC (dim1), batch (dim2)
                const int CTS = 16;
                int M = (int)(t->ne[0]*t->ne[1]), N = (int)t->ne[2];
                auto up = [&](int v){ return (size_t)((v + CTS - 1) / CTS * CTS); };
                size_t gws[3]={up(M), up(N), (size_t)t->ne[3]}, lws[3]={(size_t)CTS,(size_t)CTS,1};
                ck(clEnqueueNDRangeKernel(I.q, kr, 3, nullptr, gws, lws, 0, nullptr, nullptr), "conv2d");
                break;
            }
            case Op::GroupNorm: {
                cl_mem a=I.buf(t->src[0]), o=I.buf(t);
                cl_kernel kr=I.k("k_group_norm"); int G=t->iparam[0]; float eps=t->fparam[1];
                setM(kr,0,a); setM(kr,1,o);
                setI(kr,2,(int)t->ne[0]); setI(kr,3,(int)t->ne[1]); setI(kr,4,(int)t->ne[2]); setI(kr,5,(int)t->ne[3]);
                setI(kr,6,G); ck(clSetKernelArg(kr,7,sizeof(float),&eps),"argf");
                // Workgroup-parallel: one workgroup of TG threads per (n,g), tree reduction.
                const size_t TG = 256;
                size_t gws = (size_t)(t->ne[3]*G) * TG, lws = TG;
                ck(clEnqueueNDRangeKernel(I.q, kr, 1, nullptr, &gws, &lws, 0, nullptr, nullptr), "group_norm");
                break;
            }
            case Op::Upscale: {
                cl_mem a=I.buf(t->src[0]), o=I.buf(t);
                cl_kernel kr=I.k("k_upscale"); int f=t->iparam[0];
                setM(kr,0,a); setM(kr,1,o);
                setI(kr,2,(int)t->ne[0]); setI(kr,3,(int)t->ne[1]); setI(kr,4,(int)t->ne[2]); setI(kr,5,(int)t->ne[3]); setI(kr,6,f);
                size_t gws[3]={(size_t)t->ne[0],(size_t)t->ne[1],(size_t)(t->ne[2]*t->ne[3])};
                ck(clEnqueueNDRangeKernel(I.q, kr, 3, nullptr, gws, nullptr, 0, nullptr, nullptr), "upscale");
                break;
            }
            default:
                throw std::runtime_error("opencl: op not implemented yet (" + std::to_string((int)t->op) + ")");
        }
        if (prof) {
            clFinish(I.q);
            // split MulMat into quantized-FC (400) vs F32 (401) so we can tell which dominates
            int key = (int)t->op;
            if (t->op == Op::MulMat)
                key = (t->src[0]->type == Type::Q8_0 || t->src[0]->type == Type::Q4_K) ? 400 : 401;
            I.prof_ms[key] += std::chrono::duration<double, std::milli>(clk::now() - pt0).count();
            I.prof_n[key]++;
        }
    }
    if (prof && I.prof_print) {   // ad-hoc per-compute() dump when MG_OCL_PROF is set
        std::printf("[ocl-prof] per-op-type cumulative wall ms (clFinish-serialized):\n");
        for (auto& r : profile_report())
            std::printf("  %-12s n=%-4d %.1f ms\n", r.op.c_str(), r.count, r.ms);
        std::fflush(stdout);
    }
    // Keep all intermediates on the GPU; read back only the final output node.
    // (Inference only needs the graph output; this avoids ~hundreds of blocking
    // device->host transfers + syncs that dominated the naive executor.)
    if (!g.nodes.empty()) {
        Tensor* out = g.nodes.back();
        ck(clEnqueueReadBuffer(I.q, I.buf(out), CL_TRUE, 0, out->nbytes(), out->data, 0, nullptr, nullptr),
           ("readback " + out->name).c_str());
    }
    clFinish(I.q);

    // Free this graph's computed-node buffers, keeping leaf weight/input buffers cached.
    // Tested: this barely changes device latency (Mali is compute-bound on the kernels,
    // not buffer-alloc-bound), but it keeps peak RSS at the single-step working set
    // rather than accumulating the transformer's scratch through the VQGAN decode
    // (~730 MB lower on the phone). The size-checked buf() cache makes leaf reuse safe;
    // recomputed scratch is cheap to recreate next step.
    for (Tensor* t : g.nodes) {
        if (t->op == Op::None || t->op == Op::Reshape || t->op == Op::Permute || t->op == Op::View)
            continue;
        auto it = I.bufs.find(t);
        if (it != I.bufs.end()) { clReleaseMemObject(it->second.mem); I.bufs.erase(it); }
    }
}

} // namespace mg
