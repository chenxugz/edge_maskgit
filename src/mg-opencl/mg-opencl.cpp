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

#include <cstdio>
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
// ggml Q8_0 dequant-fused matmul. w = N rows x (K/32) blocks, each block = 34 bytes
// (fp16 scale + 32 int8). x[K,M] f32 -> out[N,M]. out[n,m]=sum_b d_b*sum_i q[i]*x[m*K+b*32+i].
__kernel void k_mul_mat_q8(__global const uchar* w, __global const float* x, __global float* o,
                           int K, int N, int M) {
    int n = get_global_id(0), m = get_global_id(1); if (n >= N || m >= M) return;
    int nb = K / 32; float acc = 0.0f;
    for (int b = 0; b < nb; b++) {
        __global const uchar* blk = w + (long)(n*nb + b) * 34;
        float d = vload_half(0, (__global const half*)blk);
        __global const char* qs = (__global const char*)(blk + 2);
        long xo = (long)m*K + (long)b*32;
        for (int i = 0; i < 32; i++) acc += d * (float)qs[i] * x[xo + i];
    }
    o[(long)m*N + n] = acc;
}
// gather rows of a[E,R] by I32 ids -> out[E,n]; negative id wraps (mask token).
__kernel void k_get_rows(__global const float* a, __global const int* ids, __global float* o,
                         int E, int R) {
    int e = get_global_id(0), i = get_global_id(1);
    int r = ids[i]; if (r < 0) r += R;
    o[(long)i*E + e] = a[(long)r*E + e];
}
// LayerNorm over ne0 (no affine).
__kernel void k_norm(__global const float* a, __global float* o, int D, float eps) {
    int row = get_global_id(0);
    __global const float* r = a + (long)row*D;
    float mean = 0.0f; for (int d=0; d<D; d++) mean += r[d]; mean /= D;
    float var = 0.0f; for (int d=0; d<D; d++) { float t = r[d]-mean; var += t*t; } var /= D;
    float inv = rsqrt(var + eps);
    for (int d=0; d<D; d++) o[(long)row*D + d] = (r[d]-mean) * inv;
}
// softmax over ne0 with scale.
__kernel void k_soft_max(__global const float* a, __global float* o, int D, float scale) {
    int row = get_global_id(0);
    __global const float* r = a + (long)row*D;
    float mx = -INFINITY; for (int d=0; d<D; d++) mx = fmax(mx, r[d]*scale);
    float sum = 0.0f; for (int d=0; d<D; d++) { float e = exp(r[d]*scale - mx); o[(long)row*D+d]=e; sum+=e; }
    float inv = 1.0f/sum; for (int d=0; d<D; d++) o[(long)row*D+d] *= inv;
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
// GroupNorm (no affine) over {W,H,C,N}: per (n,group) across W*H*(C/G).
__kernel void k_group_norm(__global const float* a, __global float* o,
                           int W,int H,int C,int N,int G,float eps) {
    int ng = get_global_id(0); if (ng >= N*G) return;
    int g=ng%G, n=ng/G, cpg=C/G; long cnt=(long)W*H*cpg;
    float mean=0.0f;
    for (int c=g*cpg; c<(g+1)*cpg; c++) for (int h=0;h<H;h++) for (int w=0;w<W;w++)
        mean += a[(long)w + W*((long)h + H*((long)c + C*(long)n))];
    mean /= cnt;
    float var=0.0f;
    for (int c=g*cpg; c<(g+1)*cpg; c++) for (int h=0;h<H;h++) for (int w=0;w<W;w++) {
        float t = a[(long)w + W*((long)h + H*((long)c + C*(long)n))] - mean; var += t*t; }
    float inv = rsqrt(var/cnt + eps);
    for (int c=g*cpg; c<(g+1)*cpg; c++) for (int h=0;h<H;h++) for (int w=0;w<W;w++) {
        long idx=(long)w + W*((long)h + H*((long)c + C*(long)n)); o[idx]=(a[idx]-mean)*inv; }
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
    std::unordered_map<std::string, cl_kernel> kernels;
    std::unordered_map<Tensor*, cl_mem> bufs;

    cl_kernel k(const char* name) {
        auto it = kernels.find(name);
        if (it != kernels.end()) return it->second;
        cl_int e; cl_kernel ker = clCreateKernel(prog, name, &e); ck(e, name);
        kernels[name] = ker; return ker;
    }

    // Views (reshape/permute/view) carry their own strides but share the source
    // tensor's storage — resolve to the underlying buffer.
    cl_mem buf(Tensor* t) {
        if (t->op == Op::Reshape || t->op == Op::Permute || t->op == Op::View)
            return buf(t->src[0]);
        auto it = bufs.find(t);
        if (it != bufs.end()) return it->second;
        cl_int e;
        cl_mem m = clCreateBuffer(ctx, CL_MEM_READ_WRITE, t->nbytes(), nullptr, &e);
        ck(e, "clCreateBuffer");
        if (t->op == Op::None && t->data)   // leaf: upload host data once
            ck(clEnqueueWriteBuffer(q, m, CL_TRUE, 0, t->nbytes(), t->data, 0, nullptr, nullptr), "write leaf");
        bufs[t] = m; return m;
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
    cl_int e;
    p_->ctx = clCreateContext(nullptr, 1, &p_->dev, nullptr, nullptr, &e); ck(e, "context");
    p_->q = clCreateCommandQueue(p_->ctx, p_->dev, 0, &e); ck(e, "queue");
    p_->prog = clCreateProgramWithSource(p_->ctx, 1, &kKernels, nullptr, &e); ck(e, "program");
    if (clBuildProgram(p_->prog, 1, &p_->dev, "", nullptr, nullptr) != CL_SUCCESS) {
        char log[8192] = {0};
        clGetProgramBuildInfo(p_->prog, p_->dev, CL_PROGRAM_BUILD_LOG, sizeof(log), log, nullptr);
        throw std::runtime_error(std::string("opencl build failed:\n") + log);
    }
}

OpenCLRuntime::~OpenCLRuntime() {
    for (auto& kv : p_->bufs) clReleaseMemObject(kv.second);
    for (auto& kv : p_->kernels) clReleaseKernel(kv.second);
    if (p_->prog) clReleaseProgram(p_->prog);
    if (p_->q) clReleaseCommandQueue(p_->q);
    if (p_->ctx) clReleaseContext(p_->ctx);
}

const std::string& OpenCLRuntime::device_name() const { return p_->dev_name; }

void OpenCLRuntime::compute(Graph& g) {
    Impl& I = *p_;
    auto setI = [](cl_kernel kr, cl_uint idx, int v) { ck(clSetKernelArg(kr, idx, sizeof(int), &v), "argi"); };
    auto setM = [](cl_kernel kr, cl_uint idx, cl_mem m) { ck(clSetKernelArg(kr, idx, sizeof(cl_mem), &m), "argm"); };

    for (Tensor* t : g.nodes) {
        switch (t->op) {
            case Op::Reshape: case Op::Permute: case Op::View:
                break;                          // views: resolved lazily via buf()
            case Op::MulMat: {
                Tensor* w = t->src[0]; Tensor* x = t->src[1];
                cl_mem bw = I.buf(w), bx = I.buf(x), o = I.buf(t);
                int K=(int)w->ne[0], N=(int)w->ne[1], M=(int)x->ne[1], B2=(int)x->ne[2], B3=(int)x->ne[3];
                if (w->type == Type::Q8_0) {     // ggml Q8_0 dequant-fused (2D FC, x f32 contiguous)
                    cl_kernel kr = I.k("k_mul_mat_q8");
                    setM(kr,0,bw); setM(kr,1,bx); setM(kr,2,o);
                    setI(kr,3,K); setI(kr,4,N); setI(kr,5,M);
                    size_t gws[2] = {(size_t)N,(size_t)M};
                    ck(clEnqueueNDRangeKernel(I.q, kr, 2, nullptr, gws, nullptr, 0, nullptr, nullptr), "mulmat_q8");
                    break;
                }
                cl_kernel kr = I.k("k_mul_mat");
                setM(kr,0,bw); setM(kr,1,bx); setM(kr,2,o);
                setI(kr,3,K); setI(kr,4,N); setI(kr,5,M); setI(kr,6,B2);
                for (int d=0;d<4;d++) setI(kr,7+d, Impl::es(w,d));
                for (int d=0;d<4;d++) setI(kr,11+d, Impl::es(x,d));
                setI(kr,15,(int)w->ne[2]); setI(kr,16,(int)w->ne[3]);
                size_t gws[3] = {(size_t)N,(size_t)M,(size_t)(B2*B3)};
                ck(clEnqueueNDRangeKernel(I.q, kr, 3, nullptr, gws, nullptr, 0, nullptr, nullptr), "mulmat");
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
                I.run1(kr, (size_t)(t->nelements()/D));
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
                cl_kernel kr=I.k("k_conv2d");
                setM(kr,0,bin); setM(kr,1,bk); setM(kr,2,o);
                setI(kr,3,(int)in->ne[0]); setI(kr,4,(int)in->ne[1]); setI(kr,5,(int)in->ne[2]);
                setI(kr,6,(int)t->ne[0]); setI(kr,7,(int)t->ne[1]); setI(kr,8,(int)t->ne[2]);
                setI(kr,9,(int)ker->ne[0]); setI(kr,10,(int)ker->ne[1]);
                setI(kr,11,t->iparam[0]); setI(kr,12,t->iparam[1]);
                size_t gws[3]={(size_t)t->ne[0],(size_t)t->ne[1],(size_t)(t->ne[2]*t->ne[3])};
                ck(clEnqueueNDRangeKernel(I.q, kr, 3, nullptr, gws, nullptr, 0, nullptr, nullptr), "conv2d");
                break;
            }
            case Op::GroupNorm: {
                cl_mem a=I.buf(t->src[0]), o=I.buf(t);
                cl_kernel kr=I.k("k_group_norm"); int G=t->iparam[0]; float eps=t->fparam[1];
                setM(kr,0,a); setM(kr,1,o);
                setI(kr,2,(int)t->ne[0]); setI(kr,3,(int)t->ne[1]); setI(kr,4,(int)t->ne[2]); setI(kr,5,(int)t->ne[3]);
                setI(kr,6,G); ck(clSetKernelArg(kr,7,sizeof(float),&eps),"argf");
                I.run1(kr, (size_t)(t->ne[3]*G));
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
    }
    // read computed nodes back into host data (skip views — no own buffer)
    for (Tensor* t : g.nodes) {
        if (t->op == Op::Reshape || t->op == Op::Permute || t->op == Op::View) continue;
        ck(clEnqueueReadBuffer(I.q, I.bufs[t], CL_TRUE, 0, t->nbytes(), t->data, 0, nullptr, nullptr), "readback");
    }
    clFinish(I.q);
}

} // namespace mg
