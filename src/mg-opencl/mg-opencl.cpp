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
// w[K,N], x[K,M] -> out[N,M]; out[n,m] = sum_k w[k,n]*x[k,m]. (ne0 innermost)
__kernel void k_mul_mat(__global const float* w, __global const float* x,
                        __global float* o, const int K, const int N, const int M) {
    int n = get_global_id(0), m = get_global_id(1);
    if (n >= N || m >= M) return;
    float acc = 0.0f;
    for (int k = 0; k < K; k++) acc += w[n*K + k] * x[m*K + k];
    o[m*N + n] = acc;
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

    cl_mem buf(Tensor* t) {
        auto it = bufs.find(t);
        if (it != bufs.end()) return it->second;
        cl_int e;
        cl_mem m = clCreateBuffer(ctx, CL_MEM_READ_WRITE, t->nbytes(), nullptr, &e);
        ck(e, "clCreateBuffer");
        if (t->op == Op::None && t->data)   // leaf: upload host data once
            ck(clEnqueueWriteBuffer(q, m, CL_TRUE, 0, t->nbytes(), t->data, 0, nullptr, nullptr), "write leaf");
        bufs[t] = m; return m;
    }

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
            case Op::Reshape: {                 // view: alias the source buffer (contiguous)
                I.bufs[t] = I.buf(t->src[0]);
                break;
            }
            case Op::MulMat: {
                cl_mem w = I.buf(t->src[0]), x = I.buf(t->src[1]), o = I.buf(t);
                int K = (int)t->src[0]->ne[0], N = (int)t->src[0]->ne[1], M = (int)t->src[1]->ne[1];
                if (t->src[0]->ne[2] != 1 || t->src[1]->ne[2] != 1)
                    throw std::runtime_error("opencl mul_mat: batched/strided not implemented yet");
                cl_kernel kr = I.k("k_mul_mat");
                setM(kr,0,w); setM(kr,1,x); setM(kr,2,o); setI(kr,3,K); setI(kr,4,N); setI(kr,5,M);
                size_t gws[2] = {(size_t)N, (size_t)M};
                ck(clEnqueueNDRangeKernel(I.q, kr, 2, nullptr, gws, nullptr, 0, nullptr, nullptr), "mulmat");
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
            default:
                throw std::runtime_error("opencl: op not implemented yet (" + std::to_string((int)t->op) + ")");
        }
    }
    // read computed nodes back into host data
    for (Tensor* t : g.nodes) {
        if (t->op == Op::Reshape) continue;
        ck(clEnqueueReadBuffer(I.q, I.bufs[t], CL_TRUE, 0, t->nbytes(), t->data, 0, nullptr, nullptr), "readback");
    }
    clFinish(I.q);
}

} // namespace mg
