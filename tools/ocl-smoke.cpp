// ocl-smoke.cpp — confirm the OpenCL toolchain works: compile a matmul kernel at
// runtime, run it on the GPU, and check the result against a CPU reference.
// Uses the mg mul_mat convention: w[K,N], x[K,M] -> out[N,M], out[n,m]=sum_k w[k,n]*x[k,m].
#if defined(__APPLE__)
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static const char* kMatMul = R"CLC(
__kernel void mul_mat(__global const float* w, __global const float* x,
                      __global float* out, const int K, const int N, const int M) {
    int n = get_global_id(0);
    int m = get_global_id(1);
    if (n >= N || m >= M) return;
    float acc = 0.0f;
    for (int k = 0; k < K; k++) acc += w[k * N + n] * x[k * M + m];
    out[m * N + n] = acc;   // out[N,M] with n inner
}
)CLC";

#define CK(call) do { cl_int _e = (call); if (_e != CL_SUCCESS) { \
    std::fprintf(stderr, "OpenCL error %d at %s:%d\n", _e, __FILE__, __LINE__); return 1; } } while (0)

int main() {
    cl_uint nplat = 0;
    CK(clGetPlatformIDs(0, nullptr, &nplat));
    if (nplat == 0) { std::fprintf(stderr, "no OpenCL platforms\n"); return 1; }
    std::vector<cl_platform_id> plats(nplat);
    CK(clGetPlatformIDs(nplat, plats.data(), nullptr));

    cl_device_id dev = nullptr;
    for (cl_platform_id p : plats) {
        cl_uint nd = 0;
        if (clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 1, &dev, &nd) == CL_SUCCESS && nd > 0) break;
        dev = nullptr;
    }
    if (!dev) { std::fprintf(stderr, "no GPU device\n"); return 1; }
    char name[256] = {0};
    clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(name), name, nullptr);
    std::printf("GPU device: %s\n", name);

    cl_int err;
    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err); CK(err);
    cl_command_queue q = clCreateCommandQueue(ctx, dev, 0, &err); CK(err);
    cl_program prog = clCreateProgramWithSource(ctx, 1, &kMatMul, nullptr, &err); CK(err);
    if (clBuildProgram(prog, 1, &dev, "", nullptr, nullptr) != CL_SUCCESS) {
        char log[4096] = {0};
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, sizeof(log), log, nullptr);
        std::fprintf(stderr, "build failed:\n%s\n", log); return 1;
    }
    cl_kernel ker = clCreateKernel(prog, "mul_mat", &err); CK(err);

    const int K = 64, N = 48, M = 40;
    std::vector<float> w(K * N), x(K * M), out(N * M), ref(N * M);
    for (int i = 0; i < K * N; i++) w[i] = std::sin(0.1f * i + 1.0f);
    for (int i = 0; i < K * M; i++) x[i] = std::cos(0.07f * i + 0.3f);
    for (int m = 0; m < M; m++) for (int n = 0; n < N; n++) {
        double a = 0; for (int k = 0; k < K; k++) a += (double)w[k*N+n] * x[k*M+m];
        ref[m*N + n] = (float)a;
    }

    cl_mem bw = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*K*N, w.data(), &err); CK(err);
    cl_mem bx = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*K*M, x.data(), &err); CK(err);
    cl_mem bo = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof(float)*N*M, nullptr, &err); CK(err);
    CK(clSetKernelArg(ker, 0, sizeof(cl_mem), &bw));
    CK(clSetKernelArg(ker, 1, sizeof(cl_mem), &bx));
    CK(clSetKernelArg(ker, 2, sizeof(cl_mem), &bo));
    CK(clSetKernelArg(ker, 3, sizeof(int), &K));
    CK(clSetKernelArg(ker, 4, sizeof(int), &N));
    CK(clSetKernelArg(ker, 5, sizeof(int), &M));
    size_t gws[2] = {(size_t)N, (size_t)M};
    CK(clEnqueueNDRangeKernel(q, ker, 2, nullptr, gws, nullptr, 0, nullptr, nullptr));
    CK(clEnqueueReadBuffer(q, bo, CL_TRUE, 0, sizeof(float)*N*M, out.data(), 0, nullptr, nullptr));

    float maxdiff = 0;
    for (int i = 0; i < N * M; i++) maxdiff = std::fmax(maxdiff, std::fabs(out[i] - ref[i]));
    std::printf("mul_mat maxdiff=%.3e\n", maxdiff);
    std::printf("%s\n", maxdiff < 1e-3f ? "OCL SMOKE: PASS" : "OCL SMOKE: FAIL");
    return maxdiff < 1e-3f ? 0 : 1;
}
