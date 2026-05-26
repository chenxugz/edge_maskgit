// mg-opencl.hpp — OpenCL backend: execute an mg::Graph on the GPU.
//
// Mirrors the reference mg::compute(): walks the topologically-sorted graph and
// dispatches an OpenCL kernel per node. Leaf tensors (weights/inputs) are
// uploaded once; computed tensors get device buffers; results are read back into
// each node's host data. pImpl keeps the OpenCL headers out of this header.
#pragma once
#include "mg-tensor.hpp"

#include <memory>
#include <string>
#include <vector>

namespace mg {

// One row of the per-op-type GPU profile (see OpenCLRuntime::profile_report).
struct OpProfile {
    std::string op;     // op-type name (MulMat split into "MulMat(q)" / "MulMat(f32)")
    int         count;  // number of kernel dispatches of this type
    double      ms;     // accumulated wall time (clFinish-serialized)
};

class OpenCLRuntime {
public:
    OpenCLRuntime();              // picks a GPU device, builds the kernel program
    ~OpenCLRuntime();
    OpenCLRuntime(const OpenCLRuntime&) = delete;
    OpenCLRuntime& operator=(const OpenCLRuntime&) = delete;

    void compute(Graph& g);       // run graph on GPU; fills each computed node's host data
    const std::string& device_name() const;

    // Drop a tensor's cached device buffer so the next compute() re-uploads it.
    // Needed for iterative decoding: the input token leaf changes every step but its
    // host address is stable (arena reset reuses it), so without this the GPU keeps
    // the stale first-step upload. Weights/scratch stay cached.
    void invalidate(Tensor* t);

    // Per-op-type profiling. When on, compute() clFinish()es after each node and
    // accumulates wall time by op type — serialized, so absolute totals inflate vs the
    // real overlapped run, but the relative split reliably finds the bottleneck.
    // (Also auto-enabled + printed per compute() if the env var MG_OCL_PROF is set.)
    void profile_enable(bool on);
    void profile_reset();
    std::vector<OpProfile> profile_report() const;   // descending by ms

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

} // namespace mg
