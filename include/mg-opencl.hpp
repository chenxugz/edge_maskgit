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

namespace mg {

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

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

} // namespace mg
