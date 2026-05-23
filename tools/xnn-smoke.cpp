// xnn-smoke.cpp — confirm XNNPACK links and the subgraph->runtime flow works.
// Builds y = fully_connected(x[2,3], W[4,3], b[4]) -> [2,4] and checks vs a
// hand-computed reference.
#include "xnnpack.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

int main() {
    if (xnn_initialize(nullptr) != xnn_status_success) {
        std::fprintf(stderr, "xnn_initialize failed\n"); return 1;
    }

    const size_t B = 2, IC = 3, OC = 4;
    std::vector<float> x = {1, 2, 3,  -1, 0, 1};                 // [B, IC]
    std::vector<float> W = {1, 0, 0,  0, 1, 0,  0, 0, 1,  1, 1, 1};  // [OC, IC]
    std::vector<float> bias = {0.5f, -0.5f, 1.0f, 0.0f};        // [OC]
    std::vector<float> y(B * OC, 0);

    xnn_subgraph_t sg = nullptr;
    xnn_create_subgraph(/*external_value_ids=*/2, 0, &sg);

    uint32_t in_id, w_id, b_id, out_id;
    size_t xdims[2] = {B, IC};
    size_t wdims[2] = {OC, IC};
    size_t bdims[1] = {OC};
    size_t ydims[2] = {B, OC};
    xnn_define_tensor_value(sg, xnn_datatype_fp32, 2, xdims, nullptr, 0,
                            XNN_VALUE_FLAG_EXTERNAL_INPUT, &in_id);
    xnn_define_tensor_value(sg, xnn_datatype_fp32, 2, wdims, W.data(), XNN_INVALID_VALUE_ID, 0, &w_id);
    xnn_define_tensor_value(sg, xnn_datatype_fp32, 1, bdims, bias.data(), XNN_INVALID_VALUE_ID, 0, &b_id);
    xnn_define_tensor_value(sg, xnn_datatype_fp32, 2, ydims, nullptr, 1,
                            XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &out_id);

    if (xnn_define_fully_connected(sg, -INFINITY, INFINITY, in_id, w_id, b_id, out_id, 0)
            != xnn_status_success) {
        std::fprintf(stderr, "define_fully_connected failed\n"); return 1;
    }

    xnn_runtime_t rt = nullptr;
    if (xnn_create_runtime_v2(sg, /*threadpool=*/nullptr, 0, &rt) != xnn_status_success) {
        std::fprintf(stderr, "create_runtime failed\n"); return 1;
    }
    xnn_external_value ext[2] = {{in_id, x.data()}, {out_id, y.data()}};
    xnn_setup_runtime(rt, 2, ext);
    xnn_invoke_runtime(rt);

    // reference: y[b,o] = sum_i x[b,i]*W[o,i] + bias[o]
    float maxdiff = 0;
    for (size_t b = 0; b < B; b++)
        for (size_t o = 0; o < OC; o++) {
            float ref = bias[o];
            for (size_t i = 0; i < IC; i++) ref += x[b*IC+i] * W[o*IC+i];
            maxdiff = std::fmax(maxdiff, std::fabs(y[b*OC+o] - ref));
        }
    std::printf("XNNPACK FC subgraph maxdiff=%.3e  y=[%.2f %.2f %.2f %.2f / %.2f %.2f %.2f %.2f]\n",
                maxdiff, y[0],y[1],y[2],y[3],y[4],y[5],y[6],y[7]);
    xnn_delete_runtime(rt);
    xnn_delete_subgraph(sg);
    std::printf("%s\n", maxdiff < 1e-5f ? "XNN SMOKE: PASS" : "XNN SMOKE: FAIL");
    return maxdiff < 1e-5f ? 0 : 1;
}
