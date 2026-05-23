// test_kernels.cpp — validate F32 kernels against independent in-test references.
// Catches stride/indexing bugs (the real risk) by recomputing each result a
// different way (naive double-precision loops or closed-form).
#include "mg-tensor.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace mg;

static int g_fail = 0;
static void check(bool ok, const char* name, float maxdiff = -1) {
    if (ok) { std::printf("  PASS  %s", name); }
    else    { std::printf("  FAIL  %s", name); g_fail++; }
    if (maxdiff >= 0) std::printf("   (maxdiff=%.3e)", maxdiff);
    std::printf("\n");
}

static float fill(Tensor* t) {  // fill with a deterministic pattern, return nothing useful
    int64_t n = t->nelements();
    for (int64_t i = 0; i < n; i++) t->set_f32(i, std::sin(0.1f * i + 1.0f) * 2.0f);
    return 0;
}

static float run_and_max(Context& c, Tensor* out, const std::vector<float>& ref) {
    Graph g; g.build_forward(out); compute(c, g);
    float md = 0;
    for (size_t i = 0; i < ref.size(); i++)
        md = std::fmax(md, std::fabs(out->f32((int64_t)i) - ref[i]));
    return md;
}

static void test_mul_mat() {
    // Linear: weight W[N=4,K=3] (ne={3,4}), input x[M=2,K=3] (ne={3,2}) -> y[N,M] (ne={4,2})
    Context c(1 << 20);
    const int K = 3, N = 4, M = 2;
    Tensor* w = c.tensor2d(Type::F32, K, N);  fill(w);
    Tensor* x = c.tensor2d(Type::F32, K, M);  fill(x);
    Tensor* y = mul_mat(c, w, x);
    std::vector<float> ref(N * M);
    for (int m = 0; m < M; m++) for (int n = 0; n < N; n++) {
        double a = 0; for (int k = 0; k < K; k++) a += (double)w->f32(n*K+k) * x->f32(m*K+k);
        ref[m*N + n] = (float)a;
    }
    float md = run_and_max(c, y, ref);
    check(md < 1e-5f, "mul_mat (Linear y=x@W^T)", md);
}

static void test_add_bias() {
    Context c(1 << 20);
    const int D = 5, S = 3;
    Tensor* a = c.tensor2d(Type::F32, D, S);   fill(a);
    Tensor* b = c.tensor1d(Type::F32, D);      fill(b);   // bias broadcast over S
    Tensor* y = add_bias(c, a, b);
    std::vector<float> ref(D * S);
    for (int s = 0; s < S; s++) for (int d = 0; d < D; d++) ref[s*D+d] = a->f32(s*D+d) + b->f32(d);
    float md = run_and_max(c, y, ref);
    check(md < 1e-6f, "add_bias (broadcast over seq)", md);
}

static void test_softmax() {
    Context c(1 << 20);
    const int D = 6, R = 2;
    Tensor* a = c.tensor2d(Type::F32, D, R);   fill(a);
    Tensor* y = soft_max(c, a, 1.0f);
    std::vector<float> ref(D * R);
    for (int r = 0; r < R; r++) {
        double mx = -1e30, sum = 0;
        for (int d = 0; d < D; d++) mx = std::fmax(mx, a->f32(r*D+d));
        for (int d = 0; d < D; d++) sum += std::exp(a->f32(r*D+d) - mx);
        for (int d = 0; d < D; d++) ref[r*D+d] = (float)(std::exp(a->f32(r*D+d) - mx) / sum);
    }
    float md = run_and_max(c, y, ref);
    check(md < 1e-6f, "soft_max (rowwise)", md);
}

static void test_norm() {
    Context c(1 << 20);
    const int D = 8, R = 3;
    Tensor* a = c.tensor2d(Type::F32, D, R);   fill(a);
    Tensor* y = norm(c, a, 1e-5f);
    std::vector<float> ref(D * R);
    for (int r = 0; r < R; r++) {
        double mean = 0; for (int d = 0; d < D; d++) mean += a->f32(r*D+d); mean /= D;
        double var = 0; for (int d = 0; d < D; d++){ double t=a->f32(r*D+d)-mean; var+=t*t; } var/=D;
        double inv = 1.0/std::sqrt(var + 1e-5);
        for (int d = 0; d < D; d++) ref[r*D+d] = (float)((a->f32(r*D+d)-mean)*inv);
    }
    float md = run_and_max(c, y, ref);
    check(md < 1e-5f, "norm (LayerNorm, no affine)", md);
}

static void test_gelu_silu() {
    Context c(1 << 20);
    const int D = 16;
    Tensor* a = c.tensor1d(Type::F32, D); fill(a);
    Tensor* yg = gelu(c, a);
    { Graph g; g.build_forward(yg); compute(c, g); }
    Tensor* as = c.tensor1d(Type::F32, D); fill(as);
    Tensor* ys = silu(c, as);
    { Graph g; g.build_forward(ys); compute(c, g); }
    float mg_ = 0, ms = 0;
    for (int i = 0; i < D; i++) {
        float x = a->f32(i);
        float refg = 0.5f*x*(1.0f+std::erf(x*0.70710678f));
        float refs = x/(1.0f+std::exp(-x));
        mg_ = std::fmax(mg_, std::fabs(yg->f32(i)-refg));
        ms  = std::fmax(ms,  std::fabs(ys->f32(i)-refs));
    }
    check(mg_ < 1e-6f, "gelu (exact erf)", mg_);
    check(ms  < 1e-6f, "silu (swish)", ms);
}

static void test_get_rows() {
    Context c(1 << 20);
    const int E = 4, R = 5;
    Tensor* tbl = c.tensor2d(Type::F32, E, R); fill(tbl);
    Tensor* ids = c.tensor1d(Type::I32, 3);
    int32_t idv[3] = {2, 0, -1};   // -1 -> last row (mask-token semantics)
    std::memcpy(ids->data, idv, sizeof(idv));
    Tensor* y = get_rows(c, tbl, ids);
    std::vector<float> ref(E * 3);
    int rows[3] = {2, 0, R-1};
    for (int i = 0; i < 3; i++) for (int e = 0; e < E; e++) ref[i*E+e] = tbl->f32(rows[i]*E+e);
    float md = run_and_max(c, y, ref);
    check(md < 1e-7f, "get_rows (+negative index)", md);
}

static void test_conv2d() {
    Context c(1 << 22);
    // input {IW=4,IH=4,IC=2,N=1}, kernel {KW=3,KH=3,IC=2,OC=2}, stride1 pad1 -> {4,4,2,1}
    const int IW=4,IH=4,IC=2,OC=2,KW=3,KH=3,S=1,P=1;
    Tensor* in  = c.tensor4d(Type::F32, IW,IH,IC,1); fill(in);
    Tensor* ker = c.tensor4d(Type::F32, KW,KH,IC,OC); fill(ker);
    Tensor* y = conv_2d(c, ker, in, S, P);
    const int OW=4,OH=4;
    std::vector<float> ref(OW*OH*OC);
    auto IN = [&](int w,int h,int ic){ return (w<0||w>=IW||h<0||h>=IH)?0.0f:in->f32((ic*IH+h)*IW+w); };
    for (int oc=0;oc<OC;oc++) for (int oh=0;oh<OH;oh++) for (int ow=0;ow<OW;ow++){
        double acc=0;
        for (int ic=0;ic<IC;ic++) for (int kh=0;kh<KH;kh++) for (int kw=0;kw<KW;kw++)
            acc += ker->f32(((oc*IC+ic)*KH+kh)*KW+kw) * IN(ow*S-P+kw, oh*S-P+kh, ic);
        ref[(oc*OH+oh)*OW+ow]=(float)acc;
    }
    float md = run_and_max(c, y, ref);
    check(md < 1e-4f, "conv_2d (direct, pad=1)", md);
}

static void test_group_norm() {
    Context c(1 << 22);
    const int W=2,H=2,Ch=4,N=1,G=2;
    Tensor* a = c.tensor4d(Type::F32, W,H,Ch,N); fill(a);
    Tensor* y = group_norm(c, a, G, 1e-5f);
    std::vector<float> ref(W*H*Ch);
    int cpg=Ch/G;
    for (int g=0; g<G; g++){
        double mean=0; int cnt=W*H*cpg;
        for (int ch=g*cpg; ch<(g+1)*cpg; ch++) for(int h=0;h<H;h++) for(int w=0;w<W;w++) mean+=a->f32((ch*H+h)*W+w);
        mean/=cnt;
        double var=0;
        for (int ch=g*cpg; ch<(g+1)*cpg; ch++) for(int h=0;h<H;h++) for(int w=0;w<W;w++){double t=a->f32((ch*H+h)*W+w)-mean;var+=t*t;}
        var/=cnt; double inv=1.0/std::sqrt(var+1e-5);
        for (int ch=g*cpg; ch<(g+1)*cpg; ch++) for(int h=0;h<H;h++) for(int w=0;w<W;w++) ref[(ch*H+h)*W+w]=(float)((a->f32((ch*H+h)*W+w)-mean)*inv);
    }
    float md = run_and_max(c, y, ref);
    check(md < 1e-5f, "group_norm (GN, no affine)", md);
}

static void test_upscale_permute_cont() {
    Context c(1 << 20);
    // permute then cont: a {2,3} -> permute to {3,2} -> cont -> check transpose
    Tensor* a = c.tensor2d(Type::F32, 2, 3); fill(a);
    Tensor* p = permute(c, a, 1, 0, 2, 3);   // ne becomes {3,2}
    Tensor* cc = cont(c, p);
    { Graph g; g.build_forward(cc); compute(c, g); }
    float md=0;
    for (int i1=0;i1<2;i1++) for (int i0=0;i0<3;i0++)
        md=std::fmax(md, std::fabs(cc->f32(i1*3+i0) - a->f32(i0*2+i1)));
    check(md < 1e-7f, "permute+cont (transpose)", md);

    Tensor* u = upscale(c, c.tensor4d(Type::F32,2,2,1,1), 2);
    fill(u->src[0]);
    { Graph g; g.build_forward(u); compute(c, g); }
    float mu=0;
    for (int oh=0;oh<4;oh++) for(int ow=0;ow<4;ow++)
        mu=std::fmax(mu, std::fabs(u->f32(oh*4+ow) - u->src[0]->f32((oh/2)*2+(ow/2))));
    check(mu < 1e-7f, "upscale (nearest x2)", mu);
}

int main() {
    std::printf("== mg F32 kernel tests ==\n");
    test_mul_mat();
    test_add_bias();
    test_softmax();
    test_norm();
    test_gelu_silu();
    test_get_rows();
    test_conv2d();
    test_group_norm();
    test_upscale_permute_cont();
    std::printf("%s\n", g_fail == 0 ? "ALL PASS" : "FAILURES PRESENT");
    return g_fail == 0 ? 0 : 1;
}
