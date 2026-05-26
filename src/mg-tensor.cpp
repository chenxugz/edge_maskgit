// mg-tensor.cpp — Context arena, tensor/view construction, op-node builders, graph build.
#include "mg-tensor.hpp"

#include <cassert>
#include <cstring>
#include <stdexcept>

namespace mg {

static constexpr size_t ALIGN = 32;

size_t type_size(Type t) {
    switch (t) {
        case Type::F32: return 4;
        case Type::F16: return 2;
        case Type::I32: return 4;
        case Type::I8:  return 1;
        case Type::I4:  return 1;   // packed; nbytes() handles the 2-per-byte case
    }
    return 0;
}

// ---- Tensor ----
int64_t Tensor::nelements() const {
    int64_t n = 1;
    for (int i = 0; i < MAX_DIMS; i++) n *= ne[i];
    return n;
}
size_t Tensor::nbytes() const {
    if (type == Type::I4) return static_cast<size_t>(nelements() + 1) / 2;   // 2 elems/byte
    if (type == Type::Q8_0) return (static_cast<size_t>(nelements()) / Q8_0_BLOCK) * Q8_0_BYTES;
    if (type == Type::Q4_K) return (static_cast<size_t>(nelements()) / Q4_K_BLOCK) * Q4_K_BYTES;
    return static_cast<size_t>(nelements()) * type_size(type);
}

bool Tensor::is_contiguous() const {
    size_t exp = type_size(type);
    for (int i = 0; i < MAX_DIMS; i++) {
        if (ne[i] == 0) return true;
        if (nb[i] != exp) return false;
        exp *= ne[i];
    }
    return true;
}

static void set_contig_strides(Tensor* t) {
    t->nb[0] = type_size(t->type);
    for (int i = 1; i < MAX_DIMS; i++) t->nb[i] = t->nb[i - 1] * t->ne[i - 1];
}

// ---- Context ----
Context::Context(size_t arena_bytes)
    : arena_(new uint8_t[arena_bytes]), arena_size_(arena_bytes) {}

void Context::reset() { nodes_.clear(); off_ = 0; }

void* Context::alloc(size_t bytes) {
    off_ = (off_ + ALIGN - 1) & ~(ALIGN - 1);
    if (off_ + bytes > arena_size_)
        throw std::runtime_error("mg::Context arena out of memory (increase arena_bytes)");
    void* p = arena_.get() + off_;
    off_ += bytes;
    return p;
}

Tensor* Context::make(Type t, int n_dims, const int64_t ne[MAX_DIMS], Op op,
                      Tensor* s0, Tensor* s1, bool alloc_data) {
    nodes_.push_back(std::make_unique<Tensor>());
    Tensor* x = nodes_.back().get();
    x->type = t;
    x->n_dims = n_dims;
    for (int i = 0; i < MAX_DIMS; i++) x->ne[i] = (i < n_dims) ? ne[i] : 1;
    set_contig_strides(x);
    x->op = op;
    x->src[0] = s0;
    x->src[1] = s1;
    if (alloc_data) x->data = alloc(x->nbytes());
    return x;
}

Tensor* Context::tensor(Type t, int n_dims, const int64_t ne[MAX_DIMS]) {
    return make(t, n_dims, ne, Op::None, nullptr, nullptr, /*alloc_data=*/true);
}
Tensor* Context::tensor1d(Type t, int64_t n0) { int64_t ne[4]={n0,1,1,1}; return tensor(t,1,ne); }
Tensor* Context::tensor2d(Type t, int64_t n0, int64_t n1) { int64_t ne[4]={n0,n1,1,1}; return tensor(t,2,ne); }
Tensor* Context::tensor3d(Type t, int64_t n0, int64_t n1, int64_t n2) { int64_t ne[4]={n0,n1,n2,1}; return tensor(t,3,ne); }
Tensor* Context::tensor4d(Type t, int64_t n0, int64_t n1, int64_t n2, int64_t n3) { int64_t ne[4]={n0,n1,n2,n3}; return tensor(t,4,ne); }

Tensor* Context::external(Type t, int n_dims, const int64_t ne[MAX_DIMS], void* data) {
    Tensor* x = make(t, n_dims, ne, Op::None, nullptr, nullptr, /*alloc_data=*/false);
    x->data = data;
    x->is_view = true;
    return x;
}

// ---- elementwise ----
Tensor* add(Context& c, Tensor* a, Tensor* b) { return c.make(a->type, a->n_dims, a->ne, Op::Add, a, b, true); }
Tensor* mul(Context& c, Tensor* a, Tensor* b) { return c.make(a->type, a->n_dims, a->ne, Op::Mul, a, b, true); }
Tensor* add_bias(Context& c, Tensor* a, Tensor* bias) { return c.make(a->type, a->n_dims, a->ne, Op::Add, a, bias, true); }
Tensor* scale(Context& c, Tensor* a, float s) {
    Tensor* t = c.make(a->type, a->n_dims, a->ne, Op::Scale, a, nullptr, true);
    t->fparam[0] = s;
    return t;
}

Tensor* mul_mat(Context& c, Tensor* w, Tensor* x) {
    assert(w->ne[0] == x->ne[0] && "mul_mat inner dim mismatch");
    int64_t ne[4] = { w->ne[1], x->ne[1], x->ne[2], x->ne[3] };
    int nd = (ne[3] > 1) ? 4 : (ne[2] > 1 ? 3 : 2);
    return c.make(Type::F32, nd, ne, Op::MulMat, w, x, true);
}

Tensor* mul_mat_ex(Context& c, Tensor* w, Tensor* x, Tensor* bias, int act, Tensor* residual) {
    Tensor* y = mul_mat(c, w, x);
    y->src[2] = bias;          // row vector ne={N}, or null
    y->src[3] = residual;      // same shape as y, or null
    y->iparam[0] = act;        // 0=none, 1=gelu, 2=silu
    return y;
}

Tensor* get_rows(Context& c, Tensor* a, Tensor* ids) {
    int64_t ne[4] = { a->ne[0], ids->ne[0], ids->ne[1], 1 };
    int nd = (ne[2] > 1) ? 3 : 2;
    return c.make(a->type, nd, ne, Op::GetRows, a, ids, true);
}

Tensor* soft_max(Context& c, Tensor* a, float s) {
    Tensor* t = c.make(a->type, a->n_dims, a->ne, Op::SoftMax, a, nullptr, true);
    t->fparam[0] = s;
    return t;
}
Tensor* norm(Context& c, Tensor* a, float eps) {
    Tensor* t = c.make(a->type, a->n_dims, a->ne, Op::Norm, a, nullptr, true);
    t->fparam[0] = eps;
    return t;
}
Tensor* group_norm(Context& c, Tensor* a, int groups, float eps) {
    Tensor* t = c.make(a->type, a->n_dims, a->ne, Op::GroupNorm, a, nullptr, true);
    t->iparam[0] = groups;
    t->fparam[1] = eps;
    return t;
}
Tensor* gelu(Context& c, Tensor* a) { return c.make(a->type, a->n_dims, a->ne, Op::Gelu, a, nullptr, true); }
Tensor* silu(Context& c, Tensor* a) { return c.make(a->type, a->n_dims, a->ne, Op::Silu, a, nullptr, true); }

Tensor* conv_2d(Context& c, Tensor* kernel, Tensor* input, int stride, int pad) {
    int64_t KW = kernel->ne[0], KH = kernel->ne[1], OC = kernel->ne[3];
    int64_t IW = input->ne[0], IH = input->ne[1], N = input->ne[3];
    int64_t OW = (IW + 2 * pad - KW) / stride + 1;
    int64_t OH = (IH + 2 * pad - KH) / stride + 1;
    int64_t ne[4] = { OW, OH, OC, N };
    Tensor* t = c.make(Type::F32, 4, ne, Op::Conv2D, kernel, input, true);
    t->iparam[0] = stride;
    t->iparam[1] = pad;
    return t;
}

Tensor* upscale(Context& c, Tensor* a, int factor) {
    int64_t ne[4] = { a->ne[0] * factor, a->ne[1] * factor, a->ne[2], a->ne[3] };
    Tensor* t = c.make(a->type, 4, ne, Op::Upscale, a, nullptr, true);
    t->iparam[0] = factor;
    return t;
}

// ---- views ----
static Tensor* make_view(Context& c, Tensor* a, Op op) {
    Tensor* t = c.make(a->type, a->n_dims, a->ne, op, a, nullptr, /*alloc_data=*/false);
    t->data = a->data;     // share storage
    t->is_view = true;
    return t;
}

Tensor* reshape(Context& c, Tensor* a, std::vector<int64_t> ne) {
    assert(a->is_contiguous() && "reshape requires contiguous src");
    Tensor* t = make_view(c, a, Op::Reshape);
    t->n_dims = static_cast<int>(ne.size());
    for (int i = 0; i < MAX_DIMS; i++) t->ne[i] = (i < t->n_dims) ? ne[i] : 1;
    set_contig_strides(t);
    assert(t->nelements() == a->nelements() && "reshape element count mismatch");
    return t;
}

Tensor* permute(Context& c, Tensor* a, int a0, int a1, int a2, int a3) {
    int ax[4] = {a0, a1, a2, a3};
    Tensor* t = make_view(c, a, Op::Permute);
    t->n_dims = a->n_dims;
    for (int i = 0; i < MAX_DIMS; i++) { t->ne[ax[i]] = a->ne[i]; t->nb[ax[i]] = a->nb[i]; }
    return t;
}

Tensor* cont(Context& c, Tensor* a) {
    return c.make(a->type, a->n_dims, a->ne, Op::Cont, a, nullptr, true);
}

// ---- graph build (post-order DFS -> topo order) ----
static bool seen(const std::vector<Tensor*>& v, Tensor* t) {
    for (Tensor* n : v) if (n == t) return true;
    return false;
}
static void build_rec(std::vector<Tensor*>& nodes, Tensor* t) {
    if (!t || t->op == Op::None) return;   // leaf (weight/input)
    if (seen(nodes, t)) return;
    for (Tensor* s : t->src) build_rec(nodes, s);
    nodes.push_back(t);
}
void Graph::build_forward(Tensor* out) { nodes.clear(); build_rec(nodes, out); }

} // namespace mg
