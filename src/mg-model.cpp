// mg-model.cpp — GGUF v3 reader (mmap). Parses metadata + tensor infos and
// exposes weight tensors as zero-copy externals.
#include "mg-model.hpp"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace mg {

namespace {
constexpr uint32_t GGUF_MAGIC = 0x46554747;  // 'GGUF'
// metadata value types
enum { T_U8=0,T_I8,T_U16,T_I16,T_U32,T_I32,T_F32,T_BOOL,T_STRING,T_ARRAY,T_U64,T_I64,T_F64 };
// tensor type codes: GGML F32/F16/I8 (24); MG_I4 (100) is our packed-int4 weight code.
enum { GGML_F32=0, GGML_F16=1, GGML_I8=24, MG_I4=100 };

struct Cursor {
    const uint8_t* p;
    const uint8_t* end;
    template <class T> T read() {
        if (p + sizeof(T) > end) throw std::runtime_error("gguf: unexpected EOF");
        T v; std::memcpy(&v, p, sizeof(T)); p += sizeof(T); return v;
    }
    std::string str() {
        uint64_t n = read<uint64_t>();
        if (p + n > end) throw std::runtime_error("gguf: bad string length");
        std::string s(reinterpret_cast<const char*>(p), n); p += n; return s;
    }
    void skip(size_t n) { if (p + n > end) throw std::runtime_error("gguf: skip past EOF"); p += n; }
};

size_t type_byte_size_mdt(uint32_t t) {
    switch (t) {
        case T_U8: case T_I8: case T_BOOL: return 1;
        case T_U16: case T_I16: return 2;
        case T_U32: case T_I32: case T_F32: return 4;
        case T_U64: case T_I64: case T_F64: return 8;
        default: return 0;
    }
}
} // namespace

struct ModelLoader {
    // Consume one metadata value; capture into hp where the key is recognized.
    static void handle_kv(Cursor& c, const std::string& key, HParams& hp) {
        uint32_t vt = c.read<uint32_t>();
        auto as_i = [&](uint32_t t) -> int64_t {
            switch (t) {
                case T_U32: return c.read<uint32_t>();
                case T_I32: return c.read<int32_t>();
                case T_U64: return (int64_t)c.read<uint64_t>();
                case T_I64: return c.read<int64_t>();
                case T_U8:  return c.read<uint8_t>();
                case T_BOOL:return c.read<uint8_t>();
                default: throw std::runtime_error("gguf: expected int for " + key);
            }
        };
        if (vt == T_STRING) {
            std::string s = c.str();
            if (key == "general.architecture") hp.arch = s;
            else if (key == "general.name") hp.name = s;
            else if (key == "maskgit.quant") hp.quant = s;
        } else if (vt == T_F32) {
            float f = c.read<float>();
            if (key == "maskgit.layernorm_eps") hp.ln_eps = f;
            else if (key == "maskgit.sampling.choice_temperature") hp.choice_temperature = f;
        } else if (vt == T_ARRAY) {
            uint32_t sub = c.read<uint32_t>();
            uint64_t n = c.read<uint64_t>();
            if (key == "maskgit.vqgan.channel_multipliers" && (sub == T_I32 || sub == T_U32)) {
                hp.vq_channel_mult.clear();
                for (uint64_t i = 0; i < n; i++) hp.vq_channel_mult.push_back(c.read<int32_t>());
            } else {
                size_t es = type_byte_size_mdt(sub);
                if (es == 0) throw std::runtime_error("gguf: unsupported array subtype");
                c.skip(es * n);
            }
        } else {  // integer-ish scalar
            int64_t v = as_i(vt);
            if      (key == "maskgit.n_layer")        hp.n_layer = (int)v;
            else if (key == "maskgit.n_head")         hp.n_head = (int)v;
            else if (key == "maskgit.n_embd")         hp.n_embd = (int)v;
            else if (key == "maskgit.n_ffn")          hp.n_ffn = (int)v;
            else if (key == "maskgit.head_dim")       hp.head_dim = (int)v;
            else if (key == "maskgit.vocab_size")     hp.vocab_size = (int)v;
            else if (key == "maskgit.n_tokens")       hp.n_tokens = (int)v;
            else if (key == "maskgit.n_positions")    hp.n_positions = (int)v;
            else if (key == "maskgit.mask_token_id")  hp.mask_token_id = (int)v;
            else if (key == "maskgit.resolution")     hp.resolution = (int)v;
            else if (key == "maskgit.vqgan.codebook_size")    hp.vq_codebook_size = (int)v;
            else if (key == "maskgit.vqgan.embedding_dim")    hp.vq_embedding_dim = (int)v;
            else if (key == "maskgit.vqgan.filters")          hp.vq_filters = (int)v;
            else if (key == "maskgit.vqgan.num_res_blocks")   hp.vq_num_res_blocks = (int)v;
            else if (key == "maskgit.vqgan.group_norm_groups")hp.vq_gn_groups = (int)v;
            // general.alignment handled by caller via a separate pass; ignore here
        }
    }
};

namespace {
// peek alignment without disturbing hp (defaults 32)
uint32_t read_alignment(const uint8_t* base, size_t size, uint64_t n_kv) {
    Cursor c{base + 24, base + size};   // after magic/version/n_tensors/n_kv (8+8+8)
    for (uint64_t i = 0; i < n_kv; i++) {
        std::string key = c.str();
        uint32_t vt = c.read<uint32_t>();
        if (key == "general.alignment" && vt == T_U32) return c.read<uint32_t>();
        // skip this value
        if (vt == T_STRING) { c.str(); }
        else if (vt == T_ARRAY) {
            uint32_t sub = c.read<uint32_t>(); uint64_t n = c.read<uint64_t>();
            if (sub == T_STRING) { for (uint64_t j=0;j<n;j++) c.str(); }
            else c.skip(type_byte_size_mdt(sub) * n);
        } else {
            c.skip(type_byte_size_mdt(vt));
        }
    }
    return 32;
}
} // namespace

Tensor* Model::get(const std::string& n) const {
    auto it = tensors_.find(n);
    return it == tensors_.end() ? nullptr : it->second;
}
Tensor* Model::require(const std::string& n) const {
    Tensor* t = get(n);
    if (!t) throw std::runtime_error("gguf: missing required tensor '" + n + "'");
    return t;
}

std::unique_ptr<Model> Model::load(const std::string& path, Context& ctx) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("cannot open " + path);
    struct stat st{};
    if (::fstat(fd, &st) != 0) { ::close(fd); throw std::runtime_error("fstat failed"); }
    size_t size = (size_t)st.st_size;
    void* base = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) { ::close(fd); throw std::runtime_error("mmap failed"); }

    const uint8_t* b = static_cast<const uint8_t*>(base);
    Cursor c{b, b + size};
    if (c.read<uint32_t>() != GGUF_MAGIC) { ::munmap(base, size); ::close(fd); throw std::runtime_error("not a GGUF file"); }
    uint32_t version = c.read<uint32_t>();
    if (version != 3) { ::munmap(base, size); ::close(fd); throw std::runtime_error("unsupported GGUF version"); }
    uint64_t n_tensors = c.read<uint64_t>();
    uint64_t n_kv = c.read<uint64_t>();

    auto m = std::unique_ptr<Model>(new Model());
    m->mmap_base_ = base; m->mmap_size_ = size; m->fd_ = fd;

    uint32_t alignment = read_alignment(b, size, n_kv);

    // metadata
    for (uint64_t i = 0; i < n_kv; i++) {
        std::string key = c.str();
        ModelLoader::handle_kv(c, key, m->hp_);
    }

    // tensor infos
    struct TInfo { std::string name; int n_dims; int64_t ne[MAX_DIMS]; uint32_t type; uint64_t off; };
    std::vector<TInfo> infos(n_tensors);
    for (uint64_t i = 0; i < n_tensors; i++) {
        TInfo& ti = infos[i];
        ti.name = c.str();
        ti.n_dims = (int)c.read<uint32_t>();
        for (int d = 0; d < MAX_DIMS; d++) ti.ne[d] = 1;
        for (int d = 0; d < ti.n_dims; d++) ti.ne[d] = (int64_t)c.read<uint64_t>();
        ti.type = c.read<uint32_t>();
        ti.off = c.read<uint64_t>();
        if (ti.type != GGML_F32 && ti.type != GGML_I8 && ti.type != MG_I4) {
            ::munmap(base, size); ::close(fd);
            throw std::runtime_error("unsupported tensor type on '" + ti.name + "'");
        }
    }

    // data section start: align current cursor to `alignment`
    size_t pos = (size_t)(c.p - b);
    size_t data_start = (pos + alignment - 1) / alignment * alignment;

    for (const TInfo& ti : infos) {
        void* dptr = const_cast<uint8_t*>(b) + data_start + ti.off;
        Type mt = ti.type == GGML_I8 ? Type::I8 : (ti.type == MG_I4 ? Type::I4 : Type::F32);
        Tensor* t = ctx.external(mt, ti.n_dims, ti.ne, dptr);
        t->name = ti.name;
        m->tensors_[ti.name] = t;
    }
    return m;
}

Model::~Model() {
    if (mmap_base_) ::munmap(mmap_base_, mmap_size_);
    if (fd_ >= 0) ::close(fd_);
}

} // namespace mg
