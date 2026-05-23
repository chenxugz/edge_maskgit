// mg-model.hpp — mmap-based GGUF loader for the MaskGIT runtime.
#pragma once

#include "mg-tensor.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace mg {

struct HParams {
    // transformer
    int n_layer = 0, n_head = 0, n_embd = 0, n_ffn = 0, head_dim = 0;
    int vocab_size = 0, n_tokens = 0, n_positions = 0;
    int mask_token_id = -1;
    float ln_eps = 1e-12f;
    // vqgan
    int vq_codebook_size = 0, vq_embedding_dim = 0, vq_filters = 0;
    int vq_num_res_blocks = 0, vq_gn_groups = 32;
    std::vector<int> vq_channel_mult;
    // sampling
    float choice_temperature = 4.5f;
    int resolution = 256;
    std::string arch, name;
};

// Owns the mmap and exposes weight tensors (as externals in the provided Context).
class Model {
public:
    ~Model();
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    // Loads `path`, creating external Tensors inside `ctx` (no weight copy).
    static std::unique_ptr<Model> load(const std::string& path, Context& ctx);

    const HParams& hparams() const { return hp_; }
    bool     has(const std::string& name) const { return tensors_.count(name) > 0; }
    Tensor*  get(const std::string& name) const;       // nullptr if missing
    Tensor*  require(const std::string& name) const;   // throws if missing
    size_t   n_tensors() const { return tensors_.size(); }
    const std::unordered_map<std::string, Tensor*>& tensors() const { return tensors_; }

private:
    Model() = default;
    HParams hp_;
    std::unordered_map<std::string, Tensor*> tensors_;
    void*  mmap_base_ = nullptr;
    size_t mmap_size_ = 0;
    int    fd_ = -1;
    friend struct ModelLoader;
};

} // namespace mg
