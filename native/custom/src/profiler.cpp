#include "ondevai/custom/runtime.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <optional>

namespace ondevai::custom {

namespace {

std::optional<Profiler::SectionId> section_from_name(const char* name) {
    if (name == nullptr) return std::nullopt;
    if (std::strcmp(name, "rmsnorm_input") == 0) return Profiler::SectionId::rmsnorm_input;
    if (std::strcmp(name, "matvec_qkv") == 0) return Profiler::SectionId::matvec_qkv;
    if (std::strcmp(name, "rope_q") == 0) return Profiler::SectionId::rope_q;
    if (std::strcmp(name, "rope_k") == 0) return Profiler::SectionId::rope_k;
    if (std::strcmp(name, "attention") == 0) return Profiler::SectionId::attention;
    if (std::strcmp(name, "matvec_o") == 0) return Profiler::SectionId::matvec_o;
    if (std::strcmp(name, "rmsnorm_post") == 0) return Profiler::SectionId::rmsnorm_post;
    if (std::strcmp(name, "matvec_mlp") == 0) return Profiler::SectionId::matvec_mlp;
    if (std::strcmp(name, "total_layer") == 0) return Profiler::SectionId::total_layer;
    return std::nullopt;
}

std::size_t section_index(Profiler::SectionId id) {
    return static_cast<std::size_t>(id);
}

}  // namespace

Profiler& Profiler::instance() {
    static Profiler inst;
    return inst;
}

void Profiler::begin_section(const char* name) {
    if (!enabled_) return;
    const auto id = section_from_name(name);
    if (!id.has_value()) return;
    const auto idx = section_index(*id);
    active_[idx] = true;
    start_[idx] = std::chrono::steady_clock::now();
}

void Profiler::end_section(const char* name) {
    if (!enabled_) return;
    const auto id = section_from_name(name);
    if (!id.has_value()) return;
    const auto idx = section_index(*id);
    if (!active_[idx]) return;

    auto end = std::chrono::steady_clock::now();
    const auto elapsed_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start_[idx]).count());
    active_[idx] = false;

    switch (*id) {
        case SectionId::rmsnorm_input:
            rmsnorm_input_ns_ += elapsed_ns;
            break;
        case SectionId::matvec_qkv:
            matvec_qkv_ns_ += elapsed_ns;
            break;
        case SectionId::rope_q:
            rope_q_ns_ += elapsed_ns;
            break;
        case SectionId::rope_k:
            rope_k_ns_ += elapsed_ns;
            break;
        case SectionId::attention:
            attention_ns_ += elapsed_ns;
            break;
        case SectionId::matvec_o:
            matvec_o_ns_ += elapsed_ns;
            break;
        case SectionId::rmsnorm_post:
            rmsnorm_post_ns_ += elapsed_ns;
            break;
        case SectionId::matvec_mlp:
            matvec_mlp_ns_ += elapsed_ns;
            break;
        case SectionId::total_layer:
            total_layer_ns_ += elapsed_ns;
            break;
        case SectionId::count:
            break;
    }
}

void Profiler::reset() {
    active_.fill(false);
    rmsnorm_input_ns_ = 0;
    matvec_qkv_ns_ = 0;
    rope_q_ns_ = 0;
    rope_k_ns_ = 0;
    attention_ns_ = 0;
    matvec_o_ns_ = 0;
    rmsnorm_post_ns_ = 0;
    matvec_mlp_ns_ = 0;
    total_layer_ns_ = 0;
}

ProfilerResult Profiler::result() const {
    ProfilerResult r;
    r.rmsnorm_input_us = rmsnorm_input_ns_ / 1000;
    r.matvec_qkv_us = matvec_qkv_ns_ / 1000;
    r.rope_q_us = rope_q_ns_ / 1000;
    r.rope_k_us = rope_k_ns_ / 1000;
    r.attention_us = attention_ns_ / 1000;
    r.matvec_o_us = matvec_o_ns_ / 1000;
    r.rmsnorm_post_us = rmsnorm_post_ns_ / 1000;
    r.matvec_mlp_us = matvec_mlp_ns_ / 1000;
    r.total_layer_us = total_layer_ns_ / 1000;
    return r;
}

}  // namespace ondevai::custom
