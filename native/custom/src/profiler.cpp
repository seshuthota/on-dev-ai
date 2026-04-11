#include "ondevai/custom/runtime.h"

#include <algorithm>
#include <chrono>
#include <cstring>

namespace ondevai::custom {

Profiler& Profiler::instance() {
    static Profiler inst;
    return inst;
}

void Profiler::begin_section(const char* name) {
    if (!enabled_) return;
    if (section_count_ >= 16) return;
    auto& sec = sections_[section_count_++];
    sec.name = name;
    sec.elapsed_ns = 0;
    sec.start = std::chrono::steady_clock::now();
    sec.active = true;
}

void Profiler::end_section(const char* name) {
    if (!enabled_) return;
    auto end = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < section_count_; ++i) {
        auto& sec = sections_[i];
        if (sec.active && sec.name == name) {
            sec.elapsed_ns += static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(end - sec.start).count());
            sec.active = false;
            // Map section name to timing accumulator and add elapsed time
            std::uint64_t* acc = nullptr;
            if (std::strcmp(name, "rmsnorm_input") == 0) acc = &rmsnorm_input_ns_;
            else if (std::strcmp(name, "matvec_qkv") == 0) acc = &matvec_qkv_ns_;
            else if (std::strcmp(name, "rope_q") == 0) acc = &rope_q_ns_;
            else if (std::strcmp(name, "rope_k") == 0) acc = &rope_k_ns_;
            else if (std::strcmp(name, "attention") == 0) acc = &attention_ns_;
            else if (std::strcmp(name, "matvec_o") == 0) acc = &matvec_o_ns_;
            else if (std::strcmp(name, "rmsnorm_post") == 0) acc = &rmsnorm_post_ns_;
            else if (std::strcmp(name, "matvec_mlp") == 0) acc = &matvec_mlp_ns_;
            else if (std::strcmp(name, "total_layer") == 0) acc = &total_layer_ns_;
            if (acc) *acc += sec.elapsed_ns;
            return;
        }
    }
}

void Profiler::reset() {
    section_count_ = 0;
    rmsnorm_input_ns_ = 0;
    matvec_qkv_ns_ = 0;
    rope_q_ns_ = 0;
    rope_k_ns_ = 0;
    attention_ns_ = 0;
    matvec_o_ns_ = 0;
    rmsnorm_post_ns_ = 0;
    matvec_mlp_ns_ = 0;
    total_layer_ns_ = 0;
    for (auto& sec : sections_) {
        sec = SectionTiming{};
    }
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
