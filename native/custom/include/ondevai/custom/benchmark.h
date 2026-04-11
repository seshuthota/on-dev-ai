#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ondevai::custom {

struct ThermalSample {
    std::int64_t elapsed_ms = 0;
    int thermal_status = -1;
    float thermal_headroom = -1.0F;
};

struct BenchmarkResult {
    std::string backend;
    std::string model_id;
    std::string model_checksum;
    std::string prompt_id;
    std::int64_t load_time_ms = 0;
    std::int64_t ttft_ms = 0;
    double prefill_tok_per_sec = 0.0;
    double decode_tok_per_sec = 0.0;
    std::int64_t elapsed_ms = 0;
    std::vector<ThermalSample> thermal_samples;
};

}  // namespace ondevai::custom
