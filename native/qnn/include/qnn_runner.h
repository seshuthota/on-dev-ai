#pragma once

#include <string>

namespace ondevai {
namespace qnn {

struct RunResult {
    bool success = false;
    long long backend_load_ms = 0;
    long long model_load_ms = 0;
    long long graph_compose_ms = 0;
    long long graph_finalize_ms = 0;
    long long total_ms = 0;
    std::string backend_build_id;
    std::string error;
    std::string details;
};

/// Run a complete QNN model load + graph compose + finalize in-process.
///
/// This is the in-process replacement for the subprocess-based qnn_smoke / qnn-net-run path.
/// It uses dlopen to load the QNN backend and model shared libraries, then drives
/// the QNN API directly: backend → context → composeGraphs → finalizeGraphs.
///
/// @param backend_lib  Backend library soname or path (e.g. "libQnnCpu.so" or "libQnnHtp.so")
/// @param model_lib    Absolute path to the compiled model .so
/// @param output_dir   Directory for any output artifacts (created if needed)
RunResult RunModel(
    const std::string& backend_lib,
    const std::string& model_lib,
    const std::string& output_dir);

}  // namespace qnn
}  // namespace ondevai
