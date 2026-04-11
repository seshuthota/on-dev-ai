#include <dlfcn.h>
#include <errno.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct Options {
  std::vector<std::string> probeLibs;
  std::string qnnNetRunPath;
  std::string backendLib;
  std::string modelLib;
  std::string inputList;
  std::string outputDir;
  bool help = false;
};

void printUsage(const char* argv0) {
  std::cout << "Usage: " << argv0
            << " --qnn-net-run PATH --backend PATH --model PATH --input-list PATH --output-dir PATH "
               "[--probe-lib PATH ...]\n";
  std::cout << "Example:\n";
  std::cout << "  " << argv0
            << " --probe-lib ./libQnnSystem.so --probe-lib ./libQnnCpu.so "
               "--qnn-net-run ./qnn-net-run --backend ./libQnnCpu.so "
               "--model ./libqnn_model_float.so --input-list ./input_list_float.txt --output-dir ./output\n";
}

bool parseArgs(int argc, char** argv, Options& options, std::string& error) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto requireValue = [&](const std::string& name) -> const char* {
      if (i + 1 >= argc) {
        error = "missing value for " + name;
        return nullptr;
      }
      ++i;
      return argv[i];
    };

    if (arg == "-h" || arg == "--help") {
      options.help = true;
      continue;
    }
    if (arg == "--probe-lib") {
      const char* value = requireValue(arg);
      if (!value) return false;
      options.probeLibs.emplace_back(value);
      continue;
    }
    if (arg == "--qnn-net-run") {
      const char* value = requireValue(arg);
      if (!value) return false;
      options.qnnNetRunPath = value;
      continue;
    }
    if (arg == "--backend") {
      const char* value = requireValue(arg);
      if (!value) return false;
      options.backendLib = value;
      continue;
    }
    if (arg == "--model") {
      const char* value = requireValue(arg);
      if (!value) return false;
      options.modelLib = value;
      continue;
    }
    if (arg == "--input-list") {
      const char* value = requireValue(arg);
      if (!value) return false;
      options.inputList = value;
      continue;
    }
    if (arg == "--output-dir") {
      const char* value = requireValue(arg);
      if (!value) return false;
      options.outputDir = value;
      continue;
    }

    error = "unknown argument: " + arg;
    return false;
  }

  if (options.help) return true;

  if (options.qnnNetRunPath.empty()) {
    error = "--qnn-net-run is required";
    return false;
  }
  if (options.backendLib.empty()) {
    error = "--backend is required";
    return false;
  }
  if (options.modelLib.empty()) {
    error = "--model is required";
    return false;
  }
  if (options.inputList.empty()) {
    error = "--input-list is required";
    return false;
  }
  if (options.outputDir.empty()) {
    error = "--output-dir is required";
    return false;
  }

  return true;
}

bool ensureOutputDir(const std::string& path) {
  if (path.empty()) return false;
  if (::mkdir(path.c_str(), 0755) == 0) return true;
  if (errno == EEXIST) return true;
  std::cerr << "qnn_smoke_error stage=mkdir path=" << path << " errno=" << errno
            << " message=" << strerror(errno) << "\n";
  return false;
}

bool probeLibraries(const std::vector<std::string>& libs, std::vector<void*>& handles) {
  for (const auto& lib : libs) {
    void* handle = ::dlopen(lib.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!handle) {
      const char* err = ::dlerror();
      std::cerr << "qnn_smoke_probe status=fail lib=" << lib
                << " message=" << (err ? err : "unknown") << "\n";
      return false;
    }
    std::cout << "qnn_smoke_probe status=ok lib=" << lib << "\n";
    handles.push_back(handle);
  }
  return true;
}

void closeHandles(std::vector<void*>& handles) {
  for (auto it = handles.rbegin(); it != handles.rend(); ++it) {
    if (*it) {
      ::dlclose(*it);
    }
  }
  handles.clear();
}

int runQnnNetRun(const Options& options, long long& elapsedMs) {
  std::vector<std::string> argStorage = {
      options.qnnNetRunPath,
      "--model",
      options.modelLib,
      "--input_list",
      options.inputList,
      "--backend",
      options.backendLib,
      "--output_dir",
      options.outputDir,
  };

  std::vector<char*> argv;
  argv.reserve(argStorage.size() + 1);
  for (auto& item : argStorage) {
    argv.push_back(item.data());
  }
  argv.push_back(nullptr);

  auto t0 = std::chrono::steady_clock::now();
  pid_t pid = ::fork();
  if (pid < 0) {
    std::cerr << "qnn_smoke_error stage=fork errno=" << errno << " message=" << strerror(errno)
              << "\n";
    return 2;
  }

  if (pid == 0) {
    ::execv(options.qnnNetRunPath.c_str(), argv.data());
    std::cerr << "qnn_smoke_error stage=exec path=" << options.qnnNetRunPath
              << " errno=" << errno << " message=" << strerror(errno) << "\n";
    _exit(127);
  }

  int status = 0;
  while (true) {
    pid_t waited = ::waitpid(pid, &status, 0);
    if (waited == pid) break;
    if (waited < 0 && errno == EINTR) continue;
    if (waited < 0) {
      std::cerr << "qnn_smoke_error stage=waitpid errno=" << errno
                << " message=" << strerror(errno) << "\n";
      return 2;
    }
  }
  auto t1 = std::chrono::steady_clock::now();
  elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

  if (WIFEXITED(status)) {
    return WEXITSTATUS(status);
  }
  if (WIFSIGNALED(status)) {
    return 128 + WTERMSIG(status);
  }
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  std::string parseError;
  if (!parseArgs(argc, argv, options, parseError)) {
    std::cerr << "qnn_smoke_error stage=parse message=" << parseError << "\n";
    printUsage(argv[0]);
    return 1;
  }
  if (options.help) {
    printUsage(argv[0]);
    return 0;
  }

  std::cout << "qnn_smoke_start qnn_net_run=" << options.qnnNetRunPath
            << " backend=" << options.backendLib << " model=" << options.modelLib
            << " input_list=" << options.inputList << " output_dir=" << options.outputDir << "\n";

  if (!ensureOutputDir(options.outputDir)) {
    return 2;
  }

  std::vector<void*> handles;
  if (!probeLibraries(options.probeLibs, handles)) {
    closeHandles(handles);
    return 2;
  }

  long long elapsedMs = 0;
  const int rc = runQnnNetRun(options, elapsedMs);
  closeHandles(handles);

  if (rc == 0) {
    std::cout << "qnn_smoke_result status=ok exit_code=0 inference_ms=" << elapsedMs << "\n";
    return 0;
  }

  std::cerr << "qnn_smoke_result status=fail exit_code=" << rc << " inference_ms=" << elapsedMs
            << "\n";
  return rc;
}
