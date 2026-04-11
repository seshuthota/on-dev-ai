#include <jni.h>

#include "app_engine.h"

#if ONDEVAI_ENABLE_QNN
#include "qnn_runner.h"
#endif

extern "C" JNIEXPORT jstring JNICALL
Java_ai_ondev_snapdragonlab_NativeBridge_nativeInitializeImpl(
        JNIEnv* env,
        jobject /* this */,
        jstring jnative_lib_dir) {
    const char* native_lib_dir = env->GetStringUTFChars(jnative_lib_dir, nullptr);
    const std::string message = ondevai::AppEngine::Instance().Initialize(native_lib_dir);
    env->ReleaseStringUTFChars(jnative_lib_dir, native_lib_dir);
    return env->NewStringUTF(message.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_ai_ondev_snapdragonlab_NativeBridge_nativeLoadModelImpl(
        JNIEnv* env,
        jobject /* this */,
        jstring jmodel_path,
        jint context_size,
        jint thread_count,
        jstring jbackend_target) {
    const char* model_path = env->GetStringUTFChars(jmodel_path, nullptr);
    const char* backend_target = env->GetStringUTFChars(jbackend_target, nullptr);
    const std::string message = ondevai::AppEngine::Instance().LoadModel(
            model_path,
            context_size,
            thread_count,
            backend_target);
    env->ReleaseStringUTFChars(jmodel_path, model_path);
    env->ReleaseStringUTFChars(jbackend_target, backend_target);
    return env->NewStringUTF(message.c_str());
}

extern "C" JNIEXPORT void JNICALL
Java_ai_ondev_snapdragonlab_NativeBridge_nativeGenerateStreamImpl(
        JNIEnv* env,
        jobject thiz,
        jstring jprompt,
        jint max_tokens) {
    jclass bridge_class = env->GetObjectClass(thiz);
    jmethodID on_token = env->GetStaticMethodID(
            bridge_class,
            "handleNativeToken",
            "(Ljava/lang/String;)V");
    jmethodID on_complete = env->GetStaticMethodID(
            bridge_class,
            "handleNativeComplete",
            "(Ljava/lang/String;)V");
    jmethodID on_error = env->GetStaticMethodID(
            bridge_class,
            "handleNativeError",
            "(Ljava/lang/String;)V");

    const char* prompt = env->GetStringUTFChars(jprompt, nullptr);
    const ondevai::GenerationResult result = ondevai::AppEngine::Instance().GenerateStreaming(
            prompt,
            max_tokens,
            [env, bridge_class, on_token](const std::string& piece) {
                jstring jpiece = env->NewStringUTF(piece.c_str());
                env->CallStaticVoidMethod(bridge_class, on_token, jpiece);
                env->DeleteLocalRef(jpiece);
            });
    env->ReleaseStringUTFChars(jprompt, prompt);
    if (result.success) {
        jstring jsummary = env->NewStringUTF(result.summary.c_str());
        env->CallStaticVoidMethod(bridge_class, on_complete, jsummary);
        env->DeleteLocalRef(jsummary);
    } else {
        jstring jerror = env->NewStringUTF(result.error.c_str());
        env->CallStaticVoidMethod(bridge_class, on_error, jerror);
        env->DeleteLocalRef(jerror);
    }
    env->DeleteLocalRef(bridge_class);
}

extern "C" JNIEXPORT void JNICALL
Java_ai_ondev_snapdragonlab_NativeBridge_nativeStopGenerationImpl(
        JNIEnv* /* env */,
        jobject /* this */) {
    ondevai::AppEngine::Instance().RequestStop();
}

extern "C" JNIEXPORT jstring JNICALL
Java_ai_ondev_snapdragonlab_NativeBridge_nativeRunBenchmarkImpl(
        JNIEnv* env,
        jobject /* this */,
        jstring jmode) {
    const char* mode = env->GetStringUTFChars(jmode, nullptr);
    const std::string result = ondevai::AppEngine::Instance().RunBenchmark(mode);
    env->ReleaseStringUTFChars(jmode, mode);
    return env->NewStringUTF(result.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_ai_ondev_snapdragonlab_NativeBridge_nativeGetRuntimeInfoImpl(
        JNIEnv* env,
        jobject /* this */) {
    const std::string message = ondevai::AppEngine::Instance().GetRuntimeInfo();
    return env->NewStringUTF(message.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_ai_ondev_snapdragonlab_NativeBridge_nativeProbeOpenClImpl(
        JNIEnv* env,
        jobject /* this */) {
    const std::string message = ondevai::AppEngine::Instance().ProbeOpenCl();
    return env->NewStringUTF(message.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_ai_ondev_snapdragonlab_NativeBridge_nativeSetOpenClGpuLayersImpl(
        JNIEnv* env,
        jobject /* this */,
        jint gpu_layers) {
    const std::string message = ondevai::AppEngine::Instance().SetOpenClGpuLayersOverride(gpu_layers);
    return env->NewStringUTF(message.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_ai_ondev_snapdragonlab_NativeBridge_nativeRunQnnControlSmokeImpl(
        JNIEnv* env,
        jobject /* this */,
        jstring jbackend_lib,
        jstring jmodel_lib,
        jstring joutput_dir) {
#if ONDEVAI_ENABLE_QNN
    const char* backend_lib = env->GetStringUTFChars(jbackend_lib, nullptr);
    const char* model_lib = env->GetStringUTFChars(jmodel_lib, nullptr);
    const char* output_dir = env->GetStringUTFChars(joutput_dir, nullptr);

    const auto result = ondevai::qnn::RunModel(backend_lib, model_lib, output_dir);

    env->ReleaseStringUTFChars(jbackend_lib, backend_lib);
    env->ReleaseStringUTFChars(jmodel_lib, model_lib);
    env->ReleaseStringUTFChars(joutput_dir, output_dir);

    std::string status = "qnn_control_smoke"
        " | success=" + std::string(result.success ? "true" : "false") +
        " | total_ms=" + std::to_string(result.total_ms) +
        " | backend_load_ms=" + std::to_string(result.backend_load_ms) +
        " | model_load_ms=" + std::to_string(result.model_load_ms) +
        " | graph_compose_ms=" + std::to_string(result.graph_compose_ms) +
        " | graph_finalize_ms=" + std::to_string(result.graph_finalize_ms) +
        " | " + result.details;
    if (!result.error.empty()) {
        status += " | error=" + result.error;
    }
    return env->NewStringUTF(status.c_str());
#else
    (void)jbackend_lib;
    (void)jmodel_lib;
    (void)joutput_dir;
    return env->NewStringUTF("qnn_control_smoke | error=QNN support not compiled (ONDEVAI_ENABLE_QNN=0)");
#endif
}
