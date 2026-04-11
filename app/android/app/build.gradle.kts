import org.jetbrains.kotlin.gradle.dsl.JvmTarget
import org.gradle.api.GradleException
import org.gradle.api.tasks.Sync
import java.io.File

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

val repoRootDir = layout.projectDirectory.dir("../../../")
val repoLocalQairtSdkRoot = repoRootDir.dir("v2.45.0.260326/qairt/2.45.0.260326").asFile
val externalSiblingQairtSdkRoot =
    repoRootDir.asFile.resolve("../OnDevAI_external/v2.45.0.260326/qairt/2.45.0.260326").canonicalFile
val defaultQairtSdkRootPath =
    System.getenv("QAIRT_SDK_ROOT")
        ?: System.getenv("QNN_SDK_ROOT")
        ?: if (externalSiblingQairtSdkRoot.isDirectory) {
            externalSiblingQairtSdkRoot.absolutePath
        } else {
            repoLocalQairtSdkRoot.absolutePath
        }
val qairtSdkRoot = file(
    providers.gradleProperty("ondevai.qairtSdkRoot")
        .orElse(defaultQairtSdkRootPath)
        .get(),
)
val ondevaiEnableQnn = providers.gradleProperty("ondevai.enableQnn")
    .map { value ->
        when (value.trim().lowercase()) {
            "1", "true", "yes", "on" -> true
            "0", "false", "no", "off" -> false
            else -> throw GradleException("Invalid ondevai.enableQnn value '$value' (expected true/false)")
        }
    }
    .orElse(false)
    .get()

val qnnRuntimeLibs = listOf(
    "libQnnSystem.so",
    "libQnnCpu.so",
    "libQnnGpu.so",
    "libQnnHtp.so",
    "libQnnHtpPrepare.so",
    "libQnnHtpV79Stub.so",
    "libQnnHtpV81Stub.so",
    "libGenie.so",
    "libQnnGenAiTransformer.so",
)

val qnnHexagonSkels = listOf(
    "hexagon-v79" to "libQnnHtpV79Skel.so",
    "hexagon-v81" to "libQnnHtpV81Skel.so",
)

val generatedQnnRoot = layout.buildDirectory.dir("generated/ondevai/qnn")
val generatedQnnJniLibs = generatedQnnRoot.map { it.dir("jniLibs") }
val generatedQnnAssets = generatedQnnRoot.map { it.dir("assets") }
val generatedQnnControlRoot = layout.buildDirectory.dir("generated/ondevai/qnn-control")
val generatedQnnControlAssets = generatedQnnControlRoot.map { it.dir("assets") }

fun detectAndroidNdkRoot(): String {
    val direct = System.getenv("ANDROID_NDK_ROOT")
    if (!direct.isNullOrBlank() && file(direct).isDirectory) {
        return direct
    }

    val sdkRoot = System.getenv("ANDROID_SDK_ROOT") ?: System.getenv("ANDROID_HOME")
    if (!sdkRoot.isNullOrBlank()) {
        val ndkDir = file("$sdkRoot/ndk")
        if (ndkDir.isDirectory) {
            return ndkDir.listFiles()
                ?.filter { it.isDirectory }
                ?.maxByOrNull { it.name }
                ?.absolutePath
                ?: throw GradleException("No Android NDK versions found under $ndkDir")
        }
    }

    val homeNdkDir = file("${System.getProperty("user.home")}/Android/Sdk/ndk")
    if (homeNdkDir.isDirectory) {
        return homeNdkDir.listFiles()
            ?.filter { it.isDirectory }
            ?.maxByOrNull { it.name }
            ?.absolutePath
            ?: throw GradleException("No Android NDK versions found under $homeNdkDir")
    }

    throw GradleException("Could not detect Android NDK root")
}

fun runCommand(
    command: List<String>,
    workingDir: File,
    environment: Map<String, String> = emptyMap(),
) {
    val process =
        ProcessBuilder(command)
            .directory(workingDir)
            .apply {
                environment().putAll(environment)
                redirectErrorStream(true)
            }
            .start()

    val output = process.inputStream.bufferedReader().use { it.readText() }
    val exitCode = process.waitFor()
    if (exitCode != 0) {
        throw GradleException(
            buildString {
                append("Command failed (exit ")
                append(exitCode)
                append("): ")
                append(command.joinToString(" "))
                if (output.isNotBlank()) {
                    append("\n")
                    append(output.trim())
                }
            },
        )
    }
}

val qnnControlAssetsRoot = generatedQnnControlAssets.map { it.dir("qnn/control") }
val qnnControlBuildDir = layout.buildDirectory.dir("generated/ondevai/qnn/control-build")

val stageQnnControlAssets by tasks.registering {
    onlyIf { ondevaiEnableQnn }
    outputs.dir(qnnControlAssetsRoot)

    doLast {
        if (!qairtSdkRoot.isDirectory) {
            throw GradleException("QAIRT/QNN SDK root does not exist: ${qairtSdkRoot.absolutePath}")
        }

        val ndkRoot = detectAndroidNdkRoot()
        val ndkBinDir = file("$ndkRoot/toolchains/llvm/prebuilt/linux-x86_64/bin")
        if (!ndkBinDir.isDirectory) {
            throw GradleException("NDK LLVM toolchain bin dir missing: ${ndkBinDir.absolutePath}")
        }

        val controlRoot = qnnControlAssetsRoot.get().asFile
        delete(controlRoot)
        controlRoot.mkdirs()

        // Build sample model shared libraries for in-process QNN runner.
        // The in-process runner dlopen's these directly — no qnn_smoke or qnn-net-run
        // binaries needed (subprocess exec is blocked by Android app sandbox).
        val modelLibOut = qnnControlBuildDir.get().dir("model-libs").asFile
        delete(modelLibOut)
        mkdir(modelLibOut)

        listOf("qnn_model_float", "qnn_model_8bit_quantized").forEach { modelName ->
            val command = """
                set -euo pipefail
                set +u
                source '${qairtSdkRoot.absolutePath}/bin/envsetup.sh'
                set -u
                '${qairtSdkRoot.absolutePath}/bin/x86_64-linux-clang/qnn-model-lib-generator' \
                  -c '${qairtSdkRoot.absolutePath}/examples/QNN/converter/models/${modelName}.cpp' \
                  -b '${qairtSdkRoot.absolutePath}/examples/QNN/converter/models/${modelName}.bin' \
                  -t aarch64-android \
                  -l '${modelName}' \
                  -o '${modelLibOut.absolutePath}'
            """.trimIndent()
            runCommand(
                command = listOf("bash", "-lc", command),
                workingDir = repoRootDir.asFile,
                environment =
                    mapOf(
                        "PATH" to "${ndkBinDir.absolutePath}:${ndkRoot}:${System.getenv("PATH").orEmpty()}",
                    ),
            )
        }

        copy {
            from(modelLibOut.resolve("aarch64-android/libqnn_model_float.so"))
            from(modelLibOut.resolve("aarch64-android/libqnn_model_8bit_quantized.so"))
            into(controlRoot)
        }
    }
}

val stageQnnRuntimeArtifacts by tasks.registering(Sync::class) {
    onlyIf { ondevaiEnableQnn }
    into(generatedQnnRoot)

    into("jniLibs/arm64-v8a") {
        from(qnnRuntimeLibs.map { libName -> qairtSdkRoot.resolve("lib/aarch64-android/$libName") })
    }
    qnnHexagonSkels.forEach { (hexagonDir, skelName) ->
        into("assets/qnn/$hexagonDir") {
            from(qairtSdkRoot.resolve("lib/$hexagonDir/unsigned/$skelName"))
        }
    }

    doFirst {
        if (!qairtSdkRoot.isDirectory) {
            throw GradleException(
                "QAIRT/QNN SDK root does not exist: ${qairtSdkRoot.absolutePath}. " +
                    "Override with -Pondevai.qairtSdkRoot=/abs/path/to/sdk",
            )
        }

        val requiredPaths = buildList {
            qnnRuntimeLibs.forEach { add(qairtSdkRoot.resolve("lib/aarch64-android/$it")) }
            qnnHexagonSkels.forEach { (hexagonDir, skelName) ->
                add(qairtSdkRoot.resolve("lib/$hexagonDir/unsigned/$skelName"))
            }
        }
        val missing = requiredPaths.filterNot { it.isFile }
        if (missing.isNotEmpty()) {
            val details = missing.joinToString(separator = "\n") { "- ${it.absolutePath}" }
            throw GradleException("Missing required QAIRT/QNN runtime artifacts:\n$details")
        }
    }
}

android {
    namespace = "ai.ondev.snapdragonlab"
    compileSdk = 35
    ndkVersion = "27.3.13750724"

    defaultConfig {
        applicationId = "ai.ondev.snapdragonlab"
        minSdk = 31
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0"
        buildConfigField("boolean", "ONDEVAI_ENABLE_QNN", ondevaiEnableQnn.toString())

        externalNativeBuild {
            cmake {
                cppFlags += "-std=c++20"
                // Always build native code with full optimization — even in debug APKs.
                // Without this, assembleDebug uses -O0 which makes inference ~10x slower.
                arguments += "-DCMAKE_BUILD_TYPE=Release"
                arguments += "-DONDEVAI_ENABLE_QNN=${if (ondevaiEnableQnn) "ON" else "OFF"}"
                if (ondevaiEnableQnn) {
                    arguments += "-DQAIRT_SDK_ROOT=${qairtSdkRoot.absolutePath}"
                }
            }
        }

        ndk {
            abiFilters += "arm64-v8a"
        }
    }

    buildTypes {
        debug {
            isMinifyEnabled = false
        }

        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro",
            )
        }
    }

    buildFeatures {
        buildConfig = true
        compose = true
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    externalNativeBuild {
        cmake {
            path = file("../../../native/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    sourceSets {
        getByName("main") {
            if (ondevaiEnableQnn) {
                jniLibs.srcDir(generatedQnnJniLibs)
                assets.srcDir(generatedQnnAssets)
                assets.srcDir(generatedQnnControlAssets)
            }
        }
    }

    packaging {
        jniLibs {
            excludes += "**/libOpenCL.so"
        }
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
    }
}

if (ondevaiEnableQnn) {
    tasks.named("preBuild").configure {
        dependsOn(stageQnnRuntimeArtifacts)
        dependsOn(stageQnnControlAssets)
    }
}

kotlin {
    compilerOptions {
        jvmTarget = JvmTarget.JVM_17
    }
}

dependencies {
    val composeBom = platform("androidx.compose:compose-bom:2024.09.00")

    implementation("androidx.core:core-ktx:1.15.0")
    implementation("androidx.activity:activity-compose:1.10.0")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.8.7")
    implementation("androidx.lifecycle:lifecycle-viewmodel-ktx:2.8.7")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.8.7")
    implementation("androidx.navigation:navigation-compose:2.8.5")
    implementation("androidx.work:work-runtime-ktx:2.10.1")
    implementation("com.google.android.material:material:1.12.0")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0")

    implementation(composeBom)
    androidTestImplementation(composeBom)

    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.compose.material3:material3")

    debugImplementation("androidx.compose.ui:ui-tooling")
    debugImplementation("androidx.compose.ui:ui-test-manifest")

    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.6.1")
    androidTestImplementation("androidx.compose.ui:ui-test-junit4")
}
