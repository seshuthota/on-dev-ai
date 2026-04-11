package ai.ondev.snapdragonlab.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.navigation.NavGraph.Companion.findStartDestination
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import ai.ondev.snapdragonlab.MainViewModel

private enum class AppDestination(val route: String, val label: String) {
    Chat("chat", "Chat"),
    Settings("settings", "Settings"),
    Developer("developer", "Developer"),
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SnapdragonLabApp(viewModel: MainViewModel) {
    val navController = rememberNavController()
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()
    val backStackEntry by navController.currentBackStackEntryAsState()
    val currentRoute = backStackEntry?.destination?.route ?: AppDestination.Chat.route

    Scaffold(
        topBar = {
            TopAppBar(title = { Text("Snapdragon Lab") })
        },
        bottomBar = {
            NavigationBar {
                AppDestination.entries.forEach { destination ->
                    NavigationBarItem(
                        selected = currentRoute == destination.route,
                        onClick = {
                            navController.navigate(destination.route) {
                                popUpTo(navController.graph.findStartDestination().id) {
                                    saveState = true
                                }
                                launchSingleTop = true
                                restoreState = true
                            }
                        },
                        icon = {},
                        label = { Text(destination.label) },
                    )
                }
            }
        },
    ) { padding ->
        NavHost(
            navController = navController,
            startDestination = AppDestination.Chat.route,
            modifier = Modifier
                .fillMaxSize()
                .padding(padding),
        ) {
            composable(AppDestination.Chat.route) {
                ScreenColumn {
                    InfoCard(
                        title = "Native runtime",
                        body = uiState.nativeStatus,
                    )
                    InfoCard(
                        title = "Active backend target",
                        body = uiState.backendLabel,
                    )
                    InfoCard(
                        title = "Loaded backend",
                        body = uiState.loadedBackendLabel,
                    )
                    InfoCard(
                        title = "Model path",
                        body = uiState.modelPath,
                    )
                    InfoCard(
                        title = "Path diagnostics",
                        body = uiState.pathDiagnostics,
                    )
                    InfoCard(
                        title = "Model load status",
                        body = uiState.loadStatus,
                    )
                    OutlinedTextField(
                        value = uiState.prompt,
                        onValueChange = viewModel::updatePrompt,
                        modifier = Modifier.fillMaxWidth(),
                        label = { Text("Prompt") },
                    )
                    Button(
                        onClick = viewModel::loadModel,
                        modifier = Modifier.fillMaxWidth(),
                        enabled = !uiState.isBusy,
                    ) {
                        Text("Load hardcoded model")
                    }
                    Button(
                        onClick = viewModel::generate,
                        modifier = Modifier.fillMaxWidth(),
                        enabled = uiState.isModelLoaded && !uiState.isBusy,
                    ) {
                        Text("Generate on selected backend")
                    }
                    Button(
                        onClick = viewModel::stopGeneration,
                        modifier = Modifier.fillMaxWidth(),
                        enabled = uiState.isBusy,
                    ) {
                        Text("Stop generation")
                    }
                    InfoCard(
                        title = "Model target",
                        body = uiState.modelLabel,
                    )
                    InfoCard(
                        title = "Generation status",
                        body = uiState.generationSummary,
                    )
                    InfoCard(
                        title = "Output",
                        body = uiState.output,
                    )
                }
            }

            composable(AppDestination.Settings.route) {
                val isOpenClAvailable = uiState.runtimeInfo.contains("OpenCL")
                val isVulkanAvailable = uiState.runtimeInfo.contains("Vulkan")
                val isQnnBuildEnabled = uiState.isQnnBuildEnabled
                val isQnnPackaged = uiState.runtimeInfo.contains("qnn_status=runtime_packaged")
                ScreenColumn {
                    InfoCard(
                        title = "Backend target",
                        body = uiState.backendLabel,
                    )
                    Button(
                        onClick = { viewModel.selectBackendTarget("cpu") },
                        modifier = Modifier.fillMaxWidth(),
                        enabled = !uiState.isBusy,
                    ) {
                        Text("Use CPU")
                    }
                    Button(
                        onClick = { viewModel.selectBackendTarget("opencl") },
                        modifier = Modifier.fillMaxWidth(),
                        enabled = !uiState.isBusy && isOpenClAvailable,
                    ) {
                        Text(if (isOpenClAvailable) "Use OpenCL" else "Use OpenCL (unavailable)")
                    }
                    Button(
                        onClick = { viewModel.selectBackendTarget("vulkan") },
                        modifier = Modifier.fillMaxWidth(),
                        enabled = !uiState.isBusy && isVulkanAvailable,
                    ) {
                        Text(if (isVulkanAvailable) "Use Vulkan" else "Use Vulkan (unavailable)")
                    }
                    Button(
                        onClick = { viewModel.selectBackendTarget("qnn") },
                        modifier = Modifier.fillMaxWidth(),
                        enabled = !uiState.isBusy && isQnnBuildEnabled,
                    ) {
                        Text(
                            when {
                                !isQnnBuildEnabled -> "Use QNN (disabled at build)"
                                isQnnPackaged -> "Use QNN (experimental)"
                                else -> "Use QNN (runtime incomplete)"
                            },
                        )
                    }
                    InfoCard(
                        title = "Reload state",
                        body = uiState.loadedBackendLabel,
                    )
                    InfoCard(
                        title = "Backend status",
                        body = buildString {
                            append("OpenCL: ")
                            append(if (isOpenClAvailable) "available" else "unavailable or not compiled")
                            append("\n")
                            append("Vulkan: ")
                            append(if (isVulkanAvailable) "available" else "unavailable or not compiled")
                            append("\n")
                            append("QNN: ")
                            append(
                                when {
                                    !isQnnBuildEnabled -> "disabled at build"
                                    isQnnPackaged -> "packaged, experimental load path"
                                    else -> "enabled at build but runtime packaging incomplete"
                                },
                            )
                        },
                    )
                }
            }

            composable(AppDestination.Developer.route) {
                ScreenColumn {
                    InfoCard(
                        title = "Benchmark harness",
                        body = uiState.benchmarkStatus,
                    )
                    Button(
                        onClick = { viewModel.runBenchmark("smoke") },
                        modifier = Modifier.fillMaxWidth(),
                        enabled = uiState.isModelLoaded && !uiState.isBusy,
                    ) {
                        Text("Run smoke benchmark")
                    }
                    Button(
                        onClick = { viewModel.runBenchmark("standard") },
                        modifier = Modifier.fillMaxWidth(),
                        enabled = uiState.isModelLoaded && !uiState.isBusy,
                    ) {
                        Text("Run standard benchmark")
                    }
                    InfoCard(
                        title = "Latest benchmark",
                        body = uiState.latestBenchmark,
                    )
                    InfoCard(
                        title = "Benchmark history",
                        body = uiState.benchmarkHistory,
                    )
                    InfoCard(
                        title = "Runtime info",
                        body = uiState.runtimeInfo,
                    )
                }
            }
        }
    }
}

@Composable
private fun ScreenColumn(content: @Composable () -> Unit) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
        content = { content() },
    )
}

@Composable
private fun InfoCard(
    title: String,
    body: String,
    modifier: Modifier = Modifier,
) {
    Card(
        modifier = modifier.fillMaxWidth(),
    ) {
        Column(
            modifier = Modifier.padding(PaddingValues(16.dp)),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Text(
                text = title,
                style = MaterialTheme.typography.titleMedium,
            )
            Text(
                text = body,
                style = MaterialTheme.typography.bodyMedium,
            )
        }
    }
}
