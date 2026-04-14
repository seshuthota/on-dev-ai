package ai.ondev.snapdragonlab.ui

import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ChatBubble
import androidx.compose.material.icons.filled.Code
import androidx.compose.material.icons.filled.Explore
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.outlined.ChatBubbleOutline
import androidx.compose.material.icons.outlined.Code
import androidx.compose.material.icons.outlined.Explore
import androidx.compose.material.icons.outlined.Settings
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavGraph.Companion.findStartDestination
import androidx.navigation.NavType
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import androidx.navigation.navArgument
import ai.ondev.snapdragonlab.AppContainer
import ai.ondev.snapdragonlab.ui.browser.ModelBrowserScreen
import ai.ondev.snapdragonlab.ui.browser.ModelBrowserViewModel
import ai.ondev.snapdragonlab.ui.chat.ChatScreen
import ai.ondev.snapdragonlab.ui.chat.ChatViewModel
import ai.ondev.snapdragonlab.ui.conversations.ConversationsScreen
import ai.ondev.snapdragonlab.ui.conversations.ConversationsViewModel
import ai.ondev.snapdragonlab.ui.developer.DeveloperScreen
import ai.ondev.snapdragonlab.ui.developer.DeveloperViewModel
import ai.ondev.snapdragonlab.ui.settings.SettingsScreen
import ai.ondev.snapdragonlab.ui.settings.SettingsViewModel

private enum class TopDestination(
    val route: String,
    val label: String,
    val selectedIcon: ImageVector,
    val unselectedIcon: ImageVector,
) {
    Conversations("conversations", "Chats", Icons.Filled.ChatBubble, Icons.Outlined.ChatBubbleOutline),
    Browser("browser", "Models", Icons.Filled.Explore, Icons.Outlined.Explore),
    Settings("settings", "Settings", Icons.Filled.Settings, Icons.Outlined.Settings),
    Developer("developer", "Dev", Icons.Filled.Code, Icons.Outlined.Code),
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SnapdragonLabApp(
    container: AppContainer,
    benchmarkHistoryPath: String,
) {
    val navController = rememberNavController()
    val backStackEntry by navController.currentBackStackEntryAsState()
    val currentRoute = backStackEntry?.destination?.route

    // Determine if we're on a top-level vs nested screen
    val isTopLevel = TopDestination.entries.any { it.route == currentRoute }

    Scaffold(
        topBar = {
            if (isTopLevel) {
                TopAppBar(
                    title = {
                        Text(
                            text = "OnDevAI",
                            style = MaterialTheme.typography.titleLarge,
                        )
                    },
                    colors = TopAppBarDefaults.topAppBarColors(
                        containerColor = MaterialTheme.colorScheme.surface,
                    ),
                )
            }
        },
        bottomBar = {
            if (isTopLevel) {
                NavigationBar {
                    TopDestination.entries.forEach { destination ->
                        val selected = currentRoute == destination.route
                        NavigationBarItem(
                            selected = selected,
                            onClick = {
                                navController.navigate(destination.route) {
                                    popUpTo(navController.graph.findStartDestination().id) {
                                        saveState = true
                                    }
                                    launchSingleTop = true
                                    restoreState = true
                                }
                            },
                            icon = {
                                Icon(
                                    imageVector = if (selected) destination.selectedIcon else destination.unselectedIcon,
                                    contentDescription = destination.label,
                                )
                            },
                            label = { Text(destination.label) },
                        )
                    }
                }
            }
        },
    ) { padding ->
        NavHost(
            navController = navController,
            startDestination = TopDestination.Conversations.route,
            modifier = Modifier
                .fillMaxSize()
                .padding(padding),
        ) {
            composable(TopDestination.Conversations.route) {
                val vm: ConversationsViewModel = viewModel(
                    factory = ConversationsViewModel.Factory(container.chatRepository),
                )
                ConversationsScreen(
                    viewModel = vm,
                    onNavigateToChat = { conversationId ->
                        navController.navigate("chat/$conversationId")
                    },
                )
            }

            composable(
                route = "chat/{conversationId}",
                arguments = listOf(navArgument("conversationId") { type = NavType.StringType }),
            ) { backStack ->
                val conversationId = backStack.arguments?.getString("conversationId") ?: return@composable
                val vm: ChatViewModel = viewModel(
                    key = conversationId,
                    factory = ChatViewModel.Factory(
                        conversationId = conversationId,
                        chatRepository = container.chatRepository,
                        modelRepository = container.modelRepository,
                        inferenceRepository = container.inferenceRepository,
                        settingsRepository = container.settingsRepository,
                    ),
                )
                ChatScreen(
                    viewModel = vm,
                    onNavigateBack = { navController.popBackStack() },
                )
            }

            composable(TopDestination.Browser.route) {
                val vm: ModelBrowserViewModel = viewModel(
                    factory = ModelBrowserViewModel.Factory(
                        hfApi = container.hfApi,
                        downloadRepository = container.downloadRepository,
                        modelRepository = container.modelRepository,
                    ),
                )
                ModelBrowserScreen(viewModel = vm)
            }

            composable(TopDestination.Settings.route) {
                val vm: SettingsViewModel = viewModel(
                    factory = SettingsViewModel.Factory(
                        inferenceRepository = container.inferenceRepository,
                        settingsRepository = container.settingsRepository,
                    ),
                )
                SettingsScreen(viewModel = vm)
            }

            composable(TopDestination.Developer.route) {
                val vm: DeveloperViewModel = viewModel(
                    factory = DeveloperViewModel.Factory(benchmarkHistoryPath),
                )
                DeveloperScreen(viewModel = vm)
            }
        }
    }
}
