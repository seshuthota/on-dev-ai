package ai.ondev.snapdragonlab.ui.browser

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.expandVertically
import androidx.compose.animation.shrinkVertically
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CloudDownload
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.ExpandLess
import androidx.compose.material.icons.filled.ExpandMore
import androidx.compose.material.icons.filled.Search
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import ai.ondev.snapdragonlab.data.remote.model.HfModelSearchResult
import ai.ondev.snapdragonlab.domain.model.DownloadState
import ai.ondev.snapdragonlab.domain.model.ModelSummary

@Composable
fun ModelBrowserScreen(viewModel: ModelBrowserViewModel) {
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()
    val existingModels by viewModel.existingModels.collectAsStateWithLifecycle()

    Column(modifier = Modifier.fillMaxSize()) {
        // Search bar
        OutlinedTextField(
            value = uiState.searchQuery,
            onValueChange = viewModel::updateSearchQuery,
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp, vertical = 8.dp),
            placeholder = { Text("Search GGUF models on Hugging Face...") },
            leadingIcon = {
                Icon(
                    Icons.Default.Search,
                    contentDescription = null,
                    tint = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            },
            trailingIcon = {
                if (uiState.searchQuery.isNotBlank()) {
                    IconButton(onClick = { viewModel.updateSearchQuery("") }) {
                        Icon(Icons.Default.Close, contentDescription = "Clear search")
                    }
                }
            },
            singleLine = true,
            colors = OutlinedTextFieldDefaults.colors(
                focusedContainerColor = MaterialTheme.colorScheme.surfaceContainerHigh,
                unfocusedContainerColor = MaterialTheme.colorScheme.surfaceContainerHigh,
            ),
        )

        // Error banner
        if (uiState.searchError != null) {
            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp, vertical = 4.dp),
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.errorContainer,
                ),
            ) {
                Row(
                    modifier = Modifier.padding(12.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Icon(
                        Icons.Default.Warning,
                        contentDescription = null,
                        tint = MaterialTheme.colorScheme.onErrorContainer,
                        modifier = Modifier.size(18.dp),
                    )
                    Spacer(Modifier.width(8.dp))
                    Text(
                        text = uiState.searchError ?: "",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onErrorContainer,
                        modifier = Modifier.weight(1f),
                    )
                    IconButton(onClick = viewModel::clearError) {
                        Icon(
                            Icons.Default.Close,
                            contentDescription = "Dismiss",
                            modifier = Modifier.size(16.dp),
                        )
                    }
                }
            }
        }

        // Search results
        when {
            uiState.isSearching -> {
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(32.dp),
                    contentAlignment = Alignment.TopCenter,
                ) {
                    CircularProgressIndicator(
                        modifier = Modifier.padding(top = 48.dp),
                        color = MaterialTheme.colorScheme.secondary,
                    )
                }
            }
            uiState.searchQuery.isBlank() -> {
                EmptyBrowserState(
                    existingModels = existingModels,
                    onDeleteModel = viewModel::deleteModel,
                )
            }
            uiState.searchResults.isEmpty() && !uiState.isSearching -> {
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(32.dp),
                    contentAlignment = Alignment.TopCenter,
                ) {
                    Text(
                        text = "No GGUF models found for \"${uiState.searchQuery}\"",
                        style = MaterialTheme.typography.bodyLarge,
                        color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f),
                        modifier = Modifier.padding(top = 48.dp),
                    )
                }
            }
            else -> {
                LazyColumn(
                    contentPadding = PaddingValues(horizontal = 16.dp, vertical = 8.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    items(uiState.searchResults, key = { it.repoId }) { model ->
                        ModelRepoCard(
                            model = model,
                            isExpanded = uiState.expandedRepo?.repoId == model.repoId,
                            repoDetail = if (uiState.expandedRepo?.repoId == model.repoId) uiState.expandedRepo else null,
                            downloadStates = uiState.downloadStates,
                            existingModels = existingModels,
                            onExpand = { viewModel.expandRepo(model.repoId) },
                            onDownload = { file -> viewModel.requestDownload(model.repoId, file) },
                        )
                    }
                }
            }
        }
    }

    // Large file warning dialog
    uiState.showLargeFileWarning?.let { file ->
        AlertDialog(
            onDismissRequest = viewModel::dismissLargeFileWarning,
            icon = { Icon(Icons.Default.Warning, contentDescription = null, tint = MaterialTheme.colorScheme.error) },
            title = { Text("Large file download") },
            text = {
                Text(
                    "\"${file.filename}\" is ${file.sizeLabel}.\n\n" +
                        "This may take a long time and use significant storage. " +
                        "Make sure you have enough free space on your device.",
                )
            },
            confirmButton = {
                TextButton(onClick = viewModel::confirmLargeFileDownload) {
                    Text("Download anyway")
                }
            },
            dismissButton = {
                TextButton(onClick = viewModel::dismissLargeFileWarning) {
                    Text("Cancel")
                }
            },
        )
    }
}

@Composable
private fun EmptyBrowserState(
    existingModels: List<ModelSummary>,
    onDeleteModel: (String) -> Unit,
) {
    var modelToDelete by remember { mutableStateOf<ModelSummary?>(null) }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(32.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Spacer(Modifier.height(32.dp))
        Icon(
            Icons.Default.Search,
            contentDescription = null,
            modifier = Modifier.size(56.dp),
            tint = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.3f),
        )
        Spacer(Modifier.height(12.dp))
        Text(
            text = "Search for GGUF models",
            style = MaterialTheme.typography.titleMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Spacer(Modifier.height(4.dp))
        Text(
            text = "Try: \"llama 3\", \"qwen2\", \"phi-3\", \"gemma\"",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f),
        )

        if (existingModels.isNotEmpty()) {
            Spacer(Modifier.height(24.dp))
            Text(
                text = "Downloaded models",
                style = MaterialTheme.typography.titleSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Spacer(Modifier.height(8.dp))
            existingModels.forEach { model ->
                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(vertical = 2.dp),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.surfaceContainerHigh,
                    ),
                ) {
                    Row(
                        modifier = Modifier.padding(12.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Icon(
                            Icons.Default.CheckCircle,
                            contentDescription = null,
                            modifier = Modifier.size(18.dp),
                            tint = MaterialTheme.colorScheme.secondary,
                        )
                        Spacer(Modifier.width(8.dp))
                        Column(modifier = Modifier.weight(1f)) {
                            Text(
                                text = model.displayName,
                                style = MaterialTheme.typography.bodyMedium,
                            )
                            Text(
                                text = "${model.sizeLabel} • ${model.status.name}",
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.outline,
                            )
                        }
                        IconButton(onClick = { modelToDelete = model }, modifier = Modifier.size(32.dp)) {
                            Icon(
                                Icons.Default.Close,
                                contentDescription = "Delete model",
                                tint = MaterialTheme.colorScheme.onSurfaceVariant,
                                modifier = Modifier.size(18.dp),
                            )
                        }
                    }
                }
            }
        }
    }

    modelToDelete?.let { summary ->
        AlertDialog(
            onDismissRequest = { modelToDelete = null },
            icon = { Icon(Icons.Default.Warning, contentDescription = null, tint = MaterialTheme.colorScheme.error) },
            title = { Text("Delete Model?") },
            text = { Text("Are you sure you want to delete ${summary.displayName}?\nThis will remove the file and free up ${summary.sizeLabel} of space.") },
            confirmButton = {
                TextButton(onClick = {
                    onDeleteModel(summary.id)
                    modelToDelete = null
                }) {
                    Text("Delete", color = MaterialTheme.colorScheme.error)
                }
            },
            dismissButton = {
                TextButton(onClick = { modelToDelete = null }) {
                    Text("Cancel")
                }
            },
        )
    }
}

@Composable
private fun ModelRepoCard(
    model: HfModelSearchResult,
    isExpanded: Boolean,
    repoDetail: RepoDetail?,
    downloadStates: Map<String, DownloadState>,
    existingModels: List<ModelSummary>,
    onExpand: () -> Unit,
    onDownload: (GgufFileInfo) -> Unit,
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceContainerHigh,
        ),
    ) {
        Column {
            // Repo header
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clickable(onClick = onExpand)
                    .padding(16.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = model.repoId,
                        style = MaterialTheme.typography.titleSmall,
                        fontWeight = FontWeight.SemiBold,
                        color = MaterialTheme.colorScheme.primary,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                    )
                    Spacer(Modifier.height(2.dp))
                    Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                        StatLabel(
                            label = "↓ ${formatNumber(model.downloads)}",
                        )
                        StatLabel(
                            label = "♥ ${formatNumber(model.likes)}",
                        )
                    }
                }
                Icon(
                    imageVector = if (isExpanded) Icons.Default.ExpandLess else Icons.Default.ExpandMore,
                    contentDescription = if (isExpanded) "Collapse" else "Expand",
                    tint = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }

            // Expanded file list
            AnimatedVisibility(
                visible = isExpanded,
                enter = expandVertically(),
                exit = shrinkVertically(),
            ) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(start = 16.dp, end = 16.dp, bottom = 12.dp),
                ) {
                    if (repoDetail?.isLoading == true) {
                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(vertical = 16.dp),
                            contentAlignment = Alignment.Center,
                        ) {
                            CircularProgressIndicator(
                                modifier = Modifier.size(24.dp),
                                strokeWidth = 2.dp,
                            )
                        }
                    } else if (repoDetail?.ggufFiles?.isEmpty() == true) {
                        Text(
                            text = "No GGUF files found in this repo",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            modifier = Modifier.padding(vertical = 8.dp),
                        )
                    } else {
                        repoDetail?.ggufFiles?.forEach { file ->
                            GgufFileRow(
                                file = file,
                                repoId = model.repoId,
                                downloadStates = downloadStates,
                                existingModels = existingModels,
                                onDownload = { onDownload(file) },
                            )
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun GgufFileRow(
    file: GgufFileInfo,
    repoId: String,
    downloadStates: Map<String, DownloadState>,
    existingModels: List<ModelSummary>,
    onDownload: () -> Unit,
) {
    // Check if this file is already downloaded
    val existingModel = existingModels.find {
        it.hfRepoId == repoId && it.filename == file.filename
    }
    val downloadState = existingModel?.let { downloadStates[it.id] }
    val isDownloaded = existingModel?.status == ai.ondev.snapdragonlab.domain.model.ModelStatus.READY

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 3.dp),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceContainer,
        ),
    ) {
        Column(modifier = Modifier.padding(12.dp)) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = file.filename.substringAfterLast("/"),
                        style = MaterialTheme.typography.bodySmall,
                        fontWeight = FontWeight.Medium,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                    )
                    Row(
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                    ) {
                        Text(
                            text = file.sizeLabel,
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.outline,
                        )
                        file.quantSuffix?.let { quant ->
                            QuantBadge(quant = quant)
                        }
                        if (file.isLargeFile) {
                            Text(
                                text = "⚠ Large",
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.error,
                            )
                        }
                    }
                }

                when {
                    isDownloaded -> {
                        Icon(
                            Icons.Default.CheckCircle,
                            contentDescription = "Downloaded",
                            tint = MaterialTheme.colorScheme.secondary,
                            modifier = Modifier.size(28.dp),
                        )
                    }
                    downloadState is DownloadState.Downloading -> {
                        CircularProgressIndicator(
                            progress = { downloadState.progressFraction },
                            modifier = Modifier.size(28.dp),
                            strokeWidth = 3.dp,
                            color = MaterialTheme.colorScheme.primary,
                        )
                    }
                    downloadState is DownloadState.Queued -> {
                        CircularProgressIndicator(
                            modifier = Modifier.size(28.dp),
                            strokeWidth = 2.dp,
                            color = MaterialTheme.colorScheme.outline,
                        )
                    }
                    else -> {
                        IconButton(onClick = onDownload, modifier = Modifier.size(36.dp)) {
                            Icon(
                                Icons.Default.CloudDownload,
                                contentDescription = "Download",
                                tint = MaterialTheme.colorScheme.primary,
                            )
                        }
                    }
                }
            }

            // Progress bar for active downloads
            if (downloadState is DownloadState.Downloading) {
                Spacer(Modifier.height(6.dp))
                LinearProgressIndicator(
                    progress = { downloadState.progressFraction },
                    modifier = Modifier.fillMaxWidth(),
                    color = MaterialTheme.colorScheme.primary,
                    trackColor = MaterialTheme.colorScheme.surfaceContainerHighest,
                )
                Text(
                    text = "${ModelSummary.formatBytes(downloadState.progressBytes)} / ${ModelSummary.formatBytes(downloadState.totalBytes)}",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.outline,
                    modifier = Modifier.padding(top = 2.dp),
                )
            }

            if (downloadState is DownloadState.Failed) {
                Spacer(Modifier.height(4.dp))
                Text(
                    text = "Download failed: ${downloadState.error}",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.error,
                )
            }
        }
    }
}

@Composable
private fun QuantBadge(quant: String) {
    Card(
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.tertiaryContainer,
        ),
    ) {
        Text(
            text = quant,
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onTertiaryContainer,
            fontWeight = FontWeight.Medium,
            modifier = Modifier.padding(horizontal = 6.dp, vertical = 1.dp),
        )
    }
}

@Composable
private fun StatLabel(label: String) {
    Text(
        text = label,
        style = MaterialTheme.typography.labelSmall,
        color = MaterialTheme.colorScheme.outline,
    )
}

private fun formatNumber(n: Int): String = when {
    n >= 1_000_000 -> "%.1fM".format(n / 1_000_000.0)
    n >= 1_000 -> "%.1fK".format(n / 1_000.0)
    else -> n.toString()
}
