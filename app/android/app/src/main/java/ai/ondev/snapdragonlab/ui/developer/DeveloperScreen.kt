package ai.ondev.snapdragonlab.ui.developer

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle

@Composable
fun DeveloperScreen(viewModel: DeveloperViewModel) {
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text(
            text = "Developer Tools",
            style = MaterialTheme.typography.titleLarge,
            color = MaterialTheme.colorScheme.onSurface,
        )

        DeveloperInfoCard(title = "Benchmark Status", body = uiState.benchmarkStatus)

        Button(
            onClick = { viewModel.runBenchmark("smoke") },
            modifier = Modifier.fillMaxWidth(),
            enabled = !uiState.isBusy,
        ) {
            Text("Run smoke benchmark")
        }

        Button(
            onClick = { viewModel.runBenchmark("standard") },
            modifier = Modifier.fillMaxWidth(),
            enabled = !uiState.isBusy,
        ) {
            Text("Run standard benchmark")
        }

        DeveloperInfoCard(title = "Latest Benchmark", body = uiState.latestBenchmark)
        DeveloperInfoCard(title = "Benchmark History", body = uiState.benchmarkHistory)
        DeveloperInfoCard(title = "Runtime Info", body = uiState.runtimeInfo.ifBlank { "Not initialized" })
    }
}

@Composable
private fun DeveloperInfoCard(
    title: String,
    body: String,
    modifier: Modifier = Modifier,
) {
    Card(
        modifier = modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceContainerHigh,
        ),
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Text(
                text = title,
                style = MaterialTheme.typography.titleMedium,
                color = MaterialTheme.colorScheme.onSurface,
            )
            Text(
                text = body,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}
