package ai.ondev.snapdragonlab.data.repository

import android.content.Context
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import ai.ondev.snapdragonlab.domain.model.BackendTarget
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

private val Context.dataStore: DataStore<Preferences> by preferencesDataStore(name = "settings")

class SettingsRepository(private val context: Context) {

    private val systemPromptKey = stringPreferencesKey("system_prompt")
    private val backendTargetKey = stringPreferencesKey("backend_target")

    /**
     * Flow emitting the configured system prompt. 
     * If blank or not set, emits the default prompt.
     */
    val systemPromptFlow: Flow<String> = context.dataStore.data.map { preferences ->
        val prompt = preferences[systemPromptKey]
        if (prompt.isNullOrBlank()) DEFAULT_SYSTEM_PROMPT else prompt
    }

    /**
     * Flow emitting the configured backend target (CPU, OpenCL, QNN, etc).
     */
    val backendTargetFlow: Flow<BackendTarget> = context.dataStore.data.map { preferences ->
        preferences[backendTargetKey]?.let { backendName ->
            try {
                BackendTarget.valueOf(backendName)
            } catch (e: Exception) {
                BackendTarget.CPU
            }
        } ?: BackendTarget.CPU
    }

    suspend fun setSystemPrompt(prompt: String) {
        context.dataStore.edit { preferences ->
            preferences[systemPromptKey] = prompt
        }
    }

    suspend fun setBackendTarget(target: BackendTarget) {
        context.dataStore.edit { preferences ->
            preferences[backendTargetKey] = target.name
        }
    }

    companion object {
        const val DEFAULT_SYSTEM_PROMPT = "You are a helpful, smart, kind, and efficient AI assistant. You always fulfill the user's requests to the best of your ability."
    }
}
