package ai.ondev.snapdragonlab

import android.content.Context
import ai.ondev.snapdragonlab.data.local.AppDatabase
import ai.ondev.snapdragonlab.data.remote.HuggingFaceApi
import ai.ondev.snapdragonlab.data.repository.ChatRepository
import ai.ondev.snapdragonlab.data.repository.DownloadRepository
import ai.ondev.snapdragonlab.data.repository.InferenceRepository
import ai.ondev.snapdragonlab.data.repository.ModelRepository
import ai.ondev.snapdragonlab.data.repository.SettingsRepository

class AppContainer(context: Context) {
    val database: AppDatabase = AppDatabase.create(context)
    val hfApi: HuggingFaceApi = HuggingFaceApi()
    val modelRepository: ModelRepository = ModelRepository(database.modelDao())
    val chatRepository: ChatRepository = ChatRepository(
        database.conversationDao(),
        database.messageDao(),
    )
    val inferenceRepository: InferenceRepository = InferenceRepository()
    val downloadRepository: DownloadRepository = DownloadRepository(
        context = context,
        modelDao = database.modelDao(),
        downloadTaskDao = database.downloadTaskDao(),
        hfApi = hfApi,
    )
    val settingsRepository: SettingsRepository = SettingsRepository(context)
}
