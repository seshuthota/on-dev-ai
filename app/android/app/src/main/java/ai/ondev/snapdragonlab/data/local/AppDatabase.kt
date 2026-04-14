package ai.ondev.snapdragonlab.data.local

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase
import ai.ondev.snapdragonlab.data.local.entity.ConversationEntity
import ai.ondev.snapdragonlab.data.local.entity.DownloadTaskEntity
import ai.ondev.snapdragonlab.data.local.entity.MessageEntity
import ai.ondev.snapdragonlab.data.local.entity.ModelEntity

@Database(
    entities = [
        ModelEntity::class,
        ConversationEntity::class,
        MessageEntity::class,
        DownloadTaskEntity::class,
    ],
    version = 1,
    exportSchema = false,
)
abstract class AppDatabase : RoomDatabase() {
    abstract fun modelDao(): ModelDao
    abstract fun conversationDao(): ConversationDao
    abstract fun messageDao(): MessageDao
    abstract fun downloadTaskDao(): DownloadTaskDao

    companion object {
        private const val DATABASE_NAME = "ondevai.db"

        fun create(context: Context): AppDatabase =
            Room.databaseBuilder(
                context.applicationContext,
                AppDatabase::class.java,
                DATABASE_NAME,
            ).build()
    }
}
