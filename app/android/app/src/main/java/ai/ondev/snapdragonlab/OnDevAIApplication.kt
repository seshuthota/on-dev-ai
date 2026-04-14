package ai.ondev.snapdragonlab

import android.app.Application

class OnDevAIApplication : Application() {
    lateinit var container: AppContainer
        private set

    override fun onCreate() {
        super.onCreate()
        container = AppContainer(this)
    }
}
