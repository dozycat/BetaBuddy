package com.betabuddy.android.gym

import android.app.Application
import android.content.Context
import android.content.MutableContextWrapper
import android.content.res.AssetManager
import android.os.SystemClock
import android.util.Log
import android.webkit.WebView
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.core.content.edit
import androidx.lifecycle.AndroidViewModel
import com.betabuddy.android.BuildConfig
import java.io.IOException

/** Reads the pack from `assets/gympack/`. */
class AssetGymPackFiles(private val assets: AssetManager) : GymPackFiles {
    private fun full(path: String) = GymPack.ASSET_DIRECTORY + "/" + path

    override fun readText(path: String): String? = try {
        assets.open(full(path)).use { it.readBytes().toString(Charsets.UTF_8) }
    } catch (error: IOException) {
        null
    }

    // Opening a directory or a missing path throws; only a regular file opens.
    override fun exists(path: String): Boolean = try {
        assets.open(full(path)).close()
        true
    } catch (error: IOException) {
        false
    }
}

/** Why the page is not on screen. The screen turns these into text in the current language. */
sealed interface GymFailure {
    /** The page itself said it cannot go on, before it was ready. The text is the page's, possibly empty. */
    data class Page(val message: String) : GymFailure
    /** The document did not load. */
    data class Load(val description: String) : GymFailure
    /** The system took the page's renderer process away, usually for memory. */
    data object RendererGone : GymFailure
}

/** Debug-only launch options, for putting the gym in a given state from `adb shell am start`. */
data class GymDebugOptions(
    val query: String? = null,
    val openRoute: String? = null,
    val openWall: String? = null,
)

/**
 * Owns the one gym web view. It outlives the screen and the activity (it sits in [GymViewModel]),
 * so switching tabs, opening settings or a configuration change keeps the scene and the selection.
 * How the page is served and kept offline is [PackWebView]'s business.
 */
class GymSession(private val app: Application) {
    sealed interface Phase {
        data object Idle : Phase
        data object Loading : Phase
        data object Ready : Phase
        data class Failed(val failure: GymFailure) : Phase
    }

    val status: GymPackStatus = GymPack.load(AssetGymPackFiles(app.assets))

    /** The pack in this build, or null when there is none or it cannot be used. */
    val pack: GymPack? get() = (status as? GymPackStatus.Ready)?.pack

    var phase: Phase by mutableStateOf<Phase>(Phase.Idle)
        private set
    var webView: WebView? by mutableStateOf<WebView?>(null)
        private set
    private var host: PackWebView? = null

    /** Last route picked in the page. Kept in memory only, for linking a recorded climb to a route (id and name). */
    var selectedRoute: GymRoute? by mutableStateOf<GymRoute?>(null)
        private set

    private val preferences = app.getSharedPreferences("gym", Context.MODE_PRIVATE)

    /** Whether the one-time notice before first entry has been read. */
    var noticeSeen: Boolean by mutableStateOf(preferences.getBoolean(KEY_NOTICE_SEEN, false))
        private set

    /** Set when the page asks for the app's settings (`openSettings`); the shell opens them and clears it. */
    var settingsRequested: Boolean by mutableStateOf(false)
        private set

    var debug: GymDebugOptions = GymDebugOptions()

    // The web view is created against this wrapper, so it can be moved to a new activity after a
    // configuration change and never holds on to a finished one.
    private val viewContext = MutableContextWrapper(app)

    /** Set by the page's `ready`. Before it, a page error means the scene never came up; after it, errors are the page's own to show. */
    private var pageReady = false
    private var documentFailed = false
    private var loadStarted = 0L
    private var loadedLanguage: String? = null
    private var routeToRestore: String? = null
    private var active = false

    fun acknowledgeNotice() {
        noticeSeen = true
        preferences.edit { putBoolean(KEY_NOTICE_SEEN, true) }
    }

    fun consumeSettingsRequest() {
        settingsRequested = false
    }

    /** The screen is in front of this activity: the web view may use it for anything that needs a window. */
    fun attach(context: Context) {
        viewContext.baseContext = context
    }

    fun detach(context: Context) {
        if (viewContext.baseContext === context) viewContext.baseContext = app
    }

    /**
     * Opens the page if it is not open, and reloads it when the interface language has changed
     * since it was loaded (the page takes its language from the URL), going back to the selected route.
     */
    fun start(language: String) {
        val pack = pack ?: return
        val host = host
        when {
            host == null && phase == Phase.Idle -> open(pack, language)
            host != null && phase !is Phase.Failed && loadedLanguage != language -> {
                routeToRestore = selectedRoute?.id
                load(host, pack, language)
            }
        }
    }

    /** The reload button on the failure card. */
    fun reload(language: String) {
        val pack = pack ?: return
        val host = host
        if (host == null) open(pack, language) else load(host, pack, language)
    }

    fun send(command: GymCommand) {
        if (phase == Phase.Ready) host?.evaluate(command.script)
    }

    /** Pauses the page's animation while the gym is not what the user is looking at. */
    fun setActive(value: Boolean) {
        if (active == value) return
        active = value
        webView?.let { if (value) it.onResume() else it.onPause() }
    }

    fun destroy() {
        host?.dispose()
        host = null
        webView = null
        phase = Phase.Idle
    }

    private fun open(pack: GymPack, language: String) {
        val created = try {
            PackWebView(viewContext, pack, Page())
        } catch (error: Exception) {
            // No usable system WebView on this device.
            Log.e(TAG, "web view unavailable", error)
            phase = Phase.Failed(GymFailure.Load(error.message ?: error.javaClass.simpleName))
            return
        }
        host = created
        webView = created.view
        if (!active) created.view.onPause()
        load(created, pack, language)
    }

    private fun load(host: PackWebView, pack: GymPack, language: String) {
        phase = Phase.Loading
        pageReady = false
        documentFailed = false
        loadedLanguage = language
        loadStarted = SystemClock.elapsedRealtime()
        val query = if (BuildConfig.DEBUG) debug.query else null
        host.load(pack.entryUrl(hintSeen = preferences.getBoolean(KEY_HINT_SEEN, false), language = language, extraQuery = query))
    }

    /** What the web view reports. Each web view is disposed before another is made, so these are always about the current one. */
    private inner class Page : PackWebView.Listener {
        override fun onMessage(json: String?) {
            receive(GymMessage.parse(json) ?: return)
        }

        override fun onDocumentLoaded(url: String?) {
            Log.i(TAG, "document loaded in ${SystemClock.elapsedRealtime() - loadStarted} ms")
            // A page that never posts `ready` still gets shown once its document is in.
            if (phase == Phase.Loading && !documentFailed && GymPack.allows(url)) phase = Phase.Ready
        }

        override fun onLoadFailed(description: String) {
            documentFailed = true
            phase = Phase.Failed(GymFailure.Load(description))
        }

        override fun onRendererGone() {
            host = null
            webView = null
            phase = Phase.Failed(GymFailure.RendererGone)
        }
    }

    private fun receive(message: GymMessage) {
        when (message) {
            GymMessage.Ready -> {
                pageReady = true
                Log.i(TAG, "page ready in ${SystemClock.elapsedRealtime() - loadStarted} ms")
                if (phase == Phase.Loading) phase = Phase.Ready
                if (BuildConfig.DEBUG) {
                    // `--es gymOpenWall <id>` / `--es gymOpenRoute <id>` on the launch intent, for screenshots and bridge checks.
                    debug.openWall?.let { send(GymCommand.OpenWall(it)) }
                    debug.openRoute?.let { send(GymCommand.OpenRoute(it)) }
                }
                routeToRestore?.let { send(GymCommand.OpenRoute(it)) }
                routeToRestore = null
            }
            GymMessage.HintSeen -> preferences.edit { putBoolean(KEY_HINT_SEEN, true) }
            // Only while the page is what the user is looking at; a hidden page cannot open anything.
            GymMessage.OpenSettings -> if (active) settingsRequested = true
            is GymMessage.RouteSelected -> selectedRoute = message.route
            is GymMessage.Error -> {
                Log.e(TAG, "page error (${if (pageReady) "after" else "before"} ready): ${message.message}")
                if (!pageReady) phase = Phase.Failed(GymFailure.Page(message.message))
            }
        }
    }

    companion object {
        private const val TAG = "gym"   // Local logcat only: load timings and the text of page-reported errors.
        private const val KEY_NOTICE_SEEN = "gymNoticeSeen"
        private const val KEY_HINT_SEEN = "gymHintSeen"
    }
}

/** Keeps the session across activity recreation; the page is torn down when the activity is gone for good. */
class GymViewModel(application: Application) : AndroidViewModel(application) {
    val session = GymSession(application)

    override fun onCleared() {
        session.destroy()
    }
}
