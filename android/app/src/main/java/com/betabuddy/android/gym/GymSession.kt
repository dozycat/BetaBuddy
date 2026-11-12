package com.betabuddy.android.gym

import android.annotation.SuppressLint
import android.app.Application
import android.content.Context
import android.content.MutableContextWrapper
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.net.Uri
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import android.view.View
import android.view.ViewGroup
import android.webkit.ConsoleMessage
import android.webkit.JavascriptInterface
import android.webkit.RenderProcessGoneDetail
import android.webkit.WebChromeClient
import android.webkit.WebResourceError
import android.webkit.WebResourceRequest
import android.webkit.WebResourceResponse
import android.webkit.WebSettings
import android.webkit.WebStorage
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.core.content.edit
import androidx.core.graphics.toColorInt
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.lifecycle.AndroidViewModel
import androidx.webkit.WebViewAssetLoader
import androidx.webkit.WebViewCompat
import androidx.webkit.WebViewFeature
import com.betabuddy.android.BuildConfig
import java.io.ByteArrayInputStream
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
 */
class GymSession(private val app: Application) {
    sealed interface Phase {
        data object Idle : Phase
        data object Loading : Phase
        data object Ready : Phase
        data class Failed(val failure: GymFailure) : Phase
    }

    val status: GymPackStatus = GymPack.load(AssetGymPackFiles(app.assets))
    private val pack: GymPack? get() = (status as? GymPackStatus.Ready)?.pack

    var phase: Phase by mutableStateOf<Phase>(Phase.Idle)
        private set
    var webView: WebView? by mutableStateOf<WebView?>(null)
        private set

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
    private val main = Handler(Looper.getMainLooper())

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
        val view = webView
        when {
            view == null && phase == Phase.Idle -> open(pack, language)
            view != null && phase !is Phase.Failed && loadedLanguage != language -> {
                routeToRestore = selectedRoute?.id
                load(view, pack, language)
            }
        }
    }

    /** The reload button on the failure card. */
    fun reload(language: String) {
        val pack = pack ?: return
        val view = webView
        if (view == null) open(pack, language) else load(view, pack, language)
    }

    fun send(command: GymCommand) {
        if (phase == Phase.Ready) webView?.evaluateJavascript(command.script, null)
    }

    /** Pauses the page's animation while the gym is not what the user is looking at. */
    fun setActive(value: Boolean) {
        if (active == value) return
        active = value
        webView?.let { if (value) it.onResume() else it.onPause() }
    }

    fun destroy() {
        webView?.let(::dispose)
        webView = null
        phase = Phase.Idle
    }

    private fun open(pack: GymPack, language: String) {
        val view = try {
            create(pack)
        } catch (error: Exception) {
            // No usable system WebView on this device.
            Log.e(TAG, "web view unavailable", error)
            phase = Phase.Failed(GymFailure.Load(error.message ?: error.javaClass.simpleName))
            return
        }
        webView = view
        if (!active) view.onPause()
        load(view, pack, language)
    }

    private fun load(view: WebView, pack: GymPack, language: String) {
        phase = Phase.Loading
        pageReady = false
        documentFailed = false
        loadedLanguage = language
        loadStarted = SystemClock.elapsedRealtime()
        val query = if (BuildConfig.DEBUG) debug.query else null
        view.loadUrl(pack.entryUrl(hintSeen = preferences.getBoolean(KEY_HINT_SEEN, false), language = language, extraQuery = query))
    }

    @SuppressLint("SetJavaScriptEnabled", "JavascriptInterface")
    private fun create(pack: GymPack): WebView {
        // Nothing the page stores survives a launch.
        WebStorage.getInstance().deleteAllData()
        WebView.setWebContentsDebuggingEnabled(BuildConfig.DEBUG)

        val view = WebView(viewContext)
        view.layoutParams = ViewGroup.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT)
        view.setBackgroundColor(PAPER.toColorInt())
        view.overScrollMode = View.OVER_SCROLL_NEVER
        view.isVerticalScrollBarEnabled = false
        view.isHorizontalScrollBarEnabled = false
        view.isHapticFeedbackEnabled = false
        view.setOnLongClickListener { true }   // No text selection or link menus; gestures belong to the page.
        // The screen pads the web view clear of the system bars itself, so the page must not also
        // be told about them through env(safe-area-inset-*).
        ViewCompat.setOnApplyWindowInsetsListener(view) { _, _ -> WindowInsetsCompat.CONSUMED }

        view.settings.apply {
            javaScriptEnabled = true
            domStorageEnabled = true            // In-memory use within a run; cleared at every start.
            // Offline, enforced here rather than left to the page:
            blockNetworkLoads = true
            allowFileAccess = false
            allowContentAccess = false
            mixedContentMode = WebSettings.MIXED_CONTENT_NEVER_ALLOW
            cacheMode = WebSettings.LOAD_NO_CACHE
            safeBrowsingEnabled = false
            setGeolocationEnabled(false)
            javaScriptCanOpenWindowsAutomatically = false
            setSupportMultipleWindows(false)
            // The page lays itself out; no zooming, no font scaling from the system.
            setSupportZoom(false)
            builtInZoomControls = false
            displayZoomControls = false
            textZoom = 100
            mediaPlaybackRequiresUserGesture = false
        }

        val loader = WebViewAssetLoader.Builder()
            .setDomain(GymPack.HOST)
            .setHttpAllowed(false)
            .addPathHandler(GymPack.PATH_PREFIX, PackPathHandler(app.assets, pack))
            .build()
        view.webViewClient = Client(pack, loader, documentStartShim = installBridge(view))
        view.webChromeClient = object : WebChromeClient() {
            override fun onConsoleMessage(message: ConsoleMessage): Boolean {
                if (BuildConfig.DEBUG) Log.d(TAG, "console: ${message.message()} (${message.sourceId()}:${message.lineNumber()})")
                return true
            }
        }
        return view
    }

    /**
     * Gives the page `window.webkit.messageHandlers.gym`. Returns whether the shim is installed to
     * run at document start; when the WebView is too old for that, the client injects it by hand.
     */
    @SuppressLint("JavascriptInterface", "RequiresFeature")
    private fun installBridge(view: WebView): Boolean {
        view.addJavascriptInterface(Bridge(view), GymBridge.NATIVE_NAME)
        if (!WebViewFeature.isFeatureSupported(WebViewFeature.DOCUMENT_START_SCRIPT)) return false
        WebViewCompat.addDocumentStartJavaScript(view, GymBridge.shim, setOf(GymPack.ORIGIN))
        return true
    }

    /** Called by the page, on a WebView thread. */
    private inner class Bridge(private val owner: WebView) {
        @JavascriptInterface
        fun postMessage(json: String?) {
            main.post { if (owner === webView) receive(GymMessage.parse(json) ?: return@post) }
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

    private fun fail(failure: GymFailure) {
        documentFailed = true
        phase = Phase.Failed(failure)
    }

    private fun dispose(view: WebView) {
        (view.parent as? ViewGroup)?.removeView(view)
        view.removeJavascriptInterface(GymBridge.NATIVE_NAME)
        view.destroy()
    }

    private inner class Client(
        private val pack: GymPack,
        private val loader: WebViewAssetLoader,
        private val documentStartShim: Boolean,
    ) : WebViewClient() {

        /**
         * The offline rule. A request is answered from the pack, answered with an empty refusal, or
         * (for in-memory `data:` / `blob:` / `about:` content) left to the WebView. Nothing reaches a network stack.
         */
        override fun shouldInterceptRequest(view: WebView, request: WebResourceRequest): WebResourceResponse? {
            val url = request.url
            return when {
                url.scheme?.lowercase() in IN_MEMORY_SCHEMES -> null
                GymPack.allows(url.toString()) -> loader.shouldInterceptRequest(url) ?: refusal(404, "Not Found")
                else -> refusal(403, "Blocked")
            }
        }

        /** Navigation stays inside the pack. Links to anywhere else do nothing. */
        override fun shouldOverrideUrlLoading(view: WebView, request: WebResourceRequest): Boolean {
            val url = request.url
            val blankSubframe = !request.isForMainFrame && url.scheme?.lowercase() == "about"
            return !(GymPack.allows(url.toString()) || blankSubframe)
        }

        override fun onPageStarted(view: WebView, url: String?, favicon: Bitmap?) {
            if (!documentStartShim) view.evaluateJavascript(GymBridge.shim, null)
        }

        override fun onPageFinished(view: WebView, url: String?) {
            if (view !== webView) return
            if (!documentStartShim) view.evaluateJavascript(GymBridge.shim, null)
            Log.i(TAG, "document loaded in ${SystemClock.elapsedRealtime() - loadStarted} ms")
            // A page that never posts `ready` still gets shown once its document is in.
            if (phase == Phase.Loading && !documentFailed && GymPack.allows(url)) phase = Phase.Ready
        }

        override fun onReceivedError(view: WebView, request: WebResourceRequest, error: WebResourceError) {
            if (view !== webView || !request.isForMainFrame) return
            Log.e(TAG, "load failed: ${error.errorCode} ${error.description}")
            fail(GymFailure.Load(error.description?.toString().orEmpty()))
        }

        override fun onReceivedHttpError(view: WebView, request: WebResourceRequest, errorResponse: WebResourceResponse) {
            if (view !== webView || !request.isForMainFrame) return
            Log.e(TAG, "load failed: HTTP ${errorResponse.statusCode}")
            fail(GymFailure.Load("${errorResponse.statusCode} ${errorResponse.reasonPhrase.orEmpty()}".trim()))
        }

        /** The renderer died or was killed for memory. The view is unusable; drop it and offer a reload. */
        override fun onRenderProcessGone(view: WebView, detail: RenderProcessGoneDetail): Boolean {
            Log.e(TAG, "renderer gone (crashed: ${detail.didCrash()})")
            if (view === webView) {
                webView = null
                phase = Phase.Failed(GymFailure.RendererGone)
            }
            dispose(view)
            return true   // Handled: the app keeps running.
        }
    }

    /** Serves `/gympack/<path>` from the pack in the app's assets, and only paths the manifest lists. */
    private class PackPathHandler(private val assets: AssetManager, private val pack: GymPack) : WebViewAssetLoader.PathHandler {
        override fun handle(path: String): WebResourceResponse? {
            if (path !in pack.servedPaths) return null
            val stream = try {
                assets.open(GymPack.ASSET_DIRECTORY + "/" + path)
            } catch (error: IOException) {
                return null
            }
            val (type, charset) = GymPack.contentType(path)
            return WebResourceResponse(type, charset, 200, "OK", RESPONSE_HEADERS, stream)
        }
    }

    companion object {
        private const val TAG = "gym"   // Local logcat only: load timings and the text of page-reported errors.
        private const val KEY_NOTICE_SEEN = "gymNoticeSeen"
        private const val KEY_HINT_SEEN = "gymHintSeen"
        private const val PAPER = "#F5F2E8"
        private val IN_MEMORY_SCHEMES = setOf("data", "blob", "about")

        // No remote source of any kind. The pack uses inline scripts, Blob workers and data: URLs.
        const val CONTENT_SECURITY_POLICY =
            "default-src 'self' 'unsafe-inline' 'wasm-unsafe-eval' blob: data:; object-src 'none'; base-uri 'self'; form-action 'none'"

        private val RESPONSE_HEADERS = mapOf(
            "Cache-Control" to "no-store",
            "Content-Security-Policy" to CONTENT_SECURITY_POLICY,
            "X-Content-Type-Options" to "nosniff",
        )

        private fun refusal(status: Int, reason: String) =
            WebResourceResponse("text/plain", "utf-8", status, reason, mapOf("Cache-Control" to "no-store"), ByteArrayInputStream(ByteArray(0)))
    }
}

/** Keeps the session across activity recreation; the page is torn down when the activity is gone for good. */
class GymViewModel(application: Application) : AndroidViewModel(application) {
    val session = GymSession(application)

    override fun onCleared() {
        session.destroy()
    }
}
