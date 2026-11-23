package com.betabuddy.android.gym

import android.annotation.SuppressLint
import android.content.Context
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.os.Handler
import android.os.Looper
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
import androidx.core.graphics.toColorInt
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.webkit.WebViewAssetLoader
import androidx.webkit.WebViewCompat
import androidx.webkit.WebViewFeature
import com.betabuddy.android.BuildConfig
import java.io.ByteArrayInputStream
import java.io.IOException

/**
 * One web view showing a page of the gym pack: the gym itself, or the pack's analysis page.
 * Everything that keeps such a page offline and inside the pack is here, once: the local address
 * answered from the app's assets (manifest-listed files only), the request and navigation rules,
 * and the shim that gives the page `window.webkit.messageHandlers.gym`.
 *
 * Main thread only. Throws from the constructor when the device has no usable system WebView.
 */
class PackWebView(context: Context, private val pack: GymPack, private val listener: Listener, private val files: Files? = null) {
    /** Where the pack's files are read from. The app reads its own assets; device tests serve a synthetic pack. */
    fun interface Files {
        /** The file at this pack-relative path, or null when there is none. */
        fun open(path: String): java.io.InputStream?
    }

    /** All calls arrive on the main thread, and none after [dispose]. */
    interface Listener {
        /** The JSON text of one message the page posted to its `gym` handler. */
        fun onMessage(json: String?)
        /** The main document finished loading from [url]. */
        fun onDocumentLoaded(url: String?)
        /** The main document did not load. */
        fun onLoadFailed(description: String)
        /** The system took the page's renderer process away, usually for memory. The view is already unusable. */
        fun onRendererGone()
    }

    val view: WebView
    private val main = Handler(Looper.getMainLooper())
    private var disposed = false

    init {
        clearStorageOnce()
        WebView.setWebContentsDebuggingEnabled(BuildConfig.DEBUG)
        view = create(context)
    }

    fun load(url: String) {
        if (!disposed) view.loadUrl(url)
    }

    fun evaluate(script: String) {
        if (!disposed) view.evaluateJavascript(script, null)
    }

    /** [answer] gets the JSON text of what the script evaluated to, on the main thread; nothing once disposed. */
    fun evaluate(script: String, answer: (String?) -> Unit) {
        if (disposed) return
        view.evaluateJavascript(script) { value -> if (!disposed) answer(value) }
    }

    fun dispose() {
        if (disposed) return
        disposed = true
        (view.parent as? ViewGroup)?.removeView(view)
        view.removeJavascriptInterface(GymBridge.NATIVE_NAME)
        view.destroy()
    }

    @SuppressLint("SetJavaScriptEnabled")
    private fun create(context: Context): WebView {
        val view = WebView(context)
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
            .addPathHandler(GymPack.PATH_PREFIX, PackPathHandler(files ?: AssetFiles(context.applicationContext.assets), pack))
            .build()
        view.webViewClient = Client(loader, documentStartShim = installBridge(view))
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
        view.addJavascriptInterface(Bridge(), GymBridge.NATIVE_NAME)
        if (!WebViewFeature.isFeatureSupported(WebViewFeature.DOCUMENT_START_SCRIPT)) return false
        WebViewCompat.addDocumentStartJavaScript(view, GymBridge.shim, setOf(GymPack.ORIGIN))
        return true
    }

    /** Called by the page, on a WebView thread. */
    private inner class Bridge {
        @JavascriptInterface
        fun postMessage(json: String?) {
            main.post { if (!disposed) listener.onMessage(json) }
        }
    }

    private inner class Client(
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
            if (disposed) return
            if (!documentStartShim) view.evaluateJavascript(GymBridge.shim, null)
            listener.onDocumentLoaded(url)
        }

        override fun onReceivedError(view: WebView, request: WebResourceRequest, error: WebResourceError) {
            if (disposed || !request.isForMainFrame) return
            Log.e(TAG, "load failed: ${error.errorCode} ${error.description}")
            listener.onLoadFailed(error.description?.toString().orEmpty())
        }

        override fun onReceivedHttpError(view: WebView, request: WebResourceRequest, errorResponse: WebResourceResponse) {
            if (disposed || !request.isForMainFrame) return
            Log.e(TAG, "load failed: HTTP ${errorResponse.statusCode}")
            listener.onLoadFailed("${errorResponse.statusCode} ${errorResponse.reasonPhrase.orEmpty()}".trim())
        }

        /** The renderer died or was killed for memory. The view is unusable; drop it and let the owner offer a reload. */
        override fun onRenderProcessGone(view: WebView, detail: RenderProcessGoneDetail): Boolean {
            Log.e(TAG, "renderer gone (crashed: ${detail.didCrash()})")
            if (!disposed) listener.onRendererGone()
            dispose()
            return true   // Handled: the app keeps running.
        }
    }

    private class AssetFiles(private val assets: AssetManager) : Files {
        override fun open(path: String): java.io.InputStream? = try {
            assets.open(GymPack.ASSET_DIRECTORY + "/" + path)
        } catch (error: IOException) {
            null
        }
    }

    /** Serves `/gympack/<path>` from the pack (the app's assets, outside tests), and only paths the manifest lists. */
    private class PackPathHandler(private val files: Files, private val pack: GymPack) : WebViewAssetLoader.PathHandler {
        override fun handle(path: String): WebResourceResponse? {
            if (path !in pack.servedPaths) return null
            val stream = files.open(path) ?: return null
            val (type, charset) = GymPack.contentType(path)
            return WebResourceResponse(type, charset, 200, "OK", RESPONSE_HEADERS, stream)
        }
    }

    companion object {
        private const val TAG = "gym"   // Local logcat only: load failures and the page's console in debug builds.
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

        private var storageCleared = false

        /** Nothing a pack page stores survives a launch: cleared before the first page of the process opens. */
        private fun clearStorageOnce() {
            if (storageCleared) return
            storageCleared = true
            WebStorage.getInstance().deleteAllData()
        }

        private fun refusal(status: Int, reason: String) =
            WebResourceResponse("text/plain", "utf-8", status, reason, mapOf("Cache-Control" to "no-store"), ByteArrayInputStream(ByteArray(0)))
    }
}
