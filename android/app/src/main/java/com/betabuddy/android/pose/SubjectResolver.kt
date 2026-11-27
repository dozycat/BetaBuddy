package com.betabuddy.android.pose

import android.annotation.SuppressLint
import android.content.Context
import android.webkit.WebResourceRequest
import android.webkit.WebResourceResponse
import android.webkit.WebView
import android.webkit.WebViewClient
import com.betabuddy.android.data.PoseAnalysis
import com.betabuddy.android.data.SubjectAnchor
import com.betabuddy.android.data.SubjectResolution
import com.betabuddy.android.data.SubjectSample
import com.betabuddy.android.data.selecting
import com.betabuddy.android.gym.GymManifest
import java.io.ByteArrayInputStream
import kotlin.coroutines.resume
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout
import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import org.json.JSONObject

/** Executes the same bundled policy as iOS. No bridge, file access, network or remote scripts. */
class SubjectResolver(context: Context) {
    private val context = context.applicationContext
    private val json = Json { ignoreUnknownKeys = true; encodeDefaults = true }

    fun available(): Boolean = runCatching {
        val manifest = context.assets.open("gympack/manifest.json").bufferedReader().use { GymManifest.parse(it.readText()) }
        manifest.files.any { it.path == "subject.js" }
    }.getOrDefault(false)

    @Serializable private data class Input(val schema: Int = 1, val anchors: List<SubjectAnchor>, val samples: List<SubjectSample> = emptyList())

    suspend fun resolve(analysis: PoseAnalysis, anchors: List<SubjectAnchor>): PoseAnalysis = withTimeout(120_000) {
        val samples = requireNotNull(analysis.people)
        val (source, input, chunks) = withContext(Dispatchers.Default) {
            check(available()) { "Subject policy is missing" }
            Triple(context.assets.open("gympack/subject.js").bufferedReader().use { it.readText() }, json.encodeToString(Input(anchors = anchors)),
                samples.chunked(32).map { chunk -> json.encodeToString(chunk.map { sample ->
                    sample.copy(candidates = sample.candidates.map { it.copy(pose = it.pose.copy(world = null)) })
                }) })
        }
        val encoded = execute(source, input, chunks)
        withContext(Dispatchers.Default) {
            analysis.selecting(json.decodeFromString<SubjectResolution>(encoded))
        }
    }

    @SuppressLint("SetJavaScriptEnabled")
    private suspend fun execute(source: String, input: String, chunks: List<String>): String = withContext(Dispatchers.Main) {
        val web = WebView(context)
        try {
            web.settings.apply {
                javaScriptEnabled = true
                blockNetworkLoads = true
                allowFileAccess = false
                allowContentAccess = false
                domStorageEnabled = false
                javaScriptCanOpenWindowsAutomatically = false
                setGeolocationEnabled(false)
            }
            suspendCancellableCoroutine<Unit> { continuation ->
                web.webViewClient = object : WebViewClient() {
                    override fun shouldOverrideUrlLoading(view: WebView?, request: WebResourceRequest?) = true
                    override fun shouldInterceptRequest(view: WebView?, request: WebResourceRequest?) =
                        WebResourceResponse("text/plain", "utf-8", ByteArrayInputStream(byteArrayOf()))
                    override fun onPageFinished(view: WebView?, url: String?) {
                        if (continuation.isActive) continuation.resume(Unit)
                    }
                }
                web.loadDataWithBaseURL(null, "<!doctype html><title>Local selection</title>", "text/html", "utf-8", null)
            }
            suspend fun evaluate(script: String): String = suspendCancellableCoroutine { continuation ->
                web.evaluateJavascript(script) { value -> if (continuation.isActive) continuation.resume(value) }
            }
            check(evaluate("$source; typeof BetaBuddySubject.resolve === 'function'") == "true") { "Invalid subject policy" }
            evaluate("window.__subject = JSON.parse(${JSONObject.quote(input)}); true")
            for (chunk in chunks) {
                coroutineContext.ensureActive()
                check(evaluate("window.__subject.samples.push(...JSON.parse(${JSONObject.quote(chunk)})); true") == "true")
            }
            val encoded = evaluate("JSON.stringify(BetaBuddySubject.resolve(window.__subject))")
            json.decodeFromString<String>(encoded)
        } finally {
            web.stopLoading(); web.destroy()
        }
    }
}
