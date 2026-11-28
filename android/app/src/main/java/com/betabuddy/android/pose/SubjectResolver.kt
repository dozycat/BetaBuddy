package com.betabuddy.android.pose

import android.annotation.SuppressLint
import android.content.Context
import android.webkit.WebResourceRequest
import android.webkit.WebResourceResponse
import android.webkit.WebView
import android.webkit.WebViewClient
import com.betabuddy.android.data.SubjectCandidate
import com.betabuddy.android.data.PoseFrame
import com.betabuddy.android.data.PoseAnalysis
import com.betabuddy.android.data.SubjectAnchor
import com.betabuddy.android.data.SubjectResolution
import com.betabuddy.android.data.SubjectSample
import com.betabuddy.android.data.SubjectPick
import com.betabuddy.android.data.selecting
import com.betabuddy.android.gym.GymManifest
import java.io.ByteArrayInputStream
import java.io.File
import kotlin.coroutines.resume
import kotlin.coroutines.coroutineContext
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

    /** Recreate drawn regions from the original video before resolving saved choices. Full-frame
     * pose detection alone can miss a small selected person, even when it finds other people. */
    internal suspend fun analyze(video: File, quality: AnalysisQuality, cameraClip: Boolean = false,
        anchors: List<SubjectAnchor>, progress: (Double) -> Unit): PoseAnalysis {
        val regions = anchors.filter { it.region }
        val passes = 1.0 + regions.size
        val canRecover=anchors.isEmpty()
        var detected = try {
            VideoAnalyzer(context).analyze(video,quality,cameraClip=cameraClip,collectSubjects=true) { progress(it/passes*(if(canRecover) .7 else 1.0)) }
        } catch (failure: AnalysisException) {
            if (failure.error != AnalysisError.NO_POSE || (regions.isEmpty() && !canRecover)) throw failure
            val info = VideoAnalyzer.probe(video)
            val size = FrameGeometry.displaySize(info.width,info.height,info.rotation)
            PoseAnalysis(emptyList(),0,requireNotNull(info.durationSeconds),size.first.toDouble()/size.second,people=emptyList())
        }
        for ((index, anchor) in regions.withIndex()) {
            detected = SubjectRegionAnalyzer(context).refine(video,detected,anchor.bounds,anchor.time) {
                progress((index+1+it)/passes)
            }.analysis
        }
        var result = resolve(detected,anchors)
        if(canRecover && result.needsSubjectRecovery) {
            result=recover(video,result) { progress(.7+.3*it) }
        }
        if(!result.hasSubjectCandidates) throw AnalysisException(AnalysisError.NO_POSE)
        progress(1.0)
        return result
    }

    internal suspend fun recover(video: File,analysis: PoseAnalysis,progress: (Double) -> Unit): PoseAnalysis {
        val worker=SubjectRegionAnalyzer(context)
        val proposals=worker.discover(video,analysis) { progress(it/3) }
        val seeds=proposedRegions(proposals)
        var result=analysis
        for((index,seed) in seeds.withIndex()) {
            result=worker.refine(video,result,seed.bounds,seed.time) { progress(1.0/3+2.0/3*(index+it)/seeds.size) }.analysis
        }
        result=resolve(result,emptyList());progress(1.0)
        return result
    }

    internal suspend fun proposedRegions(samples: List<SubjectSample>): List<SubjectAnchor> {
        val encoded=withPolicy { policy->
            policy.evaluate("window.__proposal = {samples:[]}; true")
            for(chunk in samples.chunked(8)) {
                coroutineContext.ensureActive()
                val input=withContext(Dispatchers.Default) { json.encodeToString(chunk) }
                check(policy.evaluate("window.__proposal.samples.push(...JSON.parse(${JSONObject.quote(input)})); true")=="true")
            }
            json.decodeFromString<String>(policy.evaluate("JSON.stringify(typeof BetaBuddySubject.proposeRegions === 'function' ? BetaBuddySubject.proposeRegions(window.__proposal) : [])"))
        }
        val picks=json.decodeFromString<List<SubjectPick>>(encoded)
        require(picks.size<=2)
        return picks.map { pick->
            val sample=requireNotNull(samples.firstOrNull { kotlin.math.abs(it.time-pick.time)<.0001 })
            val candidate=requireNotNull(sample.candidates.firstOrNull { it.id==pick.personID })
            require(candidate.bounds.valid)
            // Leave image context around a pose envelope that may hug a visible limb.
            SubjectAnchor(sample.time,candidate.copy(bounds=candidate.bounds.expanded(.15)))
        }
    }

    suspend fun resolve(analysis: PoseAnalysis, anchors: List<SubjectAnchor>): PoseAnalysis = withTimeout(120_000) {
        val samples = requireNotNull(analysis.people)
        val (input, chunks) = withContext(Dispatchers.Default) {
            check(available()) { "Subject policy is missing" }
            Pair(json.encodeToString(Input(anchors = anchors)),
                samples.chunked(32).map { chunk -> json.encodeToString(chunk.map { sample ->
                    sample.copy(candidates = sample.candidates.map { it.copy(pose = it.pose.copy(world = null)) })
                }) })
        }
        val encoded = withPolicy { policy ->
            policy.evaluate("window.__subject = JSON.parse(${JSONObject.quote(input)}); true")
            for (chunk in chunks) {
                coroutineContext.ensureActive()
                check(policy.evaluate("window.__subject.samples.push(...JSON.parse(${JSONObject.quote(chunk)})); true") == "true")
            }
            json.decodeFromString<String>(policy.evaluate("JSON.stringify(BetaBuddySubject.resolve(window.__subject))"))
        }
        withContext(Dispatchers.Default) {
            analysis.selecting(json.decodeFromString<SubjectResolution>(encoded))
        }
    }

    @Serializable private data class RegionInput(val previous: SubjectCandidate, val candidates: List<SubjectCandidate>, val dt: Double)
    @Serializable private data class TrackingInput(val previous: SubjectCandidate, val candidate: SubjectCandidate, val seed: SubjectCandidate, val dt: Double)
    @Serializable private data class Match(val id: String? = null, val stop: Boolean = false)

    /** One isolated policy runtime per region pass, rather than a new WebView per frame. */
    internal inner class Policy(private val web: WebView) {
        suspend fun evaluate(script: String): String = withTimeout(15_000) { withContext(Dispatchers.Main) {
            suspendCancellableCoroutine { continuation ->
                web.evaluateJavascript(script) { value -> if (continuation.isActive) continuation.resume(value) }
            }
        } }
        suspend fun matchRegion(previous: SubjectCandidate,candidates: List<SubjectCandidate>,delta: Double): String? {
            val input = withContext(Dispatchers.Default) { json.encodeToString(RegionInput(previous,candidates,delta)) }
            val result = evaluate("JSON.stringify(BetaBuddySubject.matchRegion(JSON.parse(${JSONObject.quote(input)})))")
            return json.decodeFromString<Match>(json.decodeFromString<String>(result)).id
        }
        suspend fun accept(previous: SubjectCandidate,candidate: SubjectCandidate,seed: SubjectCandidate,delta: Double): Boolean {
            fun compact(value: SubjectCandidate) = value.copy(pose = PoseFrame(value.pose.time,emptyList()))
            val input = withContext(Dispatchers.Default) { json.encodeToString(TrackingInput(compact(previous),compact(candidate),compact(seed),delta)) }
            return evaluate("BetaBuddySubject.acceptTracking(JSON.parse(${JSONObject.quote(input)}))") == "true"
        }
    }

    @SuppressLint("SetJavaScriptEnabled")
    internal suspend fun <T> withPolicy(action: suspend (Policy) -> T): T {
        val source = withContext(Dispatchers.IO) {
            check(available()) { "Subject policy is missing" }
            context.assets.open("gympack/subject.js").bufferedReader().use { it.readText() }
        }
        return withContext(Dispatchers.Main) {
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
                withTimeout(15_000) { suspendCancellableCoroutine<Unit> { continuation ->
                    web.webViewClient = object : WebViewClient() {
                        override fun shouldOverrideUrlLoading(view: WebView?, request: WebResourceRequest?) = true
                        override fun shouldInterceptRequest(view: WebView?, request: WebResourceRequest?) =
                            WebResourceResponse("text/plain", "utf-8", ByteArrayInputStream(byteArrayOf()))
                        override fun onPageFinished(view: WebView?, url: String?) {
                            if (continuation.isActive) continuation.resume(Unit)
                        }
                    }
                    web.loadDataWithBaseURL(null, "<!doctype html><title>Local selection</title>", "text/html", "utf-8", null)
                } }
                val policy = Policy(web)
                check(policy.evaluate("$source; typeof BetaBuddySubject.resolve === 'function'") == "true") { "Invalid subject policy" }
                action(policy)
            } finally { web.stopLoading(); web.destroy() }
        }
    }
}
