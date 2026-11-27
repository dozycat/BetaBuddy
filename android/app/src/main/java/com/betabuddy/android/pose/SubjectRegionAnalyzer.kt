package com.betabuddy.android.pose

import android.content.Context
import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.os.Build
import android.os.PowerManager
import com.betabuddy.android.data.*
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import java.io.File
import java.util.concurrent.Executors
import kotlin.math.abs
import kotlin.math.ceil
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import kotlin.coroutines.coroutineContext

/** A drawn region supplies independent positions plus real poses from a closer crop.
 * Missing/ambiguous poses stay empty; an image tracker cannot invent limb positions.
 */
internal class SubjectRegionAnalyzer(context: Context) {
    private val context = context.applicationContext
    internal var observationObserver: ((SubjectCandidate,SubjectCandidate,SubjectCandidate,Boolean) -> Unit)? = null
    data class Result(val analysis: PoseAnalysis,val anchor: SubjectAnchor)

    suspend fun refine(video: File,analysis: PoseAnalysis,bounds: SubjectBounds,time: Double,progress: suspend (Double) -> Unit): Result = serial.withLock {
        require(bounds.valid && analysis.duration.isFinite() && analysis.duration > 0)
        SubjectResolver(context).withPolicy { policy -> withContext(worker) {
            val samples = (analysis.people?.takeIf { it.isNotEmpty() }
                ?: List(maxOf(1,ceil(analysis.duration*10).toInt())) { SubjectSample(it/10.0,emptyList()) }).toMutableList()
            val seedIndex = samples.indices.minBy { abs(samples[it].time-time) }
            val token = newId()
            val reader = MediaMetadataRetriever()
            val model = VideoAnalyzer.loadModel(context)
            val options = PoseLandmarker.PoseLandmarkerOptions.builder()
                .setBaseOptions(BaseOptions.builder().setModelAssetBuffer(model).setDelegate(Delegate.CPU).build())
                .setRunningMode(RunningMode.VIDEO).setNumPoses(4).build()
            var detector: PoseLandmarker? = null
            var seedImage: Bitmap? = null
            var timestamp = 0L
            val info = VideoAnalyzer.probe(video)
            val size = FrameGeometry.targetSize(info.width,info.height,1920)
            try {
                reader.setDataSource(video.path)
                detector = PoseLandmarker.createFromOptions(context,options)
                suspend fun image(index: Int): Bitmap {
                    coroutineContext.ensureActive()
                    if (Build.VERSION.SDK_INT >= 29 && context.getSystemService(PowerManager::class.java).currentThermalStatus >= PowerManager.THERMAL_STATUS_CRITICAL)
                        throw AnalysisException(AnalysisError.OVERHEATED)
                    return (if (Build.VERSION.SDK_INT >= 27) reader.getScaledFrameAtTime((samples[index].time*1_000_000).toLong(),MediaMetadataRetriever.OPTION_CLOSEST,size.first,size.second)
                        else reader.getFrameAtTime((samples[index].time*1_000_000).toLong(),MediaMetadataRetriever.OPTION_CLOSEST))
                        ?: throw AnalysisException(AnalysisError.INVALID_VIDEO)
                }
                fun poses(index: Int,image: Bitmap,around: SubjectBounds): List<SubjectCandidate> {
                    val area = around.expanded(.45)
                    val left = (area.x*image.width).toInt().coerceIn(0,image.width-1)
                    val top = (area.y*image.height).toInt().coerceIn(0,image.height-1)
                    val right = ceil((area.x+area.width)*image.width).toInt().coerceIn(left+1,image.width)
                    val bottom = ceil((area.y+area.height)*image.height).toInt().coerceIn(top+1,image.height)
                    val crop = Bitmap.createBitmap(image,left,top,right-left,bottom-top)
                    try {
                        val input = BitmapImageBuilder(crop).build()
                        timestamp += 100
                        val result = try { detector!!.detectForVideo(input,timestamp) } finally { input.close() }
                        return result.landmarks().mapIndexedNotNull { person,points ->
                            if (points.size != 33) return@mapIndexedNotNull null
                            val joints = points.map { p ->
                                val confidence = minOf(p.visibility().orElse(0f),p.presence().orElse(0f)).toDouble()
                                Joint(rounded4((left+p.x().toDouble()*(right-left))/image.width),rounded4((top+p.y().toDouble()*(bottom-top))/image.height),rounded4(confidence))
                            }
                            val world = result.worldLandmarks().getOrNull(person)?.takeIf { it.size == 33 }?.map { p ->
                                listOf(p.x(),p.y(),p.z(),p.visibility().orElse(0f)).map { if (it.isFinite()) rounded4(it.toDouble()) else 0.0 }
                            }
                            val pose = PoseFrame(samples[index].time,joints,world)
                            if (!pose.hasReliableTorso) return@mapIndexedNotNull null
                            val visible = joints.filter { it.valid }
                            val x = visible.minOf { it.x }; val y = visible.minOf { it.y }
                            val box = SubjectBounds(x,y,visible.maxOf { it.x }-x,visible.maxOf { it.y }-y)
                            if (!box.valid) return@mapIndexedNotNull null
                            SubjectCandidate("pose-$index-$person",box,emptyList(),pose)
                        }
                    } finally { if (crop !== image) crop.recycle() }
                }
                suspend fun candidate(index: Int,observation: SubjectBoxTracker.Observation,image: Bitmap): SubjectCandidate {
                    val box = observation.bounds
                    val center = SubjectBounds(box.x+box.width*.2,box.y+box.height*.2,box.width*.6,box.height*.6)
                    val candidate = SubjectCandidate("region-$token-$index",box,appearance(image,center),PoseFrame(samples[index].time,emptyList()),SubjectTracking(token,observation.confidence,"vittrack-2023sep"))
                    val poses = poses(index,image,box)
                    val match = policy.matchRegion(candidate,poses,0.0)
                    return poses.firstOrNull { it.id == match }?.let { candidate.copy(pose = it.pose) } ?: candidate
                }
                fun keep(candidate: SubjectCandidate,index: Int) {
                    samples[index] = samples[index].copy(candidates = samples[index].candidates.filter { it.bounds.overlap(candidate.bounds) <= .5 }+candidate)
                }
                seedImage = image(seedIndex)
                val first = candidate(seedIndex,SubjectBoxTracker.Observation(bounds,1.0),seedImage)
                keep(first,seedIndex)
                var done = 1
                SubjectBoxTracker(context).use { tracker ->
                    for ((direction,indices) in listOf((seedIndex+1 until samples.size).toList(),(0 until seedIndex).reversed().toList()).withIndex()) {
                        if (indices.isEmpty()) continue
                        tracker.start(seedImage,bounds)
                        if (direction == 1) { detector!!.close(); detector = null; detector = PoseLandmarker.createFromOptions(context,options); timestamp = 0 }
                        var previous = first
                        for (index in indices) {
                            val frame = image(index)
                            try {
                                val observation = tracker.next(frame) ?: break
                                val current = candidate(index,observation,frame)
                                val accepted = policy.accept(previous,current,first,abs(samples[index].time-previous.pose.time))
                                observationObserver?.invoke(previous,current,first,accepted)
                                if (!accepted) break
                                keep(current,index); previous = current; done += 1
                                if (done%5 == 0) progress(done.toDouble()/samples.size)
                            } finally { frame.recycle() }
                        }
                    }
                }
                coroutineContext.ensureActive(); progress(1.0)
                Result(analysis.copy(sampledFrames=samples.size,people=samples,peopleReference=null),SubjectAnchor(samples[seedIndex].time,first))
            } finally { seedImage?.recycle(); detector?.close(); reader.release() }
        } }
    }
    companion object {
        private val serial = Mutex()
        private val worker = Executors.newSingleThreadExecutor().asCoroutineDispatcher()
        // Average each horizontal band rather than four isolated pixels. Tiny moving regions
        // otherwise alternate between clothing and background even when the tracker is correct.
        private fun appearance(image: Bitmap,bounds: SubjectBounds): List<Double> = (0..3).flatMap { row -> (0..2).map { channel ->
            rounded4((0 until 64).sumOf { sample ->
                val x = ((bounds.x+bounds.width*(sample%16+.5)/16)*image.width).toInt().coerceIn(0,image.width-1)
                val y = ((bounds.y+bounds.height*(row+(sample/16+.5)/4)/4)*image.height).toInt().coerceIn(0,image.height-1)
                ((image.getPixel(x,y) shr ((2-channel)*8)) and 255).toDouble()
            }/(64*255))
        } }
    }
}
