package com.betabuddy.android.pose

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.RectF
import com.betabuddy.android.data.SubjectBounds
import java.io.Closeable
import java.nio.FloatBuffer

/** Generic image-region tracking; the private policy decides whether an observation is acceptable.
 * Model preprocessing/output geometry adapted from OpenCV TrackerVit 4.12.0 (Apache-2.0):
 * https://github.com/opencv/opencv/blob/4.12.0/modules/video/src/tracking/tracker_vit.cpp
 * The seed template stays fixed, including when pose detection temporarily finds no body.
 */
internal class SubjectBoxTracker(context: Context) : Closeable {
    data class Observation(val bounds: SubjectBounds,val confidence: Double)
    private val environment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private var template: OnnxTensor? = null
    private var previous: RegionTrackerGeometry.Box? = null
    init {
        environment.setTelemetry(false)
        val bytes = context.assets.open(SubjectTrackerModel.ASSET).use { it.readBytes() }
        require(PoseModel.sha256(bytes) == SubjectTrackerModel.SHA256) { "Tracker model does not match this build" }
        session = OrtSession.SessionOptions().use {
            it.setIntraOpNumThreads(2); it.setInterOpNumThreads(1)
            environment.createSession(bytes,it)
        }
    }
    fun start(image: Bitmap,bounds: SubjectBounds) {
        require(bounds.valid)
        template?.close(); template = null
        previous = RegionTrackerGeometry.pixels(bounds,image.width,image.height)
        template = tensor(image,RegionTrackerGeometry.crop(previous!!,2),128)
    }
    fun next(image: Bitmap): Observation? {
        val box = previous ?: return null
        val seed = template ?: return null
        val crop = RegionTrackerGeometry.crop(box,4)
        return tensor(image,crop,256).use { search ->
            session.run(mapOf("template" to seed,"search" to search)).use { result ->
                fun output(name: String): FloatArray {
                    val buffer = (result.get(name).orElseThrow() as OnnxTensor).floatBuffer
                    return FloatArray(buffer.remaining()).also(buffer::get)
                }
                val prediction = RegionTrackerGeometry.prediction(crop,output("output1"),output("output2"),output("output3")) ?: return null
                val bounds = prediction.box.normalized(image.width,image.height) ?: return null
                previous = prediction.box
                Observation(bounds,prediction.confidence)
            }
        }
    }
    private fun tensor(image: Bitmap,crop: RegionTrackerGeometry.Crop,side: Int): OnnxTensor {
        val square = Bitmap.createBitmap(side,side,Bitmap.Config.ARGB_8888)
        val pixels = IntArray(side*side)
        try {
            val scale = side.toFloat()/crop.side
            val destination = RectF(-crop.x*scale,-crop.y*scale,(image.width-crop.x)*scale,(image.height-crop.y)*scale)
            Canvas(square).drawBitmap(image,null,destination,Paint(Paint.FILTER_BITMAP_FLAG))
            square.getPixels(pixels,0,side,0,0,side,side)
        } finally { square.recycle() }
        val values = FloatArray(pixels.size*3)
        val mean = floatArrayOf(.485f,.456f,.406f); val std = floatArrayOf(.229f,.224f,.225f)
        // This model's OpenCV input uses BGR without swapRB (unlike the library encoder).
        for (channel in 0..2) for (i in pixels.indices)
            values[channel*pixels.size+i] = (((pixels[i] shr (channel*8)) and 255)/255f-mean[channel])/std[channel]
        return OnnxTensor.createTensor(environment,FloatBuffer.wrap(values),longArrayOf(1,3,side.toLong(),side.toLong()))
    }
    override fun close() { template?.close(); template = null; session.close() }
}
