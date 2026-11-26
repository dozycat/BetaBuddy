package com.betabuddy.android.edit

import android.content.Context
import android.graphics.Bitmap
import com.betabuddy.android.pose.PoseModel
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.objectdetector.ObjectDetector
import java.io.IOException
import java.nio.ByteBuffer

/**
 * Finds the people in one upright picture, as boxes in fractions of it (origin top-left). The
 * edit asks through this interface so that tests can put a made-up answer in its place.
 * One finder is used from one thread at a time.
 */
interface PeopleFinder : AutoCloseable {
    fun find(frame: Bitmap): List<Box>
    override fun close()
}

/** The bundled detector model: EfficientDet-Lite2 (int8) from the MediaPipe model storage, version 1. */
object PeopleModel {
    const val ASSET = "efficientdet_lite2.tflite"
    const val SHA256 = "b3f50554cb0ea559e90328845f7d9ba4d13c8bff372914d24e06bc8bb72fa896"
    const val BYTES = 7_515_971L
}

/**
 * People found by MediaPipe's object detector with the bundled model, on the CPU. Only the
 * "person" class is asked for. The threshold is low on purpose: a region wrongly taken for a
 * person is replaced by the same wall from the plate, which shows nothing, while a person missed
 * stays in the picture.
 */
class ModelPeopleFinder(context: Context) : PeopleFinder {
    private val detector: ObjectDetector

    init {
        val bytes = context.applicationContext.assets.open(PeopleModel.ASSET).use { it.readBytes() }
        // Anything but the pinned file is refused, as with the pose model.
        if (PoseModel.sha256(bytes) != PeopleModel.SHA256) throw IOException("The people detector model is not the bundled one")
        val model = ByteBuffer.allocateDirect(bytes.size).apply { put(bytes); rewind() }
        detector = ObjectDetector.createFromOptions(
            context.applicationContext,
            ObjectDetector.ObjectDetectorOptions.builder()
                .setBaseOptions(BaseOptions.builder().setModelAssetBuffer(model).setDelegate(Delegate.CPU).build())
                .setRunningMode(RunningMode.IMAGE)
                .setCategoryAllowlist(listOf("person"))
                .setScoreThreshold(SCORE_THRESHOLD)
                .setMaxResults(MAX_PEOPLE)
                .build(),
        )
    }

    override fun find(frame: Bitmap): List<Box> {
        val width = frame.width.toDouble()
        val height = frame.height.toDouble()
        return detector.detect(BitmapImageBuilder(frame).build()).detections().mapNotNull { detection ->
            val box = detection.boundingBox()
            val left = (box.left / width).coerceIn(0.0, 1.0)
            val top = (box.top / height).coerceIn(0.0, 1.0)
            val right = (box.right / width).coerceIn(0.0, 1.0)
            val bottom = (box.bottom / height).coerceIn(0.0, 1.0)
            if (right > left && bottom > top) Box(left, top, right - left, bottom - top) else null
        }
    }

    override fun close() { runCatching { detector.close() } }

    private companion object {
        const val SCORE_THRESHOLD = 0.2f
        const val MAX_PEOPLE = 12
    }
}
