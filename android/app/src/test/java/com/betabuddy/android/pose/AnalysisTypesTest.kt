package com.betabuddy.android.pose

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class AnalysisTypesTest {
    @Test fun importLimits() {
        val mb = 1024L * 1024
        assertNull(ImportRules.check(10 * mb, 12.0, true))
        assertNull(ImportRules.check(500 * mb, 180.0, true))
        assertNull(ImportRules.check(1, 0.5, true))
        assertEquals(AnalysisError.TOO_LARGE, ImportRules.check(500 * mb + 1, 12.0, true))
        assertEquals(AnalysisError.TOO_LONG, ImportRules.check(10 * mb, 180.01, true))
        assertEquals(AnalysisError.INVALID_VIDEO, ImportRules.check(10 * mb, 0.49, true))
        assertEquals(AnalysisError.INVALID_VIDEO, ImportRules.check(10 * mb, null, true))
        assertEquals(AnalysisError.INVALID_VIDEO, ImportRules.check(10 * mb, Double.NaN, true))
        assertEquals(AnalysisError.INVALID_VIDEO, ImportRules.check(10 * mb, 12.0, false))
        assertEquals(AnalysisError.INVALID_VIDEO, ImportRules.check(0, 12.0, true))
        // Size is reported first, as on iOS where it is checked while staging the file.
        assertEquals(AnalysisError.TOO_LARGE, ImportRules.check(501 * mb, 999.0, false))
    }

    @Test fun qualityPresetsMatchIos() {
        assertEquals(10.0, AnalysisQuality.BALANCED.fps, 0.0)
        assertEquals(720, AnalysisQuality.BALANCED.maxDimension)
        assertEquals(15.0, AnalysisQuality.DETAILED.fps, 0.0)
        assertEquals(960, AnalysisQuality.DETAILED.maxDimension)
    }

    @Test fun samplerKeepsAtMostTheRequestedRate() {
        val sampler = FrameSampler(10.0)
        val kept = (0 until 30).map { it / 30.0 }.filter { sampler.shouldSample(it) }
        assertEquals(10, kept.size) // one second of 30 fps video, like the iOS smoke test (≤ 11)
        assertEquals(0.0, kept.first(), 0.0)
        assertTrue(kept.zipWithNext().all { (a, b) -> b - a >= 0.099 })
        assertFalse(FrameSampler(10.0).shouldSample(Double.NaN))
        val slow = FrameSampler(15.0) // source slower than the cap: every frame is kept
        assertEquals(5, (0 until 5).count { slow.shouldSample(it / 5.0) })
    }

    @Test fun timestampsAreStrictlyIncreasing() {
        val sampler = FrameSampler(10.0)
        assertEquals(listOf(0L, 1L, 2L, 100L), listOf(0.0, 0.0, 0.0004, 0.1).map { sampler.timestampMs(it) })
    }

    @Test fun displayAndTargetSize() {
        assertEquals(1080 to 1920, FrameGeometry.displaySize(1920, 1080, 90))
        assertEquals(1080 to 1920, FrameGeometry.displaySize(1920, 1080, 270))
        assertEquals(1080 to 1920, FrameGeometry.displaySize(1920, 1080, -90))
        assertEquals(1920 to 1080, FrameGeometry.displaySize(1920, 1080, 180))
        assertEquals(1920 to 1080, FrameGeometry.displaySize(1920, 1080, 45)) // not a right angle: ignored
        assertEquals(405 to 720, FrameGeometry.targetSize(1080, 1920, 720))
        assertEquals(960 to 540, FrameGeometry.targetSize(3840, 2160, 960))
        assertEquals(592 to 1280, FrameGeometry.targetSize(592, 1280, 1280))
        assertEquals(333 to 720, FrameGeometry.targetSize(592, 1280, 720))
        assertEquals(320 to 240, FrameGeometry.targetSize(320, 240, 720)) // never upscaled
    }

    // A 4×2 picture whose four 2×2 blocks have distinct colours, so every orientation is distinguishable
    // (asymmetric content: a 180° mix-up cannot pass, as in the iOS orientation test).
    private fun picture(pixelStride: Int = 1, rowPadding: Int = 0): Triple<YuvConverter.Plane, YuvConverter.Plane, YuvConverter.Plane> {
        val yStride = 4 + rowPadding
        val y = ByteArray(yStride * 2)
        val luma = intArrayOf(40, 40, 200, 200) // left dark, right bright
        for (row in 0 until 2) for (col in 0 until 4) y[row * yStride + col] = luma[col].toByte()
        val cStride = 2 * pixelStride + rowPadding
        val u = ByteArray(cStride)
        val v = ByteArray(cStride)
        u[0] = 128.toByte(); v[0] = 128.toByte()                              // left block: grey
        u[pixelStride] = 90.toByte(); v[pixelStride] = 220.toByte()           // right block: reddish
        return Triple(YuvConverter.Plane(y, yStride, 1), YuvConverter.Plane(u, cStride, pixelStride), YuvConverter.Plane(v, cStride, pixelStride))
    }

    private fun convert(rotation: Int, outWidth: Int, outHeight: Int, pixelStride: Int = 1, rowPadding: Int = 0): IntArray {
        val (y, u, v) = picture(pixelStride, rowPadding)
        val out = IntArray(outWidth * outHeight)
        YuvConverter.convert(y, u, v, 0, 0, 4, 2, rotation, outWidth, outHeight, bt709 = false, fullRange = true, out = out)
        return out
    }

    @Test fun conversionProducesExpectedColours() {
        val out = convert(0, 4, 2)
        val dark = out[0]
        val bright = out[3]
        assertEquals(0xFF282828.toInt(), dark) // grey: R = G = B = luma
        assertEquals(0xFF, bright ushr 24)
        assertTrue("red dominates", (bright shr 16 and 0xFF) > (bright shr 8 and 0xFF) && (bright shr 16 and 0xFF) > (bright and 0xFF))
        assertArrayEquals(intArrayOf(dark, dark, bright, bright, dark, dark, bright, bright), out)
    }

    @Test fun rotationMatchesPlaybackOrientation() {
        val upright = convert(0, 4, 2)
        val dark = upright[0]
        val bright = upright[3]
        // 90° clockwise: the left (dark) half of the coded picture ends up at the top.
        assertArrayEquals(intArrayOf(dark, dark, dark, dark, bright, bright, bright, bright), convert(90, 2, 4))
        // 270°: the left half ends up at the bottom.
        assertArrayEquals(intArrayOf(bright, bright, bright, bright, dark, dark, dark, dark), convert(270, 2, 4))
        // 180°: left and right swap.
        assertArrayEquals(intArrayOf(bright, bright, dark, dark, bright, bright, dark, dark), convert(180, 4, 2))
    }

    @Test fun rotationIsExactForAnAsymmetricLumaPattern() {
        // 2×2 chroma-neutral picture with four different luma values: a b / c d.
        val y = YuvConverter.Plane(byteArrayOf(10, 60, 110, (160).toByte()), 2, 1)
        val c = YuvConverter.Plane(byteArrayOf(128.toByte()), 1, 1)
        fun run(rotation: Int): List<Int> {
            val out = IntArray(4)
            YuvConverter.convert(y, c, c, 0, 0, 2, 2, rotation, 2, 2, bt709 = true, fullRange = true, out = out)
            return out.map { it and 0xFF }
        }
        assertEquals(listOf(10, 60, 110, 160), run(0))
        assertEquals(listOf(110, 10, 160, 60), run(90))   // c a / d b
        assertEquals(listOf(160, 110, 60, 10), run(180))  // d c / b a
        assertEquals(listOf(60, 160, 10, 110), run(270))  // b d / a c
    }

    @Test fun stridesAndCropAreRespected() {
        assertArrayEquals(convert(0, 4, 2), convert(0, 4, 2, pixelStride = 2, rowPadding = 3))
        val (y, u, v) = picture()
        val out = IntArray(4)
        YuvConverter.convert(y, u, v, 2, 0, 2, 2, 0, 2, 2, bt709 = false, fullRange = true, out = out) // right half only
        val bright = convert(0, 4, 2)[3]
        assertArrayEquals(intArrayOf(bright, bright, bright, bright), out)
    }

    @Test fun videoRangeIsExpandedAndDownscaleAverages() {
        val c = YuvConverter.Plane(ByteArray(4) { 128.toByte() }, 2, 1)
        val out = IntArray(1)
        YuvConverter.convert(YuvConverter.Plane(byteArrayOf(16, 16, 16, 16), 2, 1), c, c, 0, 0, 2, 2, 0, 1, 1, false, false, out)
        assertEquals(0xFF000000.toInt(), out[0])
        YuvConverter.convert(YuvConverter.Plane(ByteArray(4) { 235.toByte() }, 2, 1), c, c, 0, 0, 2, 2, 0, 1, 1, true, false, out)
        assertEquals(0xFFFFFFFF.toInt(), out[0])
        // Shrinking 2× averages the 2×2 luma block instead of picking one sample.
        YuvConverter.convert(YuvConverter.Plane(byteArrayOf(0, 100, 100, (200).toByte()), 2, 1), c, c, 0, 0, 2, 2, 0, 1, 1, false, true, out)
        assertEquals(100, out[0] and 0xFF)
    }
}
