package com.betabuddy.android.review

import org.junit.Assert.assertEquals
import org.junit.Test

class OverlayGeometryTest {
    @Test fun pictureIsFittedInsideTheStage() {
        // Portrait clip in a wide stage: pillarboxed, full height.
        val (w, h) = OverlayGeometry.fit(360f, 340f, 9.0 / 16)
        assertEquals(340f, h, 0.01f)
        assertEquals(191.25f, w, 0.01f)
        // Landscape clip: letterboxed, full width.
        val (w2, h2) = OverlayGeometry.fit(360f, 340f, 16.0 / 9)
        assertEquals(360f, w2, 0.01f)
        assertEquals(202.5f, h2, 0.01f)
        assertEquals(0f to 0f, OverlayGeometry.fit(360f, 340f, Double.NaN))
        assertEquals(0f to 0f, OverlayGeometry.fit(0f, 340f, 1.0))
    }

}
