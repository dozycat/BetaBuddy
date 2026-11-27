package com.betabuddy.android.pose

import com.betabuddy.android.data.SubjectBounds
import java.io.File
import org.junit.Assert.*
import org.junit.Test

class SubjectTrackerTest {
    @Test fun trackerModelIsPinnedAndBundled() {
        val model = File("src/main/assets/${SubjectTrackerModel.ASSET}")
        assertEquals(SubjectTrackerModel.BYTES,model.length())
        assertEquals(SubjectTrackerModel.SHA256,PoseModel.sha256(model.readBytes()))
    }
    @Test fun outputGeometryUsesBothAxesAndRejectsBrokenResults() {
        val box = RegionTrackerGeometry.pixels(SubjectBounds(.2,.3,.2,.4),200,100)
        assertEquals(RegionTrackerGeometry.Box(40,30,40,40),box)
        val crop = RegionTrackerGeometry.crop(box,4)
        assertEquals(RegionTrackerGeometry.Crop(-20,-30,160),crop)
        val conf = FloatArray(256);val sizes = FloatArray(512); val offsets = FloatArray(512)
        val index = 8*16+7
        conf[index] = 1f; sizes[index]=.25f;sizes[index+256]=.5f;offsets[index]=.5f;offsets[index+256]=.25f
        val predicted = RegionTrackerGeometry.prediction(crop,conf,sizes,offsets)!!
        assertEquals(RegionTrackerGeometry.Box(35,12,40,80),predicted.box)
        assertTrue(predicted.confidence > .9)
        sizes[index] = Float.NaN;assertNull(RegionTrackerGeometry.prediction(crop,conf,sizes,offsets))
        assertNull(RegionTrackerGeometry.Box(-50,0,10,10).normalized(100,100))
        assertEquals(SubjectBounds(0.0,0.0,.1,.1),RegionTrackerGeometry.Box(-10,-10,20,20).normalized(100,100))
    }
}
