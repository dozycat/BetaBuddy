package com.betabuddy.android.pose

import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/** The model ships inside the app; it is never downloaded. This pins the exact file. */
class PoseModelTest {
    // Gradle runs unit tests with the module directory (android/app) as the working directory.
    private val model = File("src/main/assets/${PoseModel.ASSET}")

    @Test fun bundledFullModelIsThePinnedFile() {
        assertTrue("Missing ${model.path}", model.isFile)
        assertEquals(PoseModel.BYTES, model.length())
        assertEquals(PoseModel.SHA256, PoseModel.sha256(model.readBytes()))
    }

    @Test fun hashIsLowercaseHex() {
        assertEquals("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", PoseModel.sha256(ByteArray(0)))
    }
}
