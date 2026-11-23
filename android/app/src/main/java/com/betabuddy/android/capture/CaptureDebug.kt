package com.betabuddy.android.capture

import com.betabuddy.android.gym.GymPack
import com.betabuddy.android.gym.PackWebView
import com.betabuddy.android.journal.DirectoryGym
import java.io.File

/**
 * Stand-ins for what an emulator or a test has no real version of. Only ever read in a debug
 * build; device tests set them before the screen opens, and a debug build can be started with
 * launch extras (see MainActivity) that fill some of them in.
 */
object CaptureDebug {
    /** A video file played in a loop in place of the camera. */
    @Volatile var video: File? = null
    /** The horizontal field of view to report for that file, degrees. */
    @Volatile var fieldOfView: Double? = null
    /** How the phone is held, for where there are no motion sensors. */
    @Volatile var attitude: CaptureFrameInfo? = null
    /** A position fix, so that nothing is asked and no location service is needed. */
    @Volatile var location: GymLocator.Fix? = null
    /** Any other state of the location (never asked, refused, waiting, no fix) in place of the system's. Wins over [location]. */
    @Volatile var locationState: GymLocator.StandIn? = null
    /** A synthetic pack and where its files are, in place of the one in the app's assets. */
    @Volatile var pack: GymPack? = null
    @Volatile var packFiles: PackWebView.Files? = null
    /** A synthetic gym directory in place of the bundled one. */
    @Volatile var directory: List<DirectoryGym>? = null
    /** Behave as a build without any pack, whatever is in the app's assets. */
    @Volatile var noPack: Boolean = false
    /** Called with each session the screen makes, before it starts: a test shortens its limits here. */
    @Volatile var configure: ((CaptureSession) -> Unit)? = null
    /** Draws what is being told to the page, and how fast it answers, over the picture. Never in a release build. */
    @Volatile var overlay: Boolean = false
    /** Where the last picture handed to the page is also written, to be looked at. */
    @Volatile var keepFrame: File? = null
    /** The source the screen last made, so a test can interrupt it. */
    @Volatile var lastSource: CaptureSource? = null

    fun reset() {
        video = null; fieldOfView = null; attitude = null; location = null; locationState = null; pack = null; packFiles = null; directory = null; lastSource = null; overlay = false; keepFrame = null; noPack = false; configure = null
    }
}
