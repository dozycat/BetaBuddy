package com.betabuddy.android.journal

import com.betabuddy.android.data.Climb
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * Whether the in-app camera is framing or recording. Nothing is analysed meanwhile: pose
 * recognition gives way to the camera, and the pack's analysis page is not run. Requests stay in
 * the archive and are picked up again when the camera lets go. One per process, like the store.
 */
object CameraGate {
    private val state = MutableStateFlow(false)
    val open: StateFlow<Boolean> = state.asStateFlow()

    fun set(open: Boolean) {
        state.value = open
    }
}

/**
 * Which record the pack's analysis page should work on next, for records that asked for it when
 * they were saved: the ones made with the in-app camera. Pose recognition comes first; a record
 * still waiting for it is left alone. Pure, so the order of things can be tested on a JVM.
 */
object AnalysisQueuePolicy {
    sealed interface Step {
        /** Nothing to do now. */
        data object Wait : Step
        /** Run the analysis page for this record. */
        data class Run(val climb: Climb) : Step
        /** The request is done with and is cleared; the record keeps its own "analyse this attempt". */
        data class Settle(val id: String) : Step
    }

    /**
     * @param results ids of records that already have a saved analysis result
     * @param settled ids not to try again in this run of the app
     * @param paused the camera is open, or the app is not in front
     * @param hasPage the pack in this build has an analysis page
     */
    fun next(climbs: List<Climb>, results: Set<String>, settled: Set<String>, paused: Boolean, hasPage: Boolean): Step {
        if (paused) return Step.Wait
        for (climb in climbs) {
            if (climb.capture?.analysisQueued != true || climb.pendingAnalysis != null || climb.id in settled) continue
            // Already analysed, no page to do it, or pose recognition found nobody (or was stopped).
            if (!hasPage || climb.id in results || !climb.analysis.hasWorldLandmarks) return Step.Settle(climb.id)
            return Step.Run(climb)
        }
        return Step.Wait
    }
}
