package com.betabuddy.android.capture

import android.icu.text.MeasureFormat
import android.icu.text.NumberFormat
import android.icu.util.Measure
import android.icu.util.MeasureUnit
import androidx.annotation.StringRes
import androidx.compose.runtime.Composable
import androidx.compose.ui.res.stringResource
import com.betabuddy.android.R
import com.betabuddy.android.journal.NearbyGym
import com.betabuddy.android.journal.appLocale
import java.util.Locale

/** Words for what the capture page reports as codes. The page's own text is never shown. */
object CaptureText {
    /** What is wrong, for a code this build knows; the generic line for any other. */
    @StringRes
    fun reason(code: String?): Int = when (CaptureGuidance.Code.of(code)) {
        null -> R.string.capture_reason_other
        CaptureGuidance.Code.NO_WALL -> R.string.capture_reason_no_wall
        CaptureGuidance.Code.TOO_DARK -> R.string.capture_reason_too_dark
        CaptureGuidance.Code.BACKLIT -> R.string.capture_reason_backlit
        CaptureGuidance.Code.BLURRY -> R.string.capture_reason_blurry
        CaptureGuidance.Code.SHAKING -> R.string.capture_reason_shaking
        CaptureGuidance.Code.TILTED -> R.string.capture_reason_tilted
        CaptureGuidance.Code.TOO_CLOSE -> R.string.capture_reason_too_close
        CaptureGuidance.Code.TOO_FAR -> R.string.capture_reason_too_far
        CaptureGuidance.Code.CUT_TOP -> R.string.capture_reason_cut_top
        CaptureGuidance.Code.CUT_BOTTOM -> R.string.capture_reason_cut_bottom
        CaptureGuidance.Code.CUT_LEFT -> R.string.capture_reason_cut_left
        CaptureGuidance.Code.CUT_RIGHT -> R.string.capture_reason_cut_right
        CaptureGuidance.Code.OBLIQUE -> R.string.capture_reason_oblique
        CaptureGuidance.Code.PITCH_HIGH -> R.string.capture_reason_pitch_high
        CaptureGuidance.Code.OCCLUDED -> R.string.capture_reason_occluded
        CaptureGuidance.Code.NO_HOLDS -> R.string.capture_reason_no_holds
        CaptureGuidance.Code.START_UNCLEAR -> R.string.capture_reason_start_unclear
        CaptureGuidance.Code.FINISH_UNCLEAR -> R.string.capture_reason_finish_unclear
    }

    /** What to do about it, or null when the page gave no move or one this build has no words for. */
    @StringRes
    fun action(move: String?): Int? = when (CaptureGuidance.Move.of(move)) {
        null -> null
        CaptureGuidance.Move.BACK -> R.string.capture_move_back
        CaptureGuidance.Move.CLOSER -> R.string.capture_move_closer
        CaptureGuidance.Move.LEFT -> R.string.capture_move_left
        CaptureGuidance.Move.RIGHT -> R.string.capture_move_right
        CaptureGuidance.Move.UP -> R.string.capture_move_up
        CaptureGuidance.Move.DOWN -> R.string.capture_move_down
        CaptureGuidance.Move.TILT_UP -> R.string.capture_move_tilt_up
        CaptureGuidance.Move.TILT_DOWN -> R.string.capture_move_tilt_down
        CaptureGuidance.Move.LEVEL -> R.string.capture_move_level
        CaptureGuidance.Move.HOLD_STILL -> R.string.capture_move_hold_still
    }

    /** Short names for what a tap marks; also the badges on marked holds. */
    @StringRes
    fun role(role: CaptureRole): Int = when (role) {
        CaptureRole.START_HAND -> R.string.capture_role_start_hand
        CaptureRole.START_FOOT -> R.string.capture_role_start_foot
        CaptureRole.FINISH -> R.string.capture_role_finish
        CaptureRole.HOLD -> R.string.capture_role_hold
        CaptureRole.NONE -> R.string.capture_role_none
    }

    /** Why a recording ended by itself; null when the user stopped it. */
    @StringRes
    fun stop(reason: CaptureStop?): Int? = when (reason) {
        CaptureStop.LIMIT -> R.string.capture_stop_limit
        CaptureStop.STORAGE -> R.string.capture_stop_storage
        CaptureStop.INTERRUPTED -> R.string.capture_stop_interrupted
        CaptureStop.OVERHEATED -> R.string.capture_stop_overheated
        CaptureStop.USER, null -> null
    }

    @StringRes
    fun notice(notice: CaptureSession.Notice): Int = when (notice) {
        CaptureSession.Notice.TOO_SHORT -> R.string.capture_notice_too_short
        CaptureSession.Notice.LOW_STORAGE -> R.string.capture_notice_low_storage
        CaptureSession.Notice.RECORDER_FAILED -> R.string.capture_notice_recorder_failed
    }

    /** Where the user is in marking the problem. The screen has a sentence for each. */
    enum class Stage { MISSED, COLOUR, START, ROUTE, FINISH, MARKED }

    /** With a recognised problem the start and finish come from the pack's data, not from a guess at the picture. */
    fun stage(tapMissed: Boolean, colourChosen: Boolean, assessment: CaptureAssessment?, recognised: Boolean = false): Stage = when {
        tapMissed -> Stage.MISSED
        !colourChosen -> Stage.COLOUR
        assessment?.startConfirmed != true -> if (recognised) Stage.ROUTE else Stage.START
        !assessment.finishConfirmed -> Stage.FINISH
        else -> Stage.MARKED
    }

    @StringRes
    fun prompt(stage: Stage): Int = when (stage) {
        Stage.MISSED -> R.string.capture_prompt_missed
        Stage.COLOUR -> R.string.capture_prompt_colour
        Stage.START -> R.string.capture_prompt_start
        Stage.ROUTE -> R.string.capture_prompt_route
        Stage.FINISH -> R.string.capture_prompt_finish
        Stage.MARKED -> R.string.capture_prompt_marked
    }

    /** Five segments for the page's score, never a number: it only orders camera positions. */
    fun level(score: Double): Int = kotlin.math.ceil(score * 5).toInt().coerceIn(1, 5)

    /** A distance in the units and words of [locale]: metres to the nearest ten below a kilometre, kilometres with one decimal above. */
    fun distance(meters: Double, locale: Locale): String {
        val numbers = NumberFormat.getInstance(locale).apply { maximumFractionDigits = 1 }
        val format = MeasureFormat.getInstance(locale, MeasureFormat.FormatWidth.SHORT, numbers)
        return if (meters < 1_000) format.format(Measure(Math.round(meters / 10) * 10, MeasureUnit.METER))
        else format.format(Measure(Math.round(meters / 100) / 10.0, MeasureUnit.KILOMETER))
    }
}

/** The one line of guidance for an issue. */
@Composable
internal fun captureLine(issue: CaptureAssessment.Issue): String {
    val reason = stringResource(CaptureText.reason(issue.code))
    val action = CaptureText.action(issue.move) ?: return reason
    return stringResource(R.string.capture_line_join, reason, stringResource(action))
}

/** How far a suggested gym is, as far as that can honestly be said; null when it cannot. */
@Composable
internal fun captureDistance(gym: NearbyGym): String? {
    val meters = gym.displayMeters ?: return null
    val text = CaptureText.distance(meters, appLocale())
    return if (gym.shown == NearbyGym.Shown.APPROXIMATE) stringResource(R.string.capture_distance_about, text) else text
}
