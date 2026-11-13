package com.betabuddy.android.review

import android.content.Context
import android.net.Uri
import android.view.TextureView
import androidx.annotation.OptIn
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableDoubleStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.media3.common.MediaItem
import androidx.media3.common.PlaybackException
import androidx.media3.common.Player
import androidx.media3.common.VideoSize
import androidx.media3.common.util.UnstableApi
import androidx.media3.exoplayer.ExoPlayer
import androidx.media3.exoplayer.SeekParameters
import com.betabuddy.android.R
import java.io.File

/**
 * The original clip in a Media3 ExoPlayer, played from the app's private file. Main thread only.
 * Counterpart of `Playback` in ios/BetaBuddy/Views/ReviewView.swift.
 */
internal class Playback(context: Context) {
    val player: ExoPlayer = ExoPlayer.Builder(context.applicationContext).build()

    var time by mutableDoubleStateOf(0.0)
        private set
    var playing by mutableStateOf(false)
        private set
    /** String resource of the reason the clip cannot be shown, or null. */
    var failure by mutableStateOf<Int?>(null)
        private set
    /** Width / height of the picture as the player displays it (rotation and pixel shape applied). */
    var videoAspect by mutableStateOf<Double?>(null)
        private set

    private var rate = 1f
    private var loaded = false
    private var prepared = false

    private val listener = object : Player.Listener {
        override fun onEvents(player: Player, events: Player.Events) {
            playing = player.playWhenReady && player.playbackState != Player.STATE_ENDED && player.playbackState != Player.STATE_IDLE
        }
        override fun onPlayerError(error: PlaybackException) {
            failure = R.string.journal_review_video_unplayable
        }
        override fun onSurfaceSizeChanged(width: Int, height: Int) {
            // Decoding starts only once the picture's TextureView has a surface to draw on. Prepared
            // earlier, the paused player shows its first frame to a surface that is not there yet, and
            // the stage stays black until the first step or play.
            if (loaded && !prepared && width > 0 && height > 0) {
                prepared = true
                player.prepare()
            }
        }
        override fun onVideoSizeChanged(videoSize: VideoSize) {
            if (videoSize.width > 0 && videoSize.height > 0) {
                videoAspect = videoSize.width.toDouble() * videoSize.pixelWidthHeightRatio / videoSize.height
            }
        }
    }

    @OptIn(UnstableApi::class)
    fun load(file: File) {
        if (loaded) return
        if (!file.exists()) { failure = R.string.journal_review_video_missing; return }
        loaded = true
        player.addListener(listener)
        player.setSeekParameters(SeekParameters.EXACT) // 0.1 s steps must land on the requested frame
        player.setMediaItem(MediaItem.fromUri(Uri.fromFile(file)))
    }

    fun attach(view: TextureView) { player.setVideoTextureView(view) }
    fun detach(view: TextureView) { runCatching { player.clearVideoTextureView(view) } } // may run after release()

    /** Call regularly while visible; the player does not push its position. */
    fun refreshTime() { if (loaded) time = maxOf(0.0, player.currentPosition / 1000.0) }

    fun toggle(duration: Double) {
        if (playing) { pause(); return }
        if (time >= duration - 0.1) seek(0.0)
        player.setPlaybackSpeed(rate)
        player.play()
    }

    fun seek(value: Double) {
        time = maxOf(0.0, if (value.isFinite()) value else 0.0)
        player.seekTo((time * 1000).toLong())
    }

    fun pause() { player.pause() }

    fun setSlow(slow: Boolean) {
        rate = if (slow) 0.5f else 1f
        player.setPlaybackSpeed(rate)
    }

    fun release() {
        player.removeListener(listener)
        player.release()
    }
}
