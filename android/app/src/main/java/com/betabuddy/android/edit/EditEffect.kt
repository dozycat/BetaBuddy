package com.betabuddy.android.edit

import android.content.Context
import android.opengl.GLES20
import androidx.annotation.OptIn
import androidx.media3.common.VideoFrameProcessingException
import androidx.media3.common.util.GlUtil
import androidx.media3.common.util.Size
import androidx.media3.common.util.UnstableApi
import androidx.media3.effect.BaseGlShaderProgram
import androidx.media3.effect.GlEffect
import androidx.media3.effect.GlShaderProgram
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * The clean plate: the part of the picture that passers-by cross, with everything that moves
 * taken out. [pixels] is RGBA with the colour premultiplied by alpha (a pixel is either known, alpha
 * 255, or not, all zero), [width] × [height], rows bottom to top (as OpenGL reads and writes
 * them); [left] and [bottom] place it in the rendered frame, in pixels from the bottom-left corner.
 */
internal class Plate(val left: Int, val bottom: Int, val width: Int, val height: Int, val pixels: ByteBuffer)

/** What the picture of one clip is made from, besides the frame itself. */
internal class EditLook(
    /** Where the climber is; these boxes are never replaced. */
    val keep: List<KeepSample> = emptyList(),
    /** The people being taken out: only those who can be, for all the time they are seen ([Passers]). */
    val removed: List<PersonTrack> = emptyList(),
    /** Soft edge of a replaced region, as a fraction of the frame's short side. */
    val feather: Double = 0.0,
    val plate: Plate? = null,
)

/**
 * Draws one clip of the edit: every frame is shown through the plan's camera window for its
 * source moment, and where a passer-by region lies outside the keep boxes the plate is shown
 * instead of the frame.
 *
 * The effect is told nothing about time by the pipeline except each frame's timestamp, which
 * counts in output time from wherever the pipeline put this clip. So the first frame an instance
 * sees is taken as the clip's first ([EditClip.t0]) and every later one is placed by how much
 * output time has passed since, times the clip's speed. One instance per clip.
 *
 * That anchoring depends on the frame processor building a new shader program for every clip, which
 * it does when a clip's list of effects differs from the one before. So every clip must be given
 * its own instance of this class, and this class must not define `equals`: two clips with equal
 * effects would share one program, and the second would be timed from the first one's first frame.
 *
 * [afterDraw] runs after each frame is drawn, with its source time and the size of the bound
 * framebuffer, on the GL thread: the passes that only look at the clip read pixels there.
 */
@OptIn(UnstableApi::class)
internal class EditFrameEffect(
    private val clip: EditClip,
    private val look: EditLook = EditLook(),
    private val afterDraw: ((time: Double, width: Int, height: Int) -> Unit)? = null,
) : GlEffect {
    override fun toGlShaderProgram(context: Context, useHdr: Boolean): GlShaderProgram = try {
        EditShaderProgram(clip, look, afterDraw, useHdr)
    } catch (error: GlUtil.GlException) {
        throw VideoFrameProcessingException(error)
    }
}

@OptIn(UnstableApi::class)
private class EditShaderProgram(
    private val clip: EditClip,
    private val look: EditLook,
    private val afterDraw: ((Double, Int, Int) -> Unit)?,
    useHdr: Boolean,
) : BaseGlShaderProgram(/* useHighPrecisionColorComponents = */ useHdr, /* texturePoolCapacity = */ 1) {
    private val program: Int
    private val position: Int
    private val vertices: FloatBuffer
    private var plateTexture = 0
    private var width = 0
    private var height = 0
    private var firstTimeUs = Long.MIN_VALUE
    private val keepValues = FloatArray(Passers.MAX_BOXES * 4)
    private val passValues = FloatArray(Passers.MAX_BOXES * 4)

    init {
        program = link(VERTEX, FRAGMENT)
        position = GLES20.glGetAttribLocation(program, "aPosition")
        vertices = ByteBuffer.allocateDirect(8 * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
            .apply { put(floatArrayOf(-1f, -1f, 1f, -1f, -1f, 1f, 1f, 1f)); rewind() }
        GlUtil.checkGlError()
    }

    override fun configure(inputWidth: Int, inputHeight: Int): Size {
        width = inputWidth
        height = inputHeight
        return Size(inputWidth, inputHeight)
    }

    override fun drawFrame(inputTexId: Int, presentationTimeUs: Long) {
        try {
            if (firstTimeUs == Long.MIN_VALUE) firstTimeUs = presentationTimeUs
            val time = EditTimeline.sourceTime(clip, (presentationTimeUs - firstTimeUs) / 1_000_000.0)
            val window = CameraPath.window(clip.camera, time, clip.cameraLinear)
            val plate = look.plate
            val passers = if (plate == null) emptyList() else Passers.at(look.removed, time)
            val keep = if (passers.isEmpty()) emptyList() else Passers.limit(KeepPath.boxes(look.keep, time), Passers.MAX_BOXES)

            GLES20.glUseProgram(program)
            GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, inputTexId)
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)
            GLES20.glUniform1i(uniform("uFrame"), 0)
            GLES20.glActiveTexture(GLES20.GL_TEXTURE1)
            // Without a plate the sampler still needs a texture; the frame itself is never mixed in from it.
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, if (plate != null && passers.isNotEmpty()) plateTexture(plate) else inputTexId)
            GLES20.glUniform1i(uniform("uPlate"), 1)

            GLES20.glUniform3f(uniform("uWindow"), window.cx.toFloat(), window.cy.toFloat(), (1 / window.s).toFloat())
            val short = minOf(width, height).toFloat()
            val feather = maxOf(look.feather.toFloat(), MIN_FEATHER)
            GLES20.glUniform2f(uniform("uFeather"), feather * short / width, feather * short / height)
            if (plate != null) {
                GLES20.glUniform4f(uniform("uPlateRect"), plate.left.toFloat() / width, plate.bottom.toFloat() / height, plate.width.toFloat() / width, plate.height.toFloat() / height)
            }
            GLES20.glUniform1i(uniform("uPassCount"), fill(passValues, passers))
            GLES20.glUniform4fv(uniform("uPass"), Passers.MAX_BOXES, passValues, 0)
            GLES20.glUniform1i(uniform("uKeepCount"), fill(keepValues, keep))
            GLES20.glUniform4fv(uniform("uKeep"), Passers.MAX_BOXES, keepValues, 0)

            GLES20.glEnableVertexAttribArray(position)
            GLES20.glVertexAttribPointer(position, 2, GLES20.GL_FLOAT, false, 0, vertices)
            GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4)
            GLES20.glDisableVertexAttribArray(position)
            GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
            GlUtil.checkGlError()
            afterDraw?.invoke(time, width, height)
        } catch (error: GlUtil.GlException) {
            throw VideoFrameProcessingException(error, presentationTimeUs)
        } catch (error: VideoFrameProcessingException) {
            throw error
        } catch (error: Exception) {
            // Whatever the pass that reads the frames ran into (the detector, a full disk): the export ends with it.
            throw VideoFrameProcessingException(error, presentationTimeUs)
        }
    }

    override fun release() {
        super.release()
        if (plateTexture != 0) GLES20.glDeleteTextures(1, intArrayOf(plateTexture), 0)
        GLES20.glDeleteProgram(program)
    }

    private fun uniform(name: String) = GLES20.glGetUniformLocation(program, name)

    private fun fill(target: FloatArray, boxes: List<Box>): Int {
        val count = minOf(boxes.size, Passers.MAX_BOXES)
        for (index in 0 until count) {
            val box = boxes[index]
            target[index * 4] = box.x.toFloat()
            target[index * 4 + 1] = box.y.toFloat()
            target[index * 4 + 2] = box.w.toFloat()
            target[index * 4 + 3] = box.h.toFloat()
        }
        return count
    }

    private fun plateTexture(plate: Plate): Int {
        if (plateTexture != 0) return plateTexture
        val ids = IntArray(1)
        GLES20.glGenTextures(1, ids, 0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, ids[0])
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glPixelStorei(GLES20.GL_UNPACK_ALIGNMENT, 1)
        // Rows go in as they were read out, bottom first, so the plate lies exactly where its pixels came from.
        GLES20.glTexImage2D(GLES20.GL_TEXTURE_2D, 0, GLES20.GL_RGBA, plate.width, plate.height, 0, GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, plate.pixels.duplicate().apply { rewind() })
        GlUtil.checkGlError()
        plateTexture = ids[0]
        return plateTexture
    }

    private fun link(vertexSource: String, fragmentSource: String): Int {
        fun compile(type: Int, source: String): Int {
            val shader = GLES20.glCreateShader(type)
            GLES20.glShaderSource(shader, source)
            GLES20.glCompileShader(shader)
            val status = IntArray(1)
            GLES20.glGetShaderiv(shader, GLES20.GL_COMPILE_STATUS, status, 0)
            if (status[0] == 0) {
                val log = GLES20.glGetShaderInfoLog(shader)
                GLES20.glDeleteShader(shader)
                throw GlUtil.GlException("shader: $log")
            }
            return shader
        }
        val vertex = compile(GLES20.GL_VERTEX_SHADER, vertexSource)
        val fragment = compile(GLES20.GL_FRAGMENT_SHADER, fragmentSource)
        val linked = GLES20.glCreateProgram()
        GLES20.glAttachShader(linked, vertex)
        GLES20.glAttachShader(linked, fragment)
        GLES20.glLinkProgram(linked)
        GLES20.glDeleteShader(vertex)
        GLES20.glDeleteShader(fragment)
        val status = IntArray(1)
        GLES20.glGetProgramiv(linked, GLES20.GL_LINK_STATUS, status, 0)
        if (status[0] == 0) {
            val log = GLES20.glGetProgramInfoLog(linked)
            GLES20.glDeleteProgram(linked)
            throw GlUtil.GlException("program: $log")
        }
        return linked
    }

    private companion object {
        /** A hard edge would show as a line; this much is always blended. */
        const val MIN_FEATHER = 0.004f

        const val VERTEX = """
            attribute vec2 aPosition;
            varying vec2 vTex;
            void main() {
              gl_Position = vec4(aPosition, 0.0, 1.0);
              vTex = aPosition * 0.5 + 0.5;
            }
        """

        // Frame textures have their first row at the bottom. `p` is the source position with the
        // origin top-left (as the plan counts), `uv` the same point as a texture coordinate.
        const val FRAGMENT = """
            #ifdef GL_FRAGMENT_PRECISION_HIGH
            precision highp float;
            #else
            precision mediump float;
            #endif
            varying vec2 vTex;
            uniform sampler2D uFrame;
            uniform sampler2D uPlate;
            uniform vec3 uWindow;
            uniform vec2 uFeather;
            uniform vec4 uPlateRect;
            uniform int uPassCount;
            uniform vec4 uPass[8];
            uniform int uKeepCount;
            uniform vec4 uKeep[8];

            float cover(vec2 p, vec4 box) {
              vec2 d = max(abs(p - (box.xy + box.zw * 0.5)) - box.zw * 0.5, 0.0) / uFeather;
              return 1.0 - smoothstep(0.0, 1.0, length(d));
            }

            void main() {
              vec2 p = uWindow.xy + (vec2(vTex.x, 1.0 - vTex.y) - 0.5) * uWindow.z;
              vec2 uv = vec2(p.x, 1.0 - p.y);
              vec4 live = texture2D(uFrame, uv);
              float pass = 0.0;
              for (int i = 0; i < 8; i++) { if (i < uPassCount) pass = max(pass, cover(p, uPass[i])); }
              if (pass > 0.0) {
                float keep = 0.0;
                for (int i = 0; i < 8; i++) { if (i < uKeepCount) keep = max(keep, cover(p, uKeep[i])); }
                vec2 q = (uv - uPlateRect.xy) / uPlateRect.zw;
                // Outside the plate there is nothing to put in the frame's place.
                float inside = step(0.0, q.x) * step(q.x, 1.0) * step(0.0, q.y) * step(q.y, 1.0);
                // The plate's colour is premultiplied by its alpha; where it has none the frame stays as it is.
                vec4 plate = texture2D(uPlate, q);
                float weight = pass * (1.0 - keep) * inside;
                live.rgb = live.rgb * (1.0 - weight * plate.a) + plate.rgb * weight;
              }
              gl_FragColor = live;
            }
        """
    }
}
