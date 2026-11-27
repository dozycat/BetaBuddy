package com.betabuddy.android.pose

import com.betabuddy.android.data.SubjectBounds
import kotlin.math.*

/** Pixel geometry for the public tracker model, independent of subject-selection policy. */
internal object RegionTrackerGeometry {
    data class Box(val x: Int, val y: Int, val width: Int, val height: Int) {
        val valid: Boolean get() = width > 0 && height > 0
        fun normalized(width: Int, height: Int): SubjectBounds? {
            val l = x.coerceAtLeast(0); val t = y.coerceAtLeast(0)
            val r = (x+this.width).coerceAtMost(width); val b = (y+this.height).coerceAtMost(height)
            return SubjectBounds(l.toDouble()/width,t.toDouble()/height,(r-l).toDouble()/width,(b-t).toDouble()/height).takeIf { it.valid }
        }
    }
    data class Crop(val x: Int, val y: Int, val side: Int)
    data class Prediction(val box: Box, val confidence: Double)
    fun pixels(bounds: SubjectBounds,width: Int,height: Int): Box = Box(
        floor(bounds.x*width).toInt(),floor(bounds.y*height).toInt(),maxOf(1,floor(bounds.width*width).toInt()),maxOf(1,floor(bounds.height*height).toInt()))
    fun crop(box: Box,factor: Int): Crop {
        require(box.valid)
        val side = ceil(sqrt(box.width.toDouble()*box.height)*factor).toInt().coerceAtLeast(1)
        return Crop(box.x+(box.width-side)/2,box.y+(box.height-side)/2,side)
    }
    private val hann = DoubleArray(16) { .5*(1-cos(2*PI/17*(it+1))) }
    fun prediction(crop: Crop,confidence: FloatArray,sizes: FloatArray,offsets: FloatArray): Prediction? {
        if (confidence.size != 256 || sizes.size != 512 || offsets.size != 512) return null
        if (!confidence.all { it.isFinite() } || !sizes.all { it.isFinite() } || !offsets.all { it.isFinite() }) return null
        val index = confidence.indices.maxBy { confidence[it]*hann[it/16]*hann[it%16] }
        val score = confidence[index]*hann[index/16]*hann[index%16]
        val cx = (index%16+offsets[index])/16.0; val cy = (index/16+offsets[index+256])/16.0
        val w = sizes[index].toDouble(); val h = sizes[index+256].toDouble()
        if (w <= 0 || h <= 0 || w > 2 || h > 2 || score !in 0.0..1.0001) return null
        val box = Box(floor((cx-w/2)*crop.side+crop.x).toInt(),floor((cy-h/2)*crop.side+crop.y).toInt(),floor(w*crop.side).toInt(),floor(h*crop.side).toInt())
        return Prediction(box,score.coerceIn(0.0,1.0)).takeIf { box.valid }
    }
}
