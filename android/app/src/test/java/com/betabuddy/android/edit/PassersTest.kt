package com.betabuddy.android.edit

import java.io.File
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

// Who is replaced by the plate, and what the plate is. All boxes are made up.
class PassersTest {
    private val climber = Box(0.40, 0.30, 0.20, 0.40)
    private val walker = Box(0.05, 0.60, 0.08, 0.30)

    private fun grey(value: Int) = (value shl 16) or (value shl 8) or value

    /** Somebody at [x] at time [t], with a ring of [around] wherever the ring is inside the picture. */
    private fun sighting(t: Double, x: Double, y: Double = 0.60, around: Int = 128): Sighting {
        val box = Box(x, y, 0.08, 0.30)
        val ring = Passers.ringPoints(Passers.region(box)).map { (px, py) -> if (px < 0 || py < 0 || px >= 1 || py >= 1) -1 else grey(around) }
        return Sighting(t, box, ring.toIntArray())
    }

    /** Somebody walking from the left edge to the right one, looked at every 0.12 s, except at [missed]. */
    private fun walk(y: Double = 0.60, missed: Set<Int> = emptySet(), around: Int = 128): List<List<Sighting>> =
        (0..23).filter { it !in missed }.map { step -> listOf(sighting(step * 0.12, 0.04 * step, y, around)) }

    /** A plate that is a plain grey wall, except that it knows nothing inside [hole]. */
    private class Wall(val value: Int = 128, val hole: Box? = null) : PlateView {
        override fun covers(region: Box) = hole == null || !hole.intersects(region)
        override fun colour(x: Double, y: Double): Int? = if (x < 0 || y < 0 || x >= 1 || y >= 1 || hole?.intersects(Box(x, y, 1e-6, 1e-6)) == true) null else (value shl 16) or (value shl 8) or value
    }

    /** What the renderer does with the people it found: who is taken out, given the keep boxes and the plate. */
    private fun removed(samples: List<List<Sighting>>, keep: List<KeepSample>, plate: PlateView, until: Double = 2.76): List<PersonTrack> {
        val everybody = Passers.link(samples)
        return Passers.removable(everybody, Passers.candidates(everybody, keep, 0.0, until), plate, keep)
    }

    private val grid = (-10..80).map { it * 0.04 }

    @Test fun aCleanTrackIsRemovedInEveryFrameIncludingAcrossAMissedDetection() {
        val keep = listOf(KeepSample(0.0, listOf(Box(0.45, 0.05, 0.15, 0.30))))   // the climber is up the wall, away from the walker
        val samples = walk(missed = setOf(11))
        val tracks = Passers.link(samples)
        assertEquals("one person, one track, the missed look bridged", 1, tracks.size)
        assertEquals(23, tracks.single().sightings.size)
        val gone = removed(samples, keep, Wall())
        assertEquals(1, gone.size)
        // Covered at every moment from their first sighting to their last, the gap included…
        for (time in grid.filter { it in 0.0..2.76 }) assertEquals("at $time", 1, Passers.at(gone, time).size)
        // …where the region is between the sightings on either side of the gap…
        val before = Passers.at(gone, 10 * 0.12).single(); val inGap = Passers.at(gone, 11 * 0.12).single(); val after = Passers.at(gone, 12 * 0.12).single()
        assertEquals((before.x + after.x) / 2, inGap.x, 1e-9)
        // …and a little before and after, but not for long.
        assertEquals(1, Passers.at(gone, -0.2).size); assertEquals(1, Passers.at(gone, 2.76 + 0.2).size)
        assertTrue(Passers.at(gone, -0.4).isEmpty()); assertTrue(Passers.at(gone, 2.76 + 0.4).isEmpty())
        // The region is the person's box, grown, and never leaves the picture.
        val region = Passers.at(gone, 0.6).single()
        val box = Passers.box(gone.single(), 0.6)
        assertTrue(region.x < box.x && region.right > box.right && region.y < box.y && region.bottom > box.bottom)
        assertEquals(0.0, Passers.at(gone, 0.0).single().x, 0.0)
    }

    @Test fun aTrackThatTouchesTheKeepBoxAtAnyMomentStaysForAllOfIt() {
        // The climber stands where the walker passes half way through.
        val keep = listOf(KeepSample(0.0, listOf(Box(0.45, 0.50, 0.12, 0.40))))
        val samples = walk()
        assertTrue(Passers.clearOfKeep(Passers.link(listOf(samples.first())).single(), keep))   // far from the climber at first…
        val gone = removed(samples, keep, Wall())
        assertTrue("…but not later, so nobody is taken out at any time", gone.isEmpty())
        for (time in grid) assertTrue(Passers.at(gone, time).isEmpty())
        // Somebody else, who never comes near the climber, is still taken out.
        val both = samples.zip(walk(y = 0.05)) { a, b -> a + b }
        val one = removed(both, keep, Wall())
        assertEquals(1, one.size)
        assertEquals(0.05, one.single().sightings.first().box.y, 0.0)
        // The climber moving away afterwards changes nothing about the moment they did touch.
        val moving = listOf(KeepSample(0.0, listOf(Box(0.45, 0.50, 0.12, 0.40))), KeepSample(2.0, listOf(Box(0.45, 0.05, 0.12, 0.30))))
        assertTrue(removed(samples, moving, Wall()).isEmpty())
        // Touching counts between two looks as well: a walker seen only on either side of the climber.
        val sparse = listOf(listOf(sighting(0.0, 0.0)), listOf(sighting(0.9, 0.7)))
        assertEquals(1, Passers.link(sparse).size)
        assertTrue(removed(sparse, keep, Wall(), until = 0.9).isEmpty())
    }

    @Test fun aTrackOverPlateThatIsNotKnownStaysForAllOfIt() {
        val keep = listOf(KeepSample(0.0, listOf(Box(0.45, 0.05, 0.15, 0.30))))
        val samples = walk()
        // The plate has no pixels in a patch the walker crosses.
        val gone = removed(samples, keep, Wall(hole = Box(0.50, 0.70, 0.05, 0.05)))
        assertTrue(gone.isEmpty())
        for (time in grid) assertTrue(Passers.at(gone, time).isEmpty())
        // A hole they never come near does not matter.
        assertEquals(1, removed(samples, keep, Wall(hole = Box(0.50, 0.05, 0.05, 0.05))).size)
    }

    @Test fun aPlateThatNoLongerShowsTheSceneIsNotUsed() {
        val keep = emptyList<KeepSample>()
        // What the frames show around the walker is much brighter than the plate: the light changed.
        assertTrue(removed(walk(around = 200), keep, Wall(128)).isEmpty())
        // A little brighter is the same scene.
        assertEquals(1, removed(walk(around = 150), keep, Wall(128)).size)
        // One sighting out of step is enough to leave the person alone for the whole track.
        val oneOff = walk().mapIndexed { index, sample -> if (index == 12) listOf(sighting(index * 0.12, 0.04 * index, around = 220)) else sample }
        assertTrue(removed(oneOff, keep, Wall(128)).isEmpty())
        // A few points of the ring may differ (something else is next to them), most may not.
        fun partly(count: Int) = walk().mapIndexed { index, sample ->
            if (index != 12) sample else sample.map { seen -> Sighting(seen.t, seen.box, seen.ring!!.copyOf().also { ring -> for (point in 0 until count) ring[point] = grey(250) }) }
        }
        assertEquals(1, removed(partly(6), keep, Wall(128)).size)
        assertTrue(removed(partly(12), keep, Wall(128)).isEmpty())
        // Without anything seen around them there is nothing to compare, so nothing is done.
        assertTrue(removed(walk().map { sample -> sample.map { Sighting(it.t, it.box, null) } }, keep, Wall(128)).isEmpty())
        assertEquals(0, Passers.channelsApart(0x102030, 0x102030)); assertEquals(0x30, Passers.channelsApart(0x102030, 0x105030))
    }

    @Test fun somebodyLostInTheMiddleOfThePictureIsNotTakenOutForTheTimeTheyWereSeen() {
        val keep = emptyList<KeepSample>()
        // Seen only from the middle of the clip, in the middle of the picture: they would appear from nowhere.
        val late = (10..23).map { step -> listOf(sighting(step * 0.12, 0.04 * step)) }
        assertTrue(removed(late, keep, Wall()).isEmpty())
        // Lost in the middle of the picture well before the end: they would vanish and come back.
        val early = (0..10).map { step -> listOf(sighting(step * 0.12, 0.04 * step)) }
        assertTrue(removed(early, keep, Wall()).isEmpty())
        // Seen from the start of the looked-at stretch to its end, wherever they stand: the whole of it.
        val standing = (0..23).map { step -> listOf(sighting(step * 0.12, 0.40 + 0.005 * step)) }
        assertEquals(1, removed(standing, keep, Wall()).size)
        // Walking in at the edge after the start and out at the edge before the end.
        val through = (5..18).map { step -> listOf(sighting(step * 0.12, (step - 5) * 0.92 / 13)) }
        assertEquals(1, removed(through, keep, Wall()).size)
    }

    @Test fun peopleAreFollowedFromOneLookToTheNext() {
        // Two people walking at different heights are two tracks, each with all its sightings.
        val two = walk(y = 0.60).zip(walk(y = 0.05)) { a, b -> b + a }
        val tracks = Passers.link(two)
        assertEquals(2, tracks.size)
        assertTrue(tracks.all { track -> track.sightings.size == 24 && track.sightings.map { it.box.y }.distinct().size == 1 })
        // Not found for longer than a second: somebody new, as far as anyone can tell.
        val apart = listOf(listOf(sighting(0.0, 0.2)), listOf(sighting(1.5, 0.2)))
        assertEquals(2, Passers.link(apart).size)
        // Found again within the second, but far from where they could have got to: also somebody new.
        assertEquals(2, Passers.link(listOf(listOf(sighting(0.0, 0.1)), listOf(sighting(0.12, 0.7)))).size)
        // A walker passing the spot where somebody else was last seen a moment ago stays their own track,
        // even when looks are irregular and the walker is fast.
        val someoneElse = (0..4).map { step -> sighting(step * 0.12, 0.46, y = 0.62) }
        val fast = listOf(0.0, 0.13, 0.30, 0.41, 0.58, 0.70, 0.85, 0.97, 1.12).map { time -> sighting(time, time * 0.9) }
        val mixed = (someoneElse + fast).groupBy { it.t }.values.toList()
        val crossing = Passers.link(mixed)
        assertEquals(2, crossing.size)
        assertEquals(fast.map { it.t }, crossing.single { it.sightings.size == fast.size }.sightings.map { it.t })
        // Somebody who stood still and is then lost is not the person who turns up two bodies away a moment later.
        val stood = (0..10).map { step -> listOf(sighting(step * 0.12, 0.10)) } + listOf(listOf(sighting(17 * 0.12, 0.60)))
        assertEquals(2, Passers.link(stood).size)
        // Looks may arrive in any order, and a look that found nobody is no look at all.
        assertEquals(1, Passers.link(walk().reversed() + listOf(emptyList())).size)
        assertTrue(Passers.link(emptyList()).isEmpty())
    }

    /** Several people's looks at the same moments, as one list of looks. */
    private fun together(vararg people: List<List<Sighting>>): List<List<Sighting>> = people.flatMap { it }.flatten().groupBy { it.t }.toSortedMap().values.toList()

    @Test fun nobodyIsTakenOutWhoseRegionEverLiesOverSomebodyWhoStays() {
        val keep = listOf(KeepSample(0.0, listOf(Box(0.45, 0.05, 0.15, 0.20))))
        // Somebody stands in the walker's way, seen only for the middle of the clip (so they stay: lost mid-picture).
        val standing = (8..14).map { step -> listOf(sighting(step * 0.12, 0.40, y = 0.62)) }
        val samples = together(walk(), standing)
        assertEquals(2, Passers.link(samples).size)
        // Alone, the walker would be taken out; walking behind the one who stays, they are left alone for all of it.
        assertEquals(1, removed(walk(), keep, Wall()).size)
        val gone = removed(samples, keep, Wall())
        assertTrue(gone.isEmpty())
        for (time in grid) assertTrue(Passers.at(gone, time).isEmpty())
        // The one who stays is taken to be there a while before and after the detector saw them…
        // (seen once at 0.36 s where the walker arrives at 0.84 s)
        val briefly = together(walk(), listOf(listOf(sighting(3 * 0.12, 0.28, y = 0.62))))
        assertEquals(2, Passers.link(briefly).size)
        assertTrue(removed(briefly, keep, Wall()).isEmpty())
        // …though not for ever: seen once where the walker only arrives two seconds later, they are taken to have gone.
        val longAgo = together(walk(), listOf(listOf(sighting(3 * 0.12, 0.84, y = 0.62))))
        assertEquals(1, removed(longAgo, keep, Wall()).size)
        // …but somebody standing where the walker never comes changes nothing.
        val apart = together(walk(), (8..14).map { step -> listOf(sighting(step * 0.12, 0.40, y = 0.02)) })
        assertEquals(1, removed(apart, keep, Wall()).size)

        // The one who stays because they touched the climber once: the same.
        val belayer = (0..23).map { step -> listOf(sighting(step * 0.12, 0.44, y = 0.30)) }
        assertFalse(Passers.clearOfKeep(Passers.link(belayer).single(), keep))
        val crossing = together(walk(y = 0.40), belayer)
        assertTrue(removed(crossing, keep, Wall()).isEmpty())
    }

    @Test fun leavingOnePersonAloneCanMeanLeavingTheNextOneToo() {
        val keep = emptyList<KeepSample>()
        // C stays (seen only mid-clip, mid-picture). B, standing, overlaps C. A, standing, overlaps B but not C.
        val c = (8..14).map { step -> listOf(sighting(step * 0.12, 0.50, y = 0.60)) }
        val b = (0..23).map { step -> listOf(sighting(step * 0.12, 0.40, y = 0.60)) }
        val a = (0..23).map { step -> listOf(sighting(step * 0.12, 0.30, y = 0.60)) }
        val far = (0..23).map { step -> listOf(sighting(step * 0.12, 0.80, y = 0.05)) }
        assertEquals("A and B alone are both taken out", 2, removed(together(a, b), keep, Wall()).size)
        // With C staying, B must stay; with B staying, A must stay. Only the one far from all of them goes.
        val gone = removed(together(a, b, c, far), keep, Wall())
        assertEquals(1, gone.size)
        assertEquals(0.80, gone.single().sightings.first().box.x, 0.0)
    }

    @Test fun moreRemovablePeopleThanTheShaderTakesAreLeftAloneNotMergedIntoOneRegion() {
        val keep = emptyList<KeepSample>()
        // Ten people standing apart along the top and bottom edges, each removable on their own. The shader takes eight regions.
        fun person(index: Int, steps: IntRange) = steps.map { step -> listOf(sighting(step * 0.12, 0.02 + (index / 2) * 0.19, y = if (index % 2 == 0) 0.02 else 0.68)) }
        // All ten in the picture at the same moments: two of them are left alone, and nobody's region is merged with another's.
        val crowd = Passers.link(together(*(0 until 10).map { person(it, 0..23) }.toTypedArray()))
        assertEquals(10, crowd.size)
        val gone = Passers.removable(crowd, Passers.candidates(crowd, keep, 0.0, 2.76).also { assertEquals(Passers.MAX_BOXES, it.size) }, Wall(), keep)
        assertEquals(Passers.MAX_BOXES, gone.size)
        for (step in -20..160) {
            val time = step * 0.02
            val regions = Passers.at(gone, time)
            assertTrue(regions.size <= Passers.MAX_BOXES)
            // Every region is exactly one removed person's own: nothing is replaced that nobody removed occupies.
            for (region in regions) assertTrue("at $time", gone.any { Passers.region(Passers.box(it, time)) == region })
        }
        assertEquals(Passers.MAX_BOXES, Passers.at(gone, 1.0).size)
        // Whoever was dropped to make room is not replaced at any time.
        val stays = crowd.filter { track -> gone.none { it === track } }
        assertEquals(2, stays.size)
        for (track in stays) for (seen in track.sightings) assertTrue(Passers.at(gone, seen.t).none { it.intersects(Passers.region(seen.box)) })
        // The same ten coming and going one after the other, never more than eight at once: all ten are taken out.
        val staggered = Passers.link(together(*(0 until 10).map { person(it, it * 5..it * 5 + 23) }.toTypedArray()))
        assertEquals(10, staggered.size)
        val until = (45 + 23) * 0.12
        assertEquals(10, Passers.removable(staggered, Passers.candidates(staggered, keep, 0.0, until), Wall(), keep).size)
    }

    @Test fun theClimberMustStayClearForAsLongAsARegionIsHeldAndBetweenLooks() {
        // The walker leaves at 2.76 s; their region is held a moment longer. The climber's box arrives there just then.
        val late = listOf(KeepSample(0.0, listOf(Box(0.45, 0.05, 0.15, 0.20))), KeepSample(2.80, listOf(Box(0.45, 0.05, 0.15, 0.20))), KeepSample(2.95, listOf(Box(0.90, 0.55, 0.08, 0.30))))
        assertFalse(Passers.clearOfKeep(Passers.link(walk()).single(), late))
        assertTrue(removed(walk(), late, Wall()).isEmpty())
        // The same box arriving after the hold is over is nobody's business.
        val later = listOf(KeepSample(0.0, listOf(Box(0.45, 0.05, 0.15, 0.20))), KeepSample(3.2, listOf(Box(0.45, 0.05, 0.15, 0.20))), KeepSample(3.3, listOf(Box(0.90, 0.55, 0.08, 0.30))))
        assertEquals(1, removed(walk(), later, Wall()).size)
        // And before the first sighting: the climber is where the walker's region is held ahead of them.
        val early = listOf(KeepSample(-0.2, listOf(Box(0.0, 0.55, 0.08, 0.30))), KeepSample(-0.05, listOf(Box(0.45, 0.05, 0.15, 0.20))), KeepSample(3.0, listOf(Box(0.45, 0.05, 0.15, 0.20))))
        assertTrue(removed(walk(), early, Wall()).isEmpty())
        // The climber's box passing through the walker's region between two looks at the walker counts too.
        val sparse = listOf(listOf(sighting(0.0, 0.30)), listOf(sighting(0.6, 0.34)))
        val through = listOf(KeepSample(0.0, listOf(Box(0.45, 0.05, 0.1, 0.1))), KeepSample(0.29, listOf(Box(0.45, 0.05, 0.1, 0.1))), KeepSample(0.31, listOf(Box(0.31, 0.62, 0.05, 0.1))), KeepSample(0.33, listOf(Box(0.45, 0.05, 0.1, 0.1))))
        assertFalse(Passers.clearOfKeep(Passers.link(sparse).single(), through))
    }

    @Test fun aClimberWhoMovedOnDoesNotHoldTheirOldSpot() {
        // The climber stood low, then left fast; the detector's track of them ends there, at the keep box.
        val keep = listOf(KeepSample(0.0, listOf(Box(0.44, 0.55, 0.14, 0.40))), KeepSample(1.0, listOf(Box(0.44, 0.55, 0.14, 0.40))), KeepSample(1.2, listOf(Box(0.44, 0.02, 0.14, 0.25))), KeepSample(3.0, listOf(Box(0.44, 0.02, 0.14, 0.25))))
        val climber = (0..8).map { step -> listOf(sighting(step * 0.12, 0.47, y = 0.62)) }
        // A walker crossing that spot half a second later is taken out all the same: the plan says where the climber is.
        val walker = (10..23).map { step -> listOf(sighting(step * 0.12, (step - 10) * 0.92 / 13)) }
        val gone = removed(together(climber, walker), keep, Wall())
        assertEquals(1, gone.size)
        assertEquals(1.2, gone.single().start, 1e-9)
    }

    @Test fun moreKeepBoxesThanTheShaderTakesAreMergedNotDropped() {
        val boxes = (0 until 12).map { Box(it * 0.08, 0.5, 0.05, 0.1) }
        val limited = Passers.limit(boxes, Passers.MAX_BOXES)
        assertEquals(Passers.MAX_BOXES, limited.size)
        for (box in boxes) assertTrue(limited.any { it.x <= box.x + 1e-9 && it.right >= box.right - 1e-9 && it.y <= box.y && it.bottom >= box.bottom })
        assertEquals(boxes.take(3), Passers.limit(boxes.take(3), 8))
    }

    @Test fun thePlateIsMadeForWhereverTheCandidatesWalkRingIncluded() {
        assertNull(Passers.extent(emptyList()))
        val track = Passers.link(walk()).single()
        val extent = Passers.extent(listOf(track))!!
        for (seen in track.sightings) for ((x, y) in Passers.ringPoints(Passers.region(seen.box))) {
            if (x < 0 || y < 0 || x > 1 || y > 1) continue
            assertTrue("$x, $y in $extent", x >= extent.x && x <= extent.right && y >= extent.y && y <= extent.bottom)
        }
        assertEquals(24, Passers.ringPoints(walker).size)
    }

    @Test fun boxesOverlapUnionAndGrow() {
        assertEquals(1.0, climber.overlap(climber), 1e-9)
        assertEquals(0.0, climber.overlap(walker), 0.0)
        assertEquals(1.0 / 3, Box(0.0, 0.0, 0.2, 0.2).overlap(Box(0.1, 0.0, 0.2, 0.2)), 1e-9)
        assertEquals(Box(0.05, 0.30, 0.55, 0.60).w, climber.union(walker).w, 1e-9)
        assertEquals(Box(0.0, 0.0, 0.3, 0.3), Box(0.1, 0.1, 0.1, 0.1).grown(0.1, 0.1).let { Box(it.x, it.y, Math.round(it.w * 10) / 10.0, Math.round(it.h * 10) / 10.0) })
        assertEquals(1.0, Box(0.9, 0.9, 0.1, 0.1).grown(0.5, 0.5).right, 1e-9)
    }

    private fun rgb(vararg grey: Int) = ByteArray(grey.size * 3) { grey[it / 3].toByte() }

    @Test fun thePlateIsThePerChannelMedianSoWhoeverPassesThroughIsGone() {
        // Five pictures of a wall of value 100, five pixels wide; in each a "person" (10 or 250) stands on another pixel.
        val frames = List(5) { index -> rgb(*IntArray(5) { pixel -> if (pixel == index) (if (index % 2 == 0) 10 else 250) else 100 }) }
        val plate = ByteArray(15); val alpha = ByteArray(5)
        PlateMedian.median(frames, null, plate, alpha)
        assertArrayEquals(rgb(100, 100, 100, 100, 100), plate)
        assertArrayEquals(ByteArray(5) { -1 }, alpha)
        // Channels are taken one by one, values above 127 are bytes too, and an even count takes the lower middle.
        val colours = listOf(byteArrayOf(200.toByte(), 1, 9), byteArrayOf(10, 2, 9), byteArrayOf(250.toByte(), 3, 9), byteArrayOf(220.toByte(), 4, 9))
        val one = ByteArray(3)
        PlateMedian.median(colours, null, one, ByteArray(1))
        assertArrayEquals(byteArrayOf(200.toByte(), 2, 9), one)
        // Only the asked-for pixels are written.
        val part = ByteArray(15) { 7 }; val partAlpha = ByteArray(5) { 7 }
        PlateMedian.median(frames, null, part, partAlpha, 1, 3)
        assertArrayEquals(byteArrayOf(7, 7, 7, 100, 100, 100, 100, 100, 100, 7, 7, 7, 7, 7, 7), part)
        assertArrayEquals(byteArrayOf(7, -1, -1, 7, 7), partAlpha)
    }

    @Test fun aPictureSaysNothingWhereTheClimberWasAndAPixelFewPicturesVouchForHasNoPlate() {
        // The climber (value 30) lingers on pixel 0 in four of five pictures: a plain median would keep them.
        val frames = List(5) { index -> rgb(if (index < 4) 30 else 100, 100, 100) }
        val plain = ByteArray(9)
        PlateMedian.median(frames, null, plain, ByteArray(3))
        assertEquals(30, plain[0].toInt())
        // With each picture sitting out its own keep box, pixel 0 has one witness: no plate there. Pixel 1
        // is left with exactly three, which is enough; pixel 2 has all five.
        val excluded = List(5) { index -> byteArrayOf(if (index < 4) 1 else 0, if (index < 2) 1 else 0, 0) }
        val plate = ByteArray(9) { 55 }; val alpha = ByteArray(3) { 55 }
        PlateMedian.median(frames, excluded, plate, alpha)
        assertArrayEquals(byteArrayOf(0, 0, 0, 100, 100, 100, 100, 100, 100), plate)
        assertArrayEquals(byteArrayOf(0, -1, -1), alpha)
    }

    @Test fun keepBoxesAreMarkedInThePlatesOwnRowsWhichCountFromTheBottom() {
        // An 8 × 16 picture; the plate region is columns 1…6 (6 wide) and rows 3…8 from the bottom (6 rows).
        val mask = ByteArray(36) { 9 }
        // A box over x 0.25…0.625 and y 0.5…0.75 from the top: columns 2…4, rows from the bottom 4…7.
        PlateMedian.mark(mask, 6, 6, left = 1, firstRow = 3, pictureWidth = 8, pictureHeight = 16, boxes = listOf(Box(0.25, 0.5, 0.375, 0.25)))
        val rows = (0 until 6).map { row -> mask.slice(row * 6 until row * 6 + 6).joinToString("") }
        assertEquals(listOf("000000", "011100", "011100", "011100", "011100", "000000"), rows)
        // A box outside the region marks nothing, and what was marked before is cleared.
        PlateMedian.mark(mask, 6, 6, 1, 3, 8, 16, listOf(Box(0.0, 0.0, 0.125, 0.125)))
        assertTrue(mask.all { it.toInt() == 0 })
    }

    @Test fun plateTimesAreThinnedEvenlyWhenThereIsLittleRoom() {
        val times = (0 until 25).map { it.toDouble() }
        assertEquals(times, EditRenderer.thinned(times, 25)); assertEquals(times, EditRenderer.thinned(times, 100))
        assertEquals(listOf(0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0), EditRenderer.thinned(times, 9))
        assertEquals(listOf(0.0, 12.0, 24.0), EditRenderer.thinned(times, 3))
        assertTrue(EditRenderer.thinned(times, 0).isEmpty()); assertTrue(EditRenderer.thinned(times, -4).isEmpty())
    }

    /** The detector model ships inside the app; it is never downloaded. This pins the exact file. */
    @Test fun bundledPeopleModelIsThePinnedFile() {
        val model = File("src/main/assets/${PeopleModel.ASSET}")
        assertTrue("Missing ${model.path}", model.isFile)
        assertEquals(PeopleModel.BYTES, model.length())
        assertEquals(PeopleModel.SHA256, com.betabuddy.android.pose.PoseModel.sha256(model.readBytes()))
        assertTrue(model.length() <= 50L * 1024 * 1024)
    }
}
