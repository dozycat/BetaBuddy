package com.betabuddy.android.edit

import kotlin.math.abs
import kotlin.math.hypot

// Who is replaced by the clean plate when passer-by removal is on. The people in the frames come
// from an on-device detector; which of them is the climber comes from the plan's keep boxes.
//
// The rule is all or nothing, per person: somebody is taken out only if they can be taken out
// cleanly for as long as they are seen. Otherwise they are left exactly as they are for the whole
// time: no half bodies, nobody flickering in and out, no ghosts. When nobody can be taken out
// the picture is simply not touched. Pure arithmetic; no Android classes.

/**
 * One person in one looked-at frame. [ring] holds what the frame showed just outside the person's
 * region, at [Passers.ringPoints] of it: packed RGB, or -1 where the point is outside the picture.
 */
class Sighting(val t: Double, val box: Box, val ring: IntArray? = null)

/** One person followed through the clip: their sightings in order of time. */
class PersonTrack(val sightings: List<Sighting>) {
    val start: Double get() = sightings.first().t
    val end: Double get() = sightings.last().t
}

/** What the removal needs to know about the plate, without its pixels. Coordinates are fractions of the picture, origin top-left. */
interface PlateView {
    /** True when the plate knows every pixel of [region] (the part of it inside the picture). */
    fun covers(region: Box): Boolean

    /** The plate's colour around ([x], [y]) as packed RGB, or null where it has none. */
    fun colour(x: Double, y: Double): Int?
}

object Passers {
    /** How far a keep box reaches beyond itself when deciding who touches it: a fraction of the picture. */
    const val KEEP_MARGIN = 0.02
    /** How far around a keep box a plate picture is not trusted either: the climber's limbs reach past the box. */
    const val PLATE_MARGIN = 0.03
    /** How much bigger than the detected person the replaced region is: a fraction of the person's own size… */
    const val GROW = 0.12
    /** …and at least this fraction of the picture, since a far-away person's box is small and never exact. */
    const val GROW_MIN = 0.012
    /** A person not found for up to this long is still the same person when found again nearby; the gap is bridged. */
    const val MAX_GAP = 1.0
    /** A removed person's region stays this long before their first and after their last sighting. */
    const val HOLD = 0.3
    /** A box this close to the picture's edge belongs to somebody walking in or out. */
    const val EDGE = 0.06
    /** How long a track's last pace is extrapolated when looking for its next sighting, in seconds. */
    private const val PACE_TRUSTED = 0.5
    /** How much a second without a sighting counts against a track when two could continue with the same sighting. */
    private const val STALE = 1.5
    /** Between sightings a track is checked at steps no longer than this, in seconds. */
    private const val STEP = 0.15
    /** How far outside a region the frame and the plate are compared: a fraction of the picture. */
    const val RING_OUT = 0.014
    /** Points compared on each side of a region. */
    const val RING_SIDE = 6
    /** Frame and plate agree at a point when no channel differs by more than this (of 255)… */
    const val RING_TOLERANCE = 32
    /** …and around a region when at least this share of the points inside the picture agree… */
    const val RING_SHARE = 0.7
    /** …of which there must be at least this many. */
    const val RING_MIN = 6
    /** The compositing shader takes this many boxes of each kind. More people than this at one moment are not all taken out. */
    const val MAX_BOXES = 8

    /** The region replaced for a person found at [box]: the box, grown a little. */
    fun region(box: Box): Box = box.grown(maxOf(box.w * GROW, GROW_MIN), maxOf(box.h * GROW, GROW_MIN))

    /**
     * Where frame and plate are compared for a person's [region]: [RING_SIDE] points along each side,
     * [RING_OUT] outside it, clockwise from the top-left. Points may fall outside the picture.
     */
    fun ringPoints(region: Box): List<Pair<Double, Double>> {
        val points = ArrayList<Pair<Double, Double>>(RING_SIDE * 4)
        for (index in 0 until RING_SIDE) {
            val along = (index + 0.5) / RING_SIDE
            points += (region.x + region.w * along) to (region.y - RING_OUT)
            points += (region.right + RING_OUT) to (region.y + region.h * along)
            points += (region.x + region.w * along) to (region.bottom + RING_OUT)
            points += (region.x - RING_OUT) to (region.y + region.h * along)
        }
        return points
    }

    /**
     * Follows each person from one looked-at frame to the next. A track expects its person where
     * they were last seen, moved on at the pace of their last two sightings; a sighting continues
     * the track that expected it most closely and most recently, if that is near enough for the
     * time that has passed (up to [MAX_GAP]); otherwise it begins a track of its own.
     * [samples] hold everybody found in each frame.
     */
    fun link(samples: List<List<Sighting>>): List<PersonTrack> {
        val tracks = ArrayList<MutableList<Sighting>>()
        for (sample in samples.filter { it.isNotEmpty() }.sortedBy { it.first().t }) {
            val time = sample.first().t
            val open = tracks.filter { time - it.last().t <= MAX_GAP && time > it.last().t }
            // Every possible pairing, best first; each track and each sighting is used once.
            val pairs = ArrayList<Triple<Double, MutableList<Sighting>, Sighting>>()
            for (track in open) for (sighting in sample) {
                val last = track.last()
                val waited = time - last.t
                val size = (maxOf(last.box.w, last.box.h) + maxOf(sighting.box.w, sighting.box.h)) / 2
                // Where the person would be by now. A pace is only trusted for a short while.
                var x = centreX(last.box)
                var y = centreY(last.box)
                val before = track.getOrNull(track.size - 2)
                val paced = before != null && last.t > before.t
                if (before != null && paced) {
                    val ahead = minOf(waited, PACE_TRUSTED) / (last.t - before.t)
                    x += (x - centreX(before.box)) * ahead
                    y += (y - centreY(before.box)) * ahead
                }
                val distance = hypot(x - centreX(sighting.box), y - centreY(sighting.box))
                // How far from the expected place the person may be, in their own sizes: less when their
                // pace is known. Somebody who stood still is not found three bodies away a moment later.
                val reach = size * if (paced) 0.5 + 1.0 * waited else 0.75 + 2.0 * waited
                if (distance <= reach) pairs += Triple(distance / size + STALE * waited, track, sighting)
            }
            val taken = java.util.Collections.newSetFromMap(java.util.IdentityHashMap<Any, Boolean>())
            for ((_, track, sighting) in pairs.sortedBy { it.first }) {
                if (track in taken || sighting in taken) continue
                taken.add(track); taken.add(sighting)
                track.add(sighting)
            }
            for (sighting in sample) if (sighting !in taken) tracks.add(mutableListOf(sighting))
        }
        return tracks.map { PersonTrack(it) }
    }

    private fun centreX(box: Box) = box.x + box.w / 2
    private fun centreY(box: Box) = box.y + box.h / 2

    /** The person's box at [t] within the track: on a straight line between the sightings around it, held at the ends. */
    fun box(track: PersonTrack, t: Double): Box {
        val sightings = track.sightings
        val next = sightings.indexOfFirst { it.t > t }
        if (next == 0) return sightings.first().box
        if (next == -1) return sightings.last().box
        val a = sightings[next - 1]
        val b = sightings[next]
        return KeepPath.lerp(a.box, b.box, (t - a.t) / (b.t - a.t))
    }

    /**
     * The moments a track is checked at, over all the time its region would be applied: from [HOLD]
     * before the first sighting to [HOLD] after the last, at every sighting, at steps of at most
     * [STEP] in between (the bridged gaps and the holds), and at each of [also] that falls inside.
     */
    private fun moments(track: PersonTrack, also: List<Double> = emptyList()): List<Double> {
        val marks = ArrayList<Double>()
        marks += track.start - HOLD
        for (sighting in track.sightings) marks += sighting.t
        marks += track.end + HOLD
        val moments = ArrayList<Double>()
        for ((index, mark) in marks.withIndex()) {
            moments += mark
            val next = marks.getOrNull(index + 1) ?: break
            val steps = Math.ceil((next - mark) / STEP).toInt()
            for (step in 1 until steps) moments += mark + (next - mark) * step / steps
        }
        for (time in also) if (time > track.start - HOLD && time < track.end + HOLD) moments += time
        return moments
    }

    /**
     * True when the person's region never touches a (slightly grown) keep box, at any moment the
     * region would be applied: bridged gaps and the holds at both ends included, and every moment
     * the plan itself says where the climber is.
     */
    fun clearOfKeep(track: PersonTrack, keep: List<KeepSample>): Boolean = moments(track, keep.map { it.t }).all { time ->
        val region = region(box(track, time))
        KeepPath.boxes(keep, time).none { it.grown(KEEP_MARGIN, KEEP_MARGIN).intersects(region) }
    }

    /**
     * True when the track is the whole of the person's time in the picture: it begins where the
     * looked-at stretch ([from]…[until]) begins or at the picture's edge, and ends likewise. Somebody
     * the detector loses in the middle of the picture would otherwise appear from nowhere.
     */
    fun whole(track: PersonTrack, from: Double, until: Double): Boolean {
        fun atEdge(box: Box) = box.x <= EDGE || box.y <= EDGE || box.right >= 1 - EDGE || box.bottom >= 1 - EDGE
        val first = track.sightings.first()
        val last = track.sightings.last()
        return (first.t - from <= MAX_GAP / 2 || atEdge(first.box)) && (until - last.t <= MAX_GAP / 2 || atEdge(last.box))
    }

    /**
     * The tracks worth making a plate for, out of everybody found ([tracks]): clear of the climber
     * throughout, whole, and never over anybody who stays ([settled]).
     */
    fun candidates(tracks: List<PersonTrack>, keep: List<KeepSample>, from: Double, until: Double): List<PersonTrack> =
        settled(tracks, tracks.filter { clearOfKeep(it, keep) && whole(it, from, until) }, keep)

    /**
     * The people really taken out: the [candidates] the [plate] is good for, minus anybody whose
     * region would ever lie over somebody who stays, minus as many as it takes for the regions of
     * one moment to fit the shader without merging any two.
     */
    fun removable(tracks: List<PersonTrack>, candidates: List<PersonTrack>, plate: PlateView, keep: List<KeepSample>): List<PersonTrack> =
        settled(tracks, candidates.filter { plateFits(it, plate) }, keep)

    /**
     * [chosen] reduced until it holds together. Whoever is not taken out stays untouched for the whole
     * clip, so nobody may be taken out whose region (bridged gaps and holds included) ever lies over
     * the region of somebody who stays; dropping one person from the list makes them somebody who
     * stays, so this is repeated until nothing changes. And no moment may need more regions than the
     * shader takes: rather than cover more than the people themselves, the shortest-seen are dropped.
     */
    fun settled(tracks: List<PersonTrack>, chosen: List<PersonTrack>, keep: List<KeepSample> = emptyList()): List<PersonTrack> {
        var out = chosen
        while (true) {
            val staying = tracks.filter { track -> out.none { it === track } }
            var next = out.filter { track -> staying.none { over(track, it, keep) } }
            if (next.size == out.size) {
                // The number of regions only changes where one begins: looking there is looking everywhere.
                val crowded = next.map { it.start - HOLD }.firstOrNull { time -> applied(next, time).size > MAX_BOXES }
                if (crowded == null) return next
                val dropped = applied(next, crowded).minByOrNull { it.sightings.size }
                next = next.filter { it !== dropped }
            }
            out = next
        }
    }

    private fun applied(removed: List<PersonTrack>, t: Double) = removed.filter { t >= it.start - HOLD && t <= it.end + HOLD }

    /**
     * True when the region [passer] would be replaced in ever lies over [other]'s region. Somebody
     * who stays is taken to be where they were first and last seen for [MAX_GAP] before and after
     * their sightings: lost by the detector is not gone from the picture. The exception is the
     * climber: where a track begins or ends at a keep box, the plan says where that person is from
     * then on (and the keep box protects them), so nothing is assumed beyond the sightings.
     */
    private fun over(passer: PersonTrack, other: PersonTrack, keep: List<KeepSample>): Boolean {
        fun atClimber(sighting: Sighting) = KeepPath.boxes(keep, sighting.t).any { it.grown(KEEP_MARGIN, KEEP_MARGIN).intersects(region(sighting.box)) }
        val from = other.start - if (atClimber(other.sightings.first())) 0.0 else MAX_GAP
        val until = other.end + if (atClimber(other.sightings.last())) 0.0 else MAX_GAP
        return moments(passer, other.sightings.map { it.t }).any { time ->
            time in from..until && region(box(passer, time)).intersects(region(box(other, time)))
        }
    }

    /**
     * True when [plate] can stand in for the person all along the track: it knows every pixel of
     * their region at every moment, and at every sighting it shows, just outside the region, what
     * the frame showed there. The second catches a plate that is not this scene any more: the
     * light changed, the camera was nudged, somebody who stood still is part of it.
     */
    fun plateFits(track: PersonTrack, plate: PlateView): Boolean = plateCovers(track, plate) && plateMatches(track, plate)

    /** The plate knows every pixel of the person's region at every moment the region would be applied. */
    fun plateCovers(track: PersonTrack, plate: PlateView): Boolean = moments(track).all { plate.covers(region(box(track, it))) }

    /** At every sighting the plate shows, just outside the person's region, what the frame showed there. */
    fun plateMatches(track: PersonTrack, plate: PlateView): Boolean {
        return track.sightings.all { sighting ->
            val ring = sighting.ring ?: return@all false
            val points = ringPoints(region(sighting.box))
            var inside = 0
            var agree = 0
            for ((index, point) in points.withIndex()) {
                val seen = ring.getOrNull(index)?.takeIf { it >= 0 } ?: continue
                inside += 1
                val known = plate.colour(point.first, point.second) ?: continue
                if (channelsApart(seen, known) <= RING_TOLERANCE) agree += 1
            }
            inside >= RING_MIN && agree >= inside * RING_SHARE
        }
    }

    /**
     * Why a person is or is not taken out, in a few words, for a report on a render: the first of the
     * rules that says no. [candidates] and [removed] are what [candidates] and [removable] gave;
     * [plate] is null when no plate could be made.
     */
    fun verdict(
        track: PersonTrack, keep: List<KeepSample>, from: Double, until: Double,
        candidates: List<PersonTrack>, plate: PlateView?, removed: List<PersonTrack>,
    ): String = when {
        removed.any { it === track } -> "removed"
        !clearOfKeep(track, keep) -> "kept: touches the climber's keep box"
        !whole(track, from, until) -> "kept: not seen from coming in to going out"
        candidates.none { it === track } -> "kept: would lie over somebody who stays"
        plate == null -> "kept: no plate could be made"
        !plateCovers(track, plate) -> "kept: the plate has no pixels somewhere they go"
        !plateMatches(track, plate) -> "kept: the plate does not show the scene around them"
        else -> "kept: would lie over somebody who stays, or too many at one moment"
    }

    internal fun channelsApart(a: Int, b: Int): Int =
        maxOf(abs((a shr 16 and 0xFF) - (b shr 16 and 0xFF)), abs((a shr 8 and 0xFF) - (b shr 8 and 0xFF)), abs((a and 0xFF) - (b and 0xFF)))

    /** One box around every region the tracks ever cover, ring included, or null without tracks: the part of the picture a plate is needed for. */
    fun extent(tracks: List<PersonTrack>): Box? = tracks.flatMap { track -> track.sightings.map { region(it.box).grown(RING_OUT * 2, RING_OUT * 2) } }.reduceOrNull(Box::union)

    /**
     * The regions to replace at source time [t] for the people being taken out ([removable]): each
     * person's own region from [HOLD] before their first sighting to [HOLD] after their last, moving
     * in a straight line between sightings. Regions are never merged: nothing is replaced that no
     * removed person occupies. ([settled] has seen to it that there are at most [MAX_BOXES].)
     */
    fun at(removed: List<PersonTrack>, t: Double): List<Box> = applied(removed, t).take(MAX_BOXES).map { region(box(it, t)) }

    /**
     * At most [count] boxes covering everything [boxes] cover: the two whose union wastes least are
     * merged until they fit. Only for boxes that keep the picture live (the plan's keep boxes), where
     * covering more does no harm; never for regions that are replaced.
     */
    fun limit(boxes: List<Box>, count: Int): List<Box> {
        val result = boxes.toMutableList()
        while (result.size > count) {
            var best = 0 to 1
            var least = Double.MAX_VALUE
            for (i in result.indices) for (j in i + 1 until result.size) {
                val waste = result[i].union(result[j]).area - result[i].area - result[j].area
                if (waste < least) { least = waste; best = i to j }
            }
            val merged = result[best.first].union(result[best.second])
            result.removeAt(best.second)
            result[best.first] = merged
        }
        return result
    }
}

/**
 * Per-channel temporal median of same-sized pictures: what stays when everything that moves is
 * taken away. A picture can be told to sit out some of its pixels (where the climber was at that
 * moment); a pixel that too few pictures can vouch for gets no plate at all.
 */
object PlateMedian {
    /** A pixel seen in fewer pictures than this has no plate and is never replaced. */
    const val MIN_SAMPLES = 3

    /**
     * [frames] are equally long RGB byte arrays, three bytes a pixel. For every pixel in [from]
     * until [until], [target] (RGB) receives the per-channel median over the frames that may speak
     * for it, and [alpha] 255; when fewer than [MIN_SAMPLES] may, [target] is left at zero and
     * [alpha] is 0. [excluded], when given, holds one byte per pixel for each frame: non-zero means
     * that frame sits this pixel out. An even count takes the lower of the two middle values, so
     * every output byte is one that was seen.
     */
    fun median(frames: List<ByteArray>, excluded: List<ByteArray>?, target: ByteArray, alpha: ByteArray, from: Int = 0, until: Int = alpha.size) {
        val count = frames.size
        require(count > 0 && from >= 0 && frames.all { it.size >= until * 3 } && target.size >= until * 3 && alpha.size >= until)
        require(excluded == null || (excluded.size == count && excluded.all { it.size >= until }))
        val values = IntArray(count)
        val usable = IntArray(count)
        for (pixel in from until until) {
            var samples = 0
            for (index in 0 until count) if (excluded == null || excluded[index][pixel].toInt() == 0) usable[samples++] = index
            if (samples < MIN_SAMPLES) {
                target[pixel * 3] = 0; target[pixel * 3 + 1] = 0; target[pixel * 3 + 2] = 0
                alpha[pixel] = 0
                continue
            }
            val middle = (samples - 1) / 2
            for (channel in 0 until 3) {
                val position = pixel * 3 + channel
                // Insertion sort: the lists are short (at most 25) and mostly in order from pixel to pixel.
                for (index in 0 until samples) {
                    val value = frames[usable[index]][position].toInt() and 0xFF
                    var slot = index
                    while (slot > 0 && values[slot - 1] > value) { values[slot] = values[slot - 1]; slot -= 1 }
                    values[slot] = value
                }
                target[position] = values[middle].toByte()
            }
            alpha[pixel] = -1
        }
    }

    /**
     * Marks, in [mask] (one byte per pixel, [width] × [rows], first row = picture row [firstRow]
     * counted from the BOTTOM of a picture [pictureHeight] rows high, columns starting at picture
     * column [left]), every pixel inside one of [boxes] (fractions of a [pictureWidth] ×
     * [pictureHeight] picture, origin top-left). Everything else is cleared.
     */
    fun mark(mask: ByteArray, width: Int, rows: Int, left: Int, firstRow: Int, pictureWidth: Int, pictureHeight: Int, boxes: List<Box>) {
        java.util.Arrays.fill(mask, 0, width * rows, 0)
        for (box in boxes) {
            val x0 = (Math.floor(box.x * pictureWidth).toInt() - left).coerceIn(0, width)
            val x1 = (Math.ceil(box.right * pictureWidth).toInt() - left).coerceIn(0, width)
            // Top-left fractions to rows counted from the bottom.
            val low = pictureHeight - Math.ceil(box.bottom * pictureHeight).toInt()
            val high = pictureHeight - Math.floor(box.y * pictureHeight).toInt()
            val r0 = (low - firstRow).coerceIn(0, rows)
            val r1 = (high - firstRow).coerceIn(0, rows)
            for (row in r0 until r1) java.util.Arrays.fill(mask, row * width + x0, row * width + x1, 1)
        }
    }
}
