package com.betabuddy.android.journal

import kotlin.math.asin
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.doubleOrNull

// The offline gym directory (`assets/gyms-cn.json`) and what can honestly be said about how far a
// gym is. Counterpart of the directory part of ios/BetaBuddy/Core/GymLocator.swift. Pure Kotlin.

/**
 * What a directory entry's coordinates point at. Only the first three say where the gym is;
 * a district or city centre is the same point for every gym in it and can be tens of kilometres off.
 */
enum class CoordinatePrecision {
    POI, ADDRESS, STATION, DISTRICT, CITY;

    /** Good enough to rank gyms by distance. */
    val ranks: Boolean get() = this == POI || this == ADDRESS || this == STATION

    companion object {
        fun of(text: String?): CoordinatePrecision? = entries.firstOrNull { it.name.lowercase() == text }
    }
}

data class DirectoryGym(
    val id: String, val name: String, val province: String = "", val city: String = "", val address: String = "",
    val sourceURL: String = "", val district: String = "", val aliases: List<String> = emptyList(),
    val lat: Double? = null, val lon: Double? = null, val coordPrecision: String? = null,
) {
    /** Null when the entry has none, or a value this build does not know: it then has no usable coordinate. */
    val precision: CoordinatePrecision? get() = CoordinatePrecision.of(coordPrecision)

    /** The entry's position (latitude, longitude) when all three fields are there and make sense. */
    val coordinate: Pair<Double, Double>?
        get() {
            val latitude = lat ?: return null
            val longitude = lon ?: return null
            if (precision == null || !latitude.isFinite() || !longitude.isFinite() || latitude !in -90.0..90.0 || longitude !in -180.0..180.0) return null
            if (latitude == 0.0 && longitude == 0.0) return null
            return latitude to longitude
        }
}

object GymDirectory {
    /** Shown wherever a distance from the directory's coordinates is shown. Not translated. */
    const val COORDINATE_CREDIT = "© OpenStreetMap contributors"

    /**
     * Reads the directory leniently: an entry without an id or a name is left out, a coordinate
     * field of the wrong type only costs that entry its coordinate, and unknown fields are ignored.
     */
    fun parse(text: String): List<DirectoryGym> {
        val root = try { Json.parseToJsonElement(text) as? JsonObject } catch (error: Exception) { null } ?: return emptyList()
        return (root["gyms"] as? JsonArray).orEmpty().mapNotNull { item ->
            val entry = item as? JsonObject ?: return@mapNotNull null
            fun string(name: String) = (entry[name] as? JsonPrimitive)?.takeIf { it.isString }?.content
            fun number(name: String) = (entry[name] as? JsonPrimitive)?.takeIf { !it.isString }?.doubleOrNull
            DirectoryGym(
                id = string("id")?.takeIf { it.isNotEmpty() } ?: return@mapNotNull null,
                name = string("name")?.takeIf { it.isNotEmpty() } ?: return@mapNotNull null,
                province = string("province").orEmpty(), city = string("city").orEmpty(), address = string("address").orEmpty(),
                sourceURL = string("sourceURL").orEmpty(), district = string("district").orEmpty(),
                aliases = (entry["aliases"] as? JsonArray).orEmpty().mapNotNull { alias: JsonElement -> (alias as? JsonPrimitive)?.takeIf { it.isString }?.content },
                lat = number("lat"), lon = number("lon"), coordPrecision = string("coordPrecision"),
            )
        }
    }
}

/** A gym from the offline directory with how far away it is. */
data class NearbyGym(val gym: DirectoryGym, val meters: Double, val shown: Shown) {
    enum class Shown {
        /** The directory knows where this gym itself is. */
        EXACT,
        /** Only the nearest station is known: shown rounded, as "about". */
        APPROXIMATE,
        /** The phone's own position is too coarse for a number; the distance only orders the list. */
        HIDDEN,
    }

    /** The distance as it may be shown: as measured, or rounded to 100 m below a kilometre and to 0.5 km above. */
    val displayMeters: Double?
        get() = when (shown) {
            Shown.EXACT -> meters
            Shown.APPROXIMATE -> if (meters < 1_000) maxOf(100.0, Math.round(meters / 100) * 100.0) else Math.round(meters / 500) * 500.0
            Shown.HIDDEN -> null
        }
}

/** What the capture screen offers when the phone's position is known. */
data class GymSuggestions(
    /** Nearest first; only gyms whose own position is known. */
    val ranked: List<NearbyGym> = emptyList(),
    /**
     * Other gyms of the city the phone seems to be in, without distances: for these the directory
     * only knows the district or the city, or nothing.
     */
    val sameCity: List<DirectoryGym> = emptyList(),
    /** A gym close enough to be where the user is standing. Offered, never chosen for them. */
    val here: NearbyGym? = null,
)

object GymDistance {
    /** Further than this is not "nearby". */
    const val REACH = 100_000.0
    /** A fix coarser than this cannot tell how far a gym is. */
    const val USEFUL_ACCURACY = 1_000.0
    /** Closer than this, with a fix at least this good, a gym is taken to be the one the user is at. */
    const val ARRIVAL = 300.0

    /** Great-circle distance in metres between two WGS-84 positions. */
    fun meters(lat1: Double, lon1: Double, lat2: Double, lon2: Double): Double {
        val radius = 6_371_008.8
        val p1 = Math.toRadians(lat1)
        val p2 = Math.toRadians(lat2)
        val dp = p2 - p1
        val dl = Math.toRadians(lon2 - lon1)
        val a = sin(dp / 2) * sin(dp / 2) + cos(p1) * cos(p2) * sin(dl / 2) * sin(dl / 2)
        return 2 * radius * asin(minOf(1.0, sqrt(a)))
    }

    fun suggestions(gyms: List<DirectoryGym>, latitude: Double, longitude: Double, accuracy: Double, limit: Int = 6, cityLimit: Int = 12): GymSuggestions {
        if (!latitude.isFinite() || !longitude.isFinite() || latitude !in -90.0..90.0 || longitude !in -180.0..180.0) return GymSuggestions()
        val precise = accuracy >= 0 && accuracy <= USEFUL_ACCURACY
        // Every entry with a usable coordinate, however coarse, within reach.
        val located = gyms.mapNotNull { gym ->
            val precision = gym.precision ?: return@mapNotNull null
            val (lat, lon) = gym.coordinate ?: return@mapNotNull null
            val distance = meters(latitude, longitude, lat, lon)
            if (distance <= REACH) Triple(gym, distance, precision) else null
        }.sortedWith(compareBy({ it.second }, { it.first.name }))
        val ranked = located.filter { it.third.ranks }.map { (gym, distance, precision) ->
            NearbyGym(gym, distance, if (!precise) NearbyGym.Shown.HIDDEN else if (precision == CoordinatePrecision.STATION) NearbyGym.Shown.APPROXIMATE else NearbyGym.Shown.EXACT)
        }
        val shown = ranked.take(limit)
        // The city is read off the nearest entry of any kind: a district centre is no good for a
        // distance, but it does say which city this is.
        val nearest = located.firstOrNull()?.first
        val sameCity = if (nearest == null) emptyList() else {
            val listed = shown.mapTo(mutableSetOf()) { it.gym.id }
            val coarse = HashMap<String, Double>()
            for ((gym, distance, _) in located) coarse[gym.id] = minOf(distance, coarse[gym.id] ?: Double.POSITIVE_INFINITY)
            gyms.filter { it.province == nearest.province && it.city == nearest.city && it.id !in listed && it.precision?.ranks != true }
                // Nearer districts first, entries without any coordinate last; never shown as a distance.
                .sortedWith(compareBy({ coarse[it.id] ?: Double.POSITIVE_INFINITY }, { it.name }))
                .take(cityLimit)
        }
        val first = ranked.firstOrNull()
        val here = first?.takeIf { it.shown == NearbyGym.Shown.EXACT && it.meters < ARRIVAL && accuracy <= ARRIVAL }
        return GymSuggestions(shown, sameCity, here)
    }
}
