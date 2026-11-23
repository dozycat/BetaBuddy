package com.betabuddy.android.journal

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/** Ranking the offline directory by distance, with what each entry's coordinate is worth. A synthetic directory around a made-up position. */
class GymDirectoryTest {
    private val text = """
    {"version":1,"updatedAt":"2026-01-01","gyms":[
     {"id":"syn:east","name":"合成岩馆（东门）","province":"合成省","city":"合成市","district":"东区","address":"东门路 1 号","sourceURL":"https://example.invalid/","lat":30.0018,"lon":120.0,"coordPrecision":"poi"},
     {"id":"syn:station","name":"只知道最近车站的岩馆","province":"合成省","city":"合成市","district":"北区","address":"","sourceURL":"https://example.invalid/","lat":30.0115,"lon":120.0,"coordPrecision":"station"},
     {"id":"syn:west","name":"合成岩馆（西门）","province":"合成省","city":"合成市","district":"西区","address":"西门路 2 号","sourceURL":"https://example.invalid/","lat":30.0,"lon":119.95,"coordPrecision":"address"},
     {"id":"syn:district","name":"只知道在哪个区的岩馆","province":"合成省","city":"合成市","district":"南区","address":"","sourceURL":"https://example.invalid/","lat":29.99,"lon":120.0,"coordPrecision":"district"},
     {"id":"syn:city","name":"只知道在哪个市的岩馆","province":"合成省","city":"合成市","address":"","sourceURL":"https://example.invalid/","lat":30.02,"lon":120.02,"coordPrecision":"city"},
     {"id":"syn:far","name":"很远的岩馆","province":"远方省","city":"远方市","address":"","sourceURL":"https://example.invalid/","lat":40.0,"lon":116.0,"coordPrecision":"poi"},
     {"id":"syn:nowhere","name":"没有坐标的岩馆","province":"合成省","city":"合成市","address":"","sourceURL":"https://example.invalid/"},
     {"id":"syn:odd","name":"精度写法不认识的岩馆","province":"合成省","city":"合成市","address":"","sourceURL":"https://example.invalid/","lat":30.0001,"lon":120.0,"coordPrecision":"planet"},
     {"id":"syn:broken","name":"坐标写坏了的岩馆","province":"合成省","city":"合成市","address":"","sourceURL":"https://example.invalid/","lat":"north","lon":null,"coordPrecision":7},
     {"name":"没有 id 的条目"}, 7
    ]}
    """
    private val gyms = GymDirectory.parse(text)
    private fun at(latitude: Double = 30.0, longitude: Double = 120.0, accuracy: Double = 30.0) = GymDistance.suggestions(gyms, latitude, longitude, accuracy)

    @Test fun theDirectoryIsReadLeniently() {
        assertEquals(9, gyms.size)   // the two entries that are not gyms are left out, the rest survive a broken neighbour
        val broken = gyms.first { it.id == "syn:broken" }
        assertNull(broken.lat); assertNull(broken.coordPrecision); assertNull(broken.coordinate)
        assertNull(gyms.first { it.id == "syn:odd" }.precision)
        assertNull(gyms.first { it.id == "syn:odd" }.coordinate)     // an unknown precision is no usable coordinate
        assertNull(gyms.first { it.id == "syn:nowhere" }.coordinate)
        assertEquals(CoordinatePrecision.STATION, gyms.first { it.id == "syn:station" }.precision)
        assertTrue(GymDirectory.parse("not json").isEmpty() && GymDirectory.parse("{}").isEmpty() && GymDirectory.parse("[]").isEmpty())
        assertNull(DirectoryGym("x", "x", lat = 0.0, lon = 0.0, coordPrecision = "poi").coordinate)
        assertNull(DirectoryGym("x", "x", lat = 91.0, lon = 10.0, coordPrecision = "poi").coordinate)
    }

    @Test fun onlyGymsWhoseOwnPositionIsKnownAreRanked() {
        val result = at()
        assertEquals(listOf("syn:east", "syn:station", "syn:west"), result.ranked.map { it.gym.id })
        assertEquals(200.0, result.ranked[0].meters, 2.0)
        assertEquals(listOf(NearbyGym.Shown.EXACT, NearbyGym.Shown.APPROXIMATE, NearbyGym.Shown.EXACT), result.ranked.map { it.shown })
        // A district centre 1.1 km away never appears as "near", whatever its distance.
        assertTrue(result.ranked.none { it.gym.precision?.ranks != true })
    }

    @Test fun stationDistancesAreRoundedAndNeverLookCloserThanAKilometre() {
        val station = at().ranked.first { it.gym.id == "syn:station" }
        assertEquals(1279.0, station.meters, 3.0)
        assertEquals(1500.0, station.displayMeters!!, 0.0)     // to half a kilometre
        // The station is typically about a kilometre from the gym itself: nothing nearer is claimed.
        assertEquals(1000.0, NearbyGym(station.gym, 649.0, NearbyGym.Shown.APPROXIMATE).displayMeters!!, 0.0)
        assertEquals(1000.0, NearbyGym(station.gym, 20.0, NearbyGym.Shown.APPROXIMATE).displayMeters!!, 0.0)
        assertEquals(1000.0, NearbyGym(station.gym, 1100.0, NearbyGym.Shown.APPROXIMATE).displayMeters!!, 0.0)
        assertEquals(2500.0, NearbyGym(station.gym, 2300.0, NearbyGym.Shown.APPROXIMATE).displayMeters!!, 0.0)
        assertEquals(649.0, NearbyGym(station.gym, 649.0, NearbyGym.Shown.EXACT).displayMeters!!, 0.0)
        assertNull(NearbyGym(station.gym, 649.0, NearbyGym.Shown.HIDDEN).displayMeters)
    }

    @Test fun theRestOfTheCityFollowsWithoutDistances() {
        val result = at()
        // Nearer districts first, entries without any coordinate after them by name; the far city is not this one.
        assertEquals(listOf("syn:district", "syn:city", "syn:broken", "syn:nowhere", "syn:odd").take(2), result.sameCity.map { it.id }.take(2))
        assertEquals(setOf("syn:district", "syn:city", "syn:broken", "syn:nowhere", "syn:odd"), result.sameCity.map { it.id }.toSet())
        assertEquals(result.sameCity.drop(2).map { it.name }.sorted(), result.sameCity.drop(2).map { it.name })
    }

    @Test fun aGymRightHereIsOfferedOnlyWhenBothPositionsAreGood() {
        assertEquals("syn:east", at().here?.gym?.id)
        assertNull(at(accuracy = 400.0).here)                       // the phone's own position is too coarse
        assertNull(at(latitude = 30.0115).here)                     // nearest is the station entry: never "here"
        assertNull(at(latitude = 30.0060).here)                     // 470 m from the nearest exact one
        assertNull(at(latitude = 29.99).here)                       // standing on a district centre offers nothing
    }

    @Test fun aCoarseFixOrdersButShowsNoNumbers() {
        // What approximate location gives: a couple of kilometres of error.
        val result = at(accuracy = 2000.0)
        assertEquals(listOf("syn:east", "syn:station", "syn:west"), result.ranked.map { it.gym.id })
        assertTrue(result.ranked.all { it.shown == NearbyGym.Shown.HIDDEN && it.displayMeters == null })
        assertNull(result.here)
        assertTrue(at(accuracy = -1.0).ranked.all { it.shown == NearbyGym.Shown.HIDDEN })
    }

    @Test fun nothingNearbyAndNonsensePositions() {
        val far = at(latitude = 10.0, longitude = 10.0)
        assertTrue(far.ranked.isEmpty() && far.sameCity.isEmpty() && far.here == null)
        assertEquals(GymSuggestions(), at(latitude = Double.NaN))
        assertEquals(GymSuggestions(), at(latitude = 95.0))
        assertEquals(listOf("syn:far"), at(latitude = 40.0, longitude = 116.0).ranked.map { it.gym.id })
        assertEquals(1, GymDistance.suggestions(gyms, 30.0, 120.0, 30.0, limit = 1).ranked.size)
    }

    @Test fun distances() {
        assertEquals(111_195.0, GymDistance.meters(0.0, 0.0, 1.0, 0.0), 1.0)
        assertEquals(0.0, GymDistance.meters(30.0, 120.0, 30.0, 120.0), 0.0)
        assertTrue(CoordinatePrecision.STATION.ranks && !CoordinatePrecision.DISTRICT.ranks && !CoordinatePrecision.CITY.ranks)
        assertNull(CoordinatePrecision.of("POI")); assertNull(CoordinatePrecision.of(null))
    }
}
