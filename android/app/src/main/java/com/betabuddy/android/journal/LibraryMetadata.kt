package com.betabuddy.android.journal

/** ISO 6709 coordinates in a video's existing metadata; never the phone's current position. */
internal object LibraryMetadata {
    fun location(value: String?): Pair<Double,Double>? {
        val match = value?.let { Regex("^([+-][0-9]{2}(?:\\.[0-9]+)?)([+-][0-9]{3}(?:\\.[0-9]+)?)(?:[+-][0-9]+(?:\\.[0-9]+)?)?/?$").matchEntire(it.trim()) } ?: return null
        val lat = match.groupValues[1].toDoubleOrNull() ?: return null
        val lon = match.groupValues[2].toDoubleOrNull() ?: return null
        return (lat to lon).takeIf { lat in -90.0..90.0 && lon in -180.0..180.0 && (lat != 0.0 || lon != 0.0) }
    }
    fun suggestedGym(environment: String?, location: Pair<Double,Double>?, gyms: List<DirectoryGym>): DirectoryGym? {
        if (environment != "indoor" || location == null) return null
        return GymDistance.suggestions(gyms,location.first,location.second,0.0).ranked.firstOrNull()?.gym
    }
}
