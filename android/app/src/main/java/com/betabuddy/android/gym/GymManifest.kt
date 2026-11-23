package com.betabuddy.android.gym

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.intOrNull

/** Why a pack that is present cannot be used. */
sealed interface GymPackError {
    data object UnreadableManifest : GymPackError
    data class UnsupportedSchema(val schema: Int) : GymPackError
    data class UnsafePath(val path: String) : GymPackError
    data class MissingFile(val path: String) : GymPackError
}

class GymPackException(val error: GymPackError) : Exception(error.toString())

data class GymFile(val path: String, val sha256: String)

/**
 * `manifest.json` of a gym pack. The pack itself (model, routes, solver, web page) is built outside
 * this repository and is never checked in; this type is the whole native-side contract.
 * See docs/ios/gym-design.md. Both camelCase and snake_case keys are accepted, as on iOS.
 */
data class GymManifest(
    val schema: Int,
    val packVersion: String,
    val generatedAt: String,
    val gymName: String,
    /** Optional per-language names, keyed by the app's language codes. */
    val gymNameLocalized: Map<String, String>,
    val routeCount: Int,
    val entry: String,
    val files: List<GymFile>,
    /** Languages the page says it can show. Informational; the app always passes its own `lang`. */
    val languages: List<String>,
    /** The gyms this pack carries, when it says so. Only used to tell the capture page which one the user is in. */
    val venues: List<Venue> = emptyList(),
) {
    /**
     * One gym of the pack. [name] and [names] (per language) are only used to recognise an entry of
     * the offline directory; [directoryIds] are that directory's ids for this gym, when the pack gives them.
     */
    data class Venue(val id: String, val name: String? = null, val names: Map<String, String> = emptyMap(), val directoryIds: List<String> = emptyList())

    /**
     * The pack's venue a gym from the offline directory stands for, or null when it is not one of
     * them. A pack that lists the directory's ids for its venues is taken at its word. Otherwise
     * this goes by name: a directory entry is written "brand（branch）", and it is taken to be a
     * venue when exactly one venue's name carries both parts. Anything less certain is left out;
     * the capture page then works without a venue.
     */
    fun venueId(gymId: String?, gymName: String?): String? {
        if (!gymId.isNullOrEmpty()) {
            val listed = venues.filter { gymId in it.directoryIds }
            if (listed.size == 1) return listed[0].id
            if (listed.size > 1) return null
        }
        fun plain(text: String) = text.lowercase().filter { it.isLetterOrDigit() }
        val parts = gymName.orEmpty().split('（', '）', '(', ')').map(::plain).filter { it.isNotEmpty() }
        if (parts.isEmpty()) return null
        val whole = parts.joinToString("")
        val matches = venues.filter { venue ->
            val known = (listOfNotNull(venue.name) + venue.names.values).map(::plain).filter { it.isNotEmpty() }
            // A brand alone says nothing about which branch it is.
            known.any { candidate -> candidate == whole || (parts.size >= 2 && parts.all { it in candidate }) }
        }
        return matches.singleOrNull()?.id
    }

    /** The gym's name in the interface language when the pack provides one; the pack's own name otherwise. */
    fun displayName(language: String): String = gymNameLocalized[language]?.takeIf { it.isNotEmpty() } ?: gymName

    /** The date part of `generatedAt`. */
    val generatedDate: String get() = generatedAt.take(10)

    companion object {
        const val SUPPORTED_SCHEMA = 1

        /** @throws GymPackException with [GymPackError.UnreadableManifest] when the text is not a usable manifest. */
        fun parse(text: String): GymManifest = try {
            decode(Json.parseToJsonElement(text) as? JsonObject ?: unreadable())
        } catch (error: GymPackException) {
            throw error
        } catch (error: Exception) {
            unreadable()
        }

        private fun unreadable(): Nothing = throw GymPackException(GymPackError.UnreadableManifest)

        private fun JsonElement?.string(): String? = (this as? JsonPrimitive)?.takeIf { it.isString }?.content
        private fun JsonElement?.int(): Int? = (this as? JsonPrimitive)?.takeIf { !it.isString }?.intOrNull

        private fun decode(root: JsonObject): GymManifest {
            fun string(vararg names: String) = names.firstNotNullOfOrNull { root[it].string() }
            fun int(vararg names: String) = names.firstNotNullOfOrNull { root[it].int() }

            val name = string("gymName", "gym_name") ?: (root["gym"] as? JsonObject)?.get("name").string()
            val localized = (root["gymNameLocalized"] as? JsonObject)
                ?.mapNotNull { (language, value) -> value.string()?.let { language to it } }?.toMap().orEmpty()
            val languages = (root["languages"] as? JsonArray)?.mapNotNull { it.string() }.orEmpty()

            // Either [{path, sha256}] or {path: sha256} / {path: {sha256}}.
            val files = when (val listed = root["files"]) {
                is JsonArray -> listed.map { item ->
                    val entry = item as? JsonObject ?: unreadable()
                    GymFile(entry["path"].string() ?: unreadable(), entry["sha256"].string() ?: unreadable())
                }
                is JsonObject -> listed.map { (path, value) ->
                    GymFile(path, value.string() ?: (value as? JsonObject)?.get("sha256").string() ?: unreadable())
                }
                else -> unreadable()
            }

            // Optional, and read leniently: a venue entry this build cannot read is left out.
            val venues = (root["venues"] as? JsonArray).orEmpty().mapNotNull { item ->
                val entry = item as? JsonObject ?: return@mapNotNull null
                Venue(
                    id = entry["id"].string()?.takeIf { it.isNotEmpty() } ?: return@mapNotNull null,
                    name = entry["name"].string(),
                    names = (entry["names"] as? JsonObject)?.mapNotNull { (language, value) -> value.string()?.let { language to it } }?.toMap().orEmpty(),
                    directoryIds = (entry["directoryIds"] as? JsonArray).orEmpty().mapNotNull { it.string() },
                )
            }

            return GymManifest(
                schema = int("schema", "schemaVersion", "schema_version") ?: 1,
                packVersion = string("packVersion", "pack_version", "version") ?: unreadable(),
                generatedAt = string("generatedAt", "generated_at") ?: unreadable(),
                gymName = name ?: unreadable(),
                gymNameLocalized = localized,
                routeCount = int("routeCount", "route_count") ?: unreadable(),
                entry = string("entry") ?: "index.html",
                files = files.sortedBy { it.path },
                languages = languages,
                venues = venues,
            )
        }
    }
}
