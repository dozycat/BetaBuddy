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
) {
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
            )
        }
    }
}
