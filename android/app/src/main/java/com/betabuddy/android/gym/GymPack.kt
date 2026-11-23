package com.betabuddy.android.gym

import java.net.URI

/** Where a pack's files come from. The app reads them from its assets; tests use a map. */
interface GymPackFiles {
    /** The text of a file in the pack directory, or null when it is not there or cannot be read. */
    fun readText(path: String): String?

    /** Whether a regular file exists at this pack-relative path. */
    fun exists(path: String): Boolean
}

sealed interface GymPackStatus {
    /** This build ships without a pack, e.g. a clone of the public repository. */
    data object Missing : GymPackStatus
    data class Invalid(val error: GymPackError) : GymPackStatus
    data class Ready(val pack: GymPack) : GymPackStatus
}

/**
 * A gym pack found in the app. Hashes are verified by `android/scripts/install-gym-pack.sh` before
 * the build and the APK is signed afterwards, so launch only checks structure and presence.
 */
class GymPack(val manifest: GymManifest) {
    /** The only paths the web view is ever served: what the manifest lists, plus the entry. */
    val servedPaths: Set<String> = manifest.files.mapTo(mutableSetOf(manifest.entry)) { it.path }

    /** `https://appassets.androidplatform.net/gympack/<entry>`: a local address, answered from the pack. */
    val entryUrl: String get() = URI("https", HOST, PATH_PREFIX + manifest.entry, null).toASCIIString()

    /**
     * Entry URL for a load. `hint=0` tells the page its first-run gesture hint has already been seen;
     * `lang` is the app's interface language. [extraQuery] is appended as given (debug builds only
     * pass one); if the result is not a pack URL it is dropped.
     */
    fun entryUrl(hintSeen: Boolean, language: String? = null, extraQuery: String? = null): String {
        val extra = extraQuery.orEmpty().trim('?', '&', ' ')
        val known = listOfNotNull("hint=0".takeIf { hintSeen }, language?.let { "lang=$it" })
        fun url(items: List<String>): String? =
            if (items.isEmpty()) entryUrl else (entryUrl + "?" + items.joinToString("&")).takeIf(::allows)
        return (if (extra.isEmpty()) null else url(known + extra)) ?: url(known) ?: entryUrl
    }

    /**
     * Address of the pack's video analysis page in the given interface language, or null when this
     * pack does not list one. No page, no analysis entry anywhere in the app.
     */
    fun analysisUrl(language: String): String? {
        if (manifest.files.none { it.path == ANALYSIS_PAGE }) return null
        return URI("https", HOST, PATH_PREFIX + ANALYSIS_PAGE, "lang=$language", null).toASCIIString().takeIf(::allows)
    }

    /**
     * Whether this pack carries the capture page. Without it the camera still records and saves;
     * there is just no check of the camera position and nothing is recognised.
     */
    val supportsCapture: Boolean get() = manifest.files.any { it.path == CAPTURE_PAGE }

    /** Address of the pack's capture page, loaded the same way as the entry page, or null when the pack has none. */
    fun captureUrl(language: String): String? {
        if (!supportsCapture) return null
        return URI("https", HOST, PATH_PREFIX + CAPTURE_PAGE, "lang=$language", null).toASCIIString().takeIf(::allows)
    }

    companion object {
        /** Directory under the app's assets. Ignored by Git; filled by the install script. */
        const val ASSET_DIRECTORY = "gympack"
        const val MANIFEST = "manifest.json"
        const val ANALYSIS_PAGE = "analysis.html"
        const val CAPTURE_PAGE = "capture.html"

        /** WebViewAssetLoader's reserved domain. It is never resolved; requests to it are answered locally. */
        const val HOST = "appassets.androidplatform.net"
        const val ORIGIN = "https://$HOST"
        const val PATH_PREFIX = "/gympack/"

        fun load(files: GymPackFiles): GymPackStatus {
            val text = files.readText(MANIFEST) ?: return if (files.exists(MANIFEST)) {
                GymPackStatus.Invalid(GymPackError.UnreadableManifest)
            } else {
                GymPackStatus.Missing
            }
            return try {
                val manifest = GymManifest.parse(text)
                if (manifest.schema != GymManifest.SUPPORTED_SCHEMA) throw GymPackException(GymPackError.UnsupportedSchema(manifest.schema))
                for (path in listOf(manifest.entry) + manifest.files.map { it.path }) {
                    if (!isSafe(path)) throw GymPackException(GymPackError.UnsafePath(path))
                    if (!files.exists(path)) throw GymPackException(GymPackError.MissingFile(path))
                }
                GymPackStatus.Ready(GymPack(manifest))
            } catch (error: GymPackException) {
                GymPackStatus.Invalid(error.error)
            }
        }

        /** Relative, inside the pack, no parent traversal. */
        fun isSafe(path: String): Boolean =
            path.isNotEmpty() && !path.startsWith("/") && '\\' !in path && '\u0000' !in path &&
                path.split("/").none { it.isEmpty() || it == "." || it == ".." }

        /** Only the pack's own local address may be loaded. Everything else, including any network URL, is refused. */
        fun allows(url: String?): Boolean {
            val uri = try { URI(url ?: return false) } catch (error: Exception) { return false }
            return uri.scheme.equals("https", ignoreCase = true) && uri.host.equals(HOST, ignoreCase = true) &&
                (uri.port == -1 || uri.port == 443) && uri.userInfo == null && uri.path.orEmpty().startsWith(PATH_PREFIX)
        }

        /** MIME type and, for text, the charset. The same table as the iOS app. */
        fun contentType(path: String): Pair<String, String?> = when (path.substringAfterLast('.', "").lowercase()) {
            "html", "htm" -> "text/html" to "utf-8"
            "js", "mjs" -> "text/javascript" to "utf-8"
            "json", "map" -> "application/json" to "utf-8"
            "css" -> "text/css" to "utf-8"
            "glb" -> "model/gltf-binary" to null
            "gltf" -> "model/gltf+json" to null
            "wasm" -> "application/wasm" to null
            "png" -> "image/png" to null
            "jpg", "jpeg" -> "image/jpeg" to null
            "webp" -> "image/webp" to null
            "svg" -> "image/svg+xml" to null
            "ktx2" -> "image/ktx2" to null
            "txt" -> "text/plain" to "utf-8"
            "md" -> "text/markdown" to "utf-8"
            else -> "application/octet-stream" to null   // .bin and anything unknown
        }
    }
}
