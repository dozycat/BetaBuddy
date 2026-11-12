package com.betabuddy.android.gym

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive

/** A route the user picked inside the pack page. Only the id and display name cross the bridge. */
data class GymRoute(val id: String, val name: String)

/** Page → app messages, the ones the page posts to its `gym` message handler. */
sealed interface GymMessage {
    data object Ready : GymMessage
    data object HintSeen : GymMessage
    /** The page asks for the app's settings (its own "language and settings" item). */
    data object OpenSettings : GymMessage
    data class RouteSelected(val route: GymRoute?) : GymMessage
    data class Error(val message: String) : GymMessage

    companion object {
        private const val MAX_LENGTH = 16_384

        /** Parses the JSON text of one message. Unknown types and malformed input give null; extra fields are ignored. */
        fun parse(json: String?): GymMessage? {
            if (json == null || json.length > MAX_LENGTH) return null
            val body = try { Json.parseToJsonElement(json) as? JsonObject } catch (error: Exception) { null } ?: return null
            fun string(name: String) = (body[name] as? JsonPrimitive)?.takeIf { it.isString }?.content
            return when (string("type")) {
                "ready" -> Ready
                "hintSeen" -> HintSeen
                "openSettings" -> OpenSettings
                "routeSelected" -> {
                    val id = string("id")
                    RouteSelected(if (id.isNullOrEmpty()) null else GymRoute(id.take(120), (string("name") ?: id).take(120)))
                }
                "error" -> Error(string("message").orEmpty().take(300))
                else -> null
            }
        }
    }
}

/** App → page calls. The page exposes `window.BetaBuddyGym.openWall(id)` / `.openRoute(id)`. */
sealed interface GymCommand {
    data class OpenWall(val id: String) : GymCommand
    data class OpenRoute(val id: String) : GymCommand

    val script: String
        get() {
            val (function, id) = when (this) {
                is OpenWall -> "openWall" to id
                is OpenRoute -> "openRoute" to id
            }
            // JSON string encoding keeps ids from breaking out of the call.
            val argument = JsonPrimitive(id).toString()
            return "window.BetaBuddyGym && typeof window.BetaBuddyGym.$function === 'function' ? (window.BetaBuddyGym.$function($argument), true) : false"
        }
}

/**
 * The page was written for the iOS shell and posts to `window.webkit.messageHandlers.gym`.
 * This runs before any page script and gives it that object, backed by the native bridge.
 */
object GymBridge {
    /** Name of the object the app injects into the page. */
    const val NATIVE_NAME = "BetaBuddyGymNative"

    val shim: String = """
        (function () {
          if (window.top !== window) return;
          var native = window.$NATIVE_NAME;
          if (!native || typeof native.postMessage !== 'function') return;
          var webkit = window.webkit || (window.webkit = {});
          var handlers = webkit.messageHandlers || (webkit.messageHandlers = {});
          if (handlers.gym) return;
          handlers.gym = {
            postMessage: function (message) {
              try { native.postMessage(JSON.stringify(message)); } catch (error) {}
            }
          };
        })();
    """.trimIndent()
}
