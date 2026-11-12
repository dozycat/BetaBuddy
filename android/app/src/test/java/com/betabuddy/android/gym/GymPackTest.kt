package com.betabuddy.android.gym

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

// Fixtures here are synthetic. Real venue data never enters this repository.
class GymPackTest {
    private class FakeFiles(val files: Map<String, String>) : GymPackFiles {
        override fun readText(path: String): String? = files[path]
        override fun exists(path: String): Boolean = path in files
    }

    private val manifest = """
        {"schema":1,"packVersion":"0.0.1","generatedAt":"2026-01-02T03:04:05Z","gymName":"测试岩馆",
         "gymNameLocalized":{"en":"Test Gym","ja":""},"routeCount":7,"languages":["zh-Hans","en"],
         "entry":"index.html","files":[{"path":"index.html","sha256":"aa","bytes":7},{"path":"data/scene.bin","sha256":"bb","bytes":7}],
         "somethingNew":{"ignored":true}}
    """.trimIndent()

    private fun pack(manifest: String = this.manifest, files: List<String> = listOf("index.html", "data/scene.bin")) =
        GymPack.load(FakeFiles(files.associateWith { "fixture" } + (GymPack.MANIFEST to manifest)))

    private fun ready(): GymPack = (pack() as GymPackStatus.Ready).pack

    @Test fun manifestDecodes() {
        val decoded = GymManifest.parse(manifest)
        assertEquals("0.0.1", decoded.packVersion)
        assertEquals("2026-01-02T03:04:05Z", decoded.generatedAt)
        assertEquals("2026-01-02", decoded.generatedDate)
        assertEquals("测试岩馆", decoded.gymName)
        assertEquals(7, decoded.routeCount)
        assertEquals("index.html", decoded.entry)
        assertEquals(listOf("data/scene.bin", "index.html"), decoded.files.map { it.path })
        assertEquals(listOf("zh-Hans", "en"), decoded.languages)
    }

    @Test fun manifestAcceptsSnakeCaseAndFileMaps() {
        val decoded = GymManifest.parse("""{"pack_version":"2","generated_at":"2026-01-02","gym":{"name":"G"},"route_count":3,"files":{"index.html":"aa","b.js":{"sha256":"bb"}}}""")
        assertEquals(1, decoded.schema)
        assertEquals("G", decoded.gymName)
        assertEquals(3, decoded.routeCount)
        assertEquals("index.html", decoded.entry)
        assertEquals(listOf(GymFile("b.js", "bb"), GymFile("index.html", "aa")), decoded.files)
    }

    @Test fun manifestWithoutRequiredFieldIsRejected() {
        for (broken in listOf(
            """{"packVersion":"1","generatedAt":"2026-01-02","routeCount":3,"files":[]}""",
            """{"packVersion":"1","generatedAt":"2026-01-02","gymName":"G","routeCount":"3","files":[]}""",
            """{"packVersion":"1","generatedAt":"2026-01-02","gymName":"G","routeCount":3}""",
            """{"packVersion":"1","generatedAt":"2026-01-02","gymName":"G","routeCount":3,"files":[{"sha256":"aa"}]}""",
            "[]", "not json", "",
        )) {
            val error = runCatching { GymManifest.parse(broken) }.exceptionOrNull()
            assertEquals(broken, GymPackError.UnreadableManifest, (error as? GymPackException)?.error)
        }
    }

    @Test fun displayNameFollowsTheInterfaceLanguage() {
        val decoded = GymManifest.parse(manifest)
        assertEquals("Test Gym", decoded.displayName("en"))
        assertEquals("测试岩馆", decoded.displayName("ja"))      // Empty translation: the pack's own name.
        assertEquals("测试岩馆", decoded.displayName("ko"))
    }

    @Test fun statusReflectsWhatIsInTheBuild() {
        assertEquals(GymPackStatus.Missing, GymPack.load(FakeFiles(emptyMap())))
        assertTrue(pack() is GymPackStatus.Ready)
        assertEquals(GymPackStatus.Invalid(GymPackError.UnreadableManifest), pack(manifest = "{}"))
        assertEquals(GymPackStatus.Invalid(GymPackError.UnsupportedSchema(2)), pack(manifest = manifest.replace("\"schema\":1", "\"schema\":2")))
        assertEquals(GymPackStatus.Invalid(GymPackError.MissingFile("data/scene.bin")), pack(files = listOf("index.html")))
        assertEquals(
            GymPackStatus.Invalid(GymPackError.UnsafePath("../secret")),
            pack(manifest = manifest.replace("data/scene.bin", "../secret"), files = listOf("index.html", "../secret")),
        )
    }

    @Test fun pathsMustStayInsideThePack() {
        for (path in listOf("index.html", "data/scene.bin", "a.b/c-d_e.js")) assertTrue(path, GymPack.isSafe(path))
        for (path in listOf("", "/etc/passwd", "../x", "a/../b", "a//b", "./a", "a/.", "a\\b", "a/")) assertFalse(path, GymPack.isSafe(path))
    }

    @Test fun onlyThePackAddressIsAllowed() {
        for (url in listOf(
            "https://appassets.androidplatform.net/gympack/index.html",
            "https://appassets.androidplatform.net/gympack/data/scene.bin?x=1#y",
            "HTTPS://APPASSETS.ANDROIDPLATFORM.NET/gympack/index.html",
        )) assertTrue(url, GymPack.allows(url))
        for (url in listOf(
            null, "", "about:blank", "data:text/html,hi", "file:///android_asset/gympack/index.html",
            "http://appassets.androidplatform.net/gympack/index.html",
            "https://appassets.androidplatform.net/index.html",
            "https://appassets.androidplatform.net/gympackx/index.html",
            "https://appassets.androidplatform.net:8443/gympack/index.html",
            "https://user@appassets.androidplatform.net/gympack/index.html",
            "https://appassets.androidplatform.net.example.com/gympack/index.html",
            "https://example.com/gympack/index.html", "wss://example.com/socket", "ftp://example.com/x", "not a url",
        )) assertFalse(url.toString(), GymPack.allows(url))
    }

    @Test fun entryUrlCarriesHintLanguageAndDebugQuery() {
        val pack = ready()
        val base = "https://appassets.androidplatform.net/gympack/index.html"
        assertEquals(base, pack.entryUrl)
        assertEquals(base, pack.entryUrl(hintSeen = false))
        assertEquals("$base?hint=0", pack.entryUrl(hintSeen = true))
        assertEquals("$base?lang=ja", pack.entryUrl(hintSeen = false, language = "ja"))
        assertEquals("$base?hint=0&lang=zh-Hant", pack.entryUrl(hintSeen = true, language = "zh-Hant"))
        assertEquals("$base?hint=0&lang=en&sheet=half&tab=steps", pack.entryUrl(hintSeen = true, language = "en", extraQuery = "?sheet=half&tab=steps&"))
        assertEquals("$base?lang=en", pack.entryUrl(hintSeen = false, language = "en", extraQuery = "a b|c"))   // Not a URL: dropped.
        assertEquals("$base?lang=en", pack.entryUrl(hintSeen = false, language = "en", extraQuery = " "))
    }

    @Test fun onlyListedFilesAreServed() {
        assertEquals(setOf("index.html", "data/scene.bin"), ready().servedPaths)
    }

    @Test fun contentTypesFollowTheExtension() {
        assertEquals("text/html" to "utf-8", GymPack.contentType("index.html"))
        assertEquals("text/javascript" to "utf-8", GymPack.contentType("a/b.MJS"))
        assertEquals("model/gltf-binary" to null, GymPack.contentType("venue.glb"))
        assertEquals("application/wasm" to null, GymPack.contentType("x.wasm"))
        assertEquals("application/octet-stream" to null, GymPack.contentType("scene.bin"))
        assertEquals("application/octet-stream" to null, GymPack.contentType("noextension"))
    }

    @Test fun messagesParse() {
        assertEquals(GymMessage.Ready, GymMessage.parse("""{"type":"ready","packVersion":"1","routeCount":3}"""))
        assertEquals(GymMessage.HintSeen, GymMessage.parse("""{"type":"hintSeen"}"""))
        assertEquals(GymMessage.OpenSettings, GymMessage.parse("""{"type":"openSettings"}"""))
        assertEquals(
            GymMessage.RouteSelected(GymRoute("r1", "Route one")),
            GymMessage.parse("""{"type":"routeSelected","id":"r1","name":"Route one","nameZh":"一","wall":"w","grade":"","lang":"en"}"""),
        )
        assertEquals(GymMessage.RouteSelected(GymRoute("r1", "r1")), GymMessage.parse("""{"type":"routeSelected","id":"r1"}"""))
        assertEquals(GymMessage.RouteSelected(null), GymMessage.parse("""{"type":"routeSelected"}"""))
        assertEquals(GymMessage.Error("boom"), GymMessage.parse("""{"type":"error","message":"boom"}"""))
        assertEquals(GymMessage.Error(""), GymMessage.parse("""{"type":"error"}"""))
        assertEquals(300, (GymMessage.parse("""{"type":"error","message":"${"x".repeat(999)}"}""") as GymMessage.Error).message.length)
        for (ignored in listOf(null, "", "ready", "[]", """{"type":"other"}""", """{"type":7}""", """{"id":"r1"}""", "x".repeat(20_000))) {
            assertNull(GymMessage.parse(ignored))
        }
    }

    @Test fun commandsQuoteTheirArgument() {
        assertEquals(
            "window.BetaBuddyGym && typeof window.BetaBuddyGym.openRoute === 'function' ? (window.BetaBuddyGym.openRoute(\"r1\"), true) : false",
            GymCommand.OpenRoute("r1").script,
        )
        val hostile = GymCommand.OpenWall("\"); alert(1); (\"\n").script
        assertTrue(hostile, hostile.contains("""openWall("\"); alert(1); (\"\n")"""))
    }

    @Test fun shimMapsTheIosHandlerOntoTheNativeBridge() {
        assertTrue(GymBridge.shim.contains("window.${GymBridge.NATIVE_NAME}"))
        assertTrue(GymBridge.shim.contains("handlers.gym"))
        assertTrue(GymBridge.shim.contains("JSON.stringify"))
    }
}
