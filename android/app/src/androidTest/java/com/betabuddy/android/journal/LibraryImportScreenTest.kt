package com.betabuddy.android.journal

import android.Manifest
import android.content.Intent
import android.graphics.Bitmap
import android.os.Build
import androidx.compose.ui.test.*
import androidx.compose.ui.test.junit4.createEmptyComposeRule
import androidx.lifecycle.ViewModelProvider
import androidx.test.core.app.ActivityScenario
import androidx.test.platform.app.InstrumentationRegistry
import com.betabuddy.android.MainActivity
import com.betabuddy.android.R
import java.io.File
import java.io.FileOutputStream
import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import org.junit.Assume.assumeTrue
import android.view.accessibility.AccessibilityNodeInfo
import kotlinx.coroutines.runBlocking
import androidx.lifecycle.Lifecycle

class LibraryImportScreenTest {
    @get:Rule val rule = createEmptyComposeRule()

    /** Opt-in device check: use a disposable emulator with shell-owned videos and reset video permissions. */
    @Test fun selectedVideoPermissionUsesSystemPickerAndCanBeExpanded() {
        assumeTrue(Build.VERSION.SDK_INT >= 34)
        assumeTrue(InstrumentationRegistry.getArguments().getString("permissionMatrix") == "selected")
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val fixtures=InstrumentationRegistry.getArguments().getString("libraryVideoUris")?.split(",")?.map(android.net.Uri::parse)?.toSet()
        require(fixtures != null && fixtures.size == 3) { "Provide three shell-owned synthetic media URIs" }
        assertEquals(LibraryAccess.NONE,LibraryPermission.access(context))
        val preferences = context.getSharedPreferences("library-import",0)
        val prior = preferences.getBoolean("enabled",false)
        preferences.edit().putBoolean("enabled",false).commit()
        val scenario = ActivityScenario.launch<MainActivity>(Intent(context,MainActivity::class.java).putExtra("initialTab",1))
        fun tree(): List<AccessibilityNodeInfo> {
            fun children(node: AccessibilityNodeInfo): List<AccessibilityNodeInfo> = listOf(node) +
                (0 until node.childCount).flatMap { node.getChild(it)?.let(::children) ?: emptyList() }
            return instrumentation.uiAutomation.rootInActiveWindow?.let(::children) ?: emptyList()
        }
        fun awaitNode(predicate: (AccessibilityNodeInfo) -> Boolean): AccessibilityNodeInfo {
            val end = android.os.SystemClock.elapsedRealtime()+15_000
            while (android.os.SystemClock.elapsedRealtime()<end) {
                tree().firstOrNull(predicate)?.let { return it }
                android.os.SystemClock.sleep(100)
            }
            error(tree().joinToString("\n") { "${it.packageName} ${it.viewIdResourceName}: ${it.text} / ${it.contentDescription}" })
        }
        fun click(node: AccessibilityNodeInfo) {
            var target: AccessibilityNodeInfo? = node
            while (target != null && !target.isClickable) target=target.parent
            assertTrue(target?.performAction(AccessibilityNodeInfo.ACTION_CLICK) == true)
        }
        val output = File(context.getExternalFilesDir(null),"library-permissions").apply { mkdirs() }
        fun shot(name: String) {
            instrumentation.waitForIdleSync()
            android.os.SystemClock.sleep(300)
            instrumentation.uiAutomation.takeScreenshot()?.let { bitmap ->
                FileOutputStream(File(output,name)).use { bitmap.compress(Bitmap.CompressFormat.PNG,100,it) }; bitmap.recycle()
            }
        }
        fun checkHidden(allowed: Set<android.net.Uri>) {
            assertTrue(fixtures.containsAll(allowed))
            for (uri in fixtures-allowed) {
                assertTrue(runBlocking { LibraryMedia(context).videos(uri) }.isEmpty())
                var readable=false
                try { context.contentResolver.openFileDescriptor(uri,"r")?.use { readable=true } }
                catch (_: SecurityException) { }
                catch (_: java.io.FileNotFoundException) { }
                assertFalse("An unselected video must not be readable",readable)
            }
        }
        try {
            rule.waitUntil(20_000) { rule.onAllNodesWithTag("libraryImport").fetchSemanticsNodes().isNotEmpty() }
            rule.onNodeWithTag("libraryImport").performScrollTo().performClick()
            rule.onNodeWithTag("libraryImportToggle").performClick()
            scenario.onActivity { ViewModelProvider(it)[LibraryImport::class.java].stop() }
            val select=awaitNode { it.viewIdResourceName?.endsWith("/permission_allow_selected_button") == true }
            shot("request.png"); click(select)
            val thumbnail=awaitNode { it.viewIdResourceName?.endsWith("/icon_thumbnail") == true }
            shot("picker.png"); click(thumbnail)
            click(awaitNode { it.text?.toString()?.startsWith("Allow (") == true })
            awaitNode { it.packageName == context.packageName && it.text?.toString() == context.getString(R.string.library_heading) }
            rule.waitUntil(15_000) { LibraryPermission.access(context) == LibraryAccess.SELECTED }
            val first=runBlocking { LibraryMedia(context).videos() }.map { it.uri }.toSet()
            assertEquals("Only one externally owned video was selected",1,first.size)
            checkHidden(first)
            rule.onNodeWithText(context.getString(R.string.library_selected)).assertExists()
            shot("selected.png")
            rule.onNodeWithText(context.getString(R.string.library_reselect)).performScrollTo().performClick()
            click(awaitNode { it.viewIdResourceName?.endsWith("/permission_allow_selected_button") == true })
            awaitNode { it.viewIdResourceName?.endsWith("/icon_thumbnail") == true }
            val another=tree().filter { it.viewIdResourceName?.endsWith("/icon_thumbnail") == true }[1]
            click(another)
            click(awaitNode { it.text?.toString()?.startsWith("Allow (") == true })
            awaitNode { it.packageName == context.packageName && it.text?.toString() == context.getString(R.string.library_heading) }
            rule.waitUntil(15_000) { runBlocking { LibraryMedia(context).videos() }.size > first.size }
            val expanded=runBlocking { LibraryMedia(context).videos() }.map { it.uri }.toSet()
            assertTrue(expanded.containsAll(first)); assertEquals(2,expanded.size)
            checkHidden(expanded)
            assertEquals(LibraryAccess.SELECTED,LibraryPermission.access(context))
            scenario.moveToState(Lifecycle.State.CREATED)
            scenario.moveToState(Lifecycle.State.RESUMED)
            assertEquals(expanded,runBlocking { LibraryMedia(context).videos() }.map { it.uri }.toSet())
            rule.onNodeWithText(context.getString(R.string.library_selected)).assertExists()
            shot("expanded.png")
        } finally {
            scenario.onActivity { ViewModelProvider(it)[LibraryImport::class.java].setEnabled(false) }
            scenario.close(); preferences.edit().putBoolean("enabled",prior).commit()
        }
    }

    /** Run after revoking the preceding selected-video grant outside the test process. */
    @Test fun revokedVideoPermissionKeepsImportPausedAfterReopening() {
        assumeTrue(InstrumentationRegistry.getArguments().getString("permissionMatrix") == "revoked")
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        assertEquals(LibraryAccess.NONE,LibraryPermission.access(context))
        val preferences = context.getSharedPreferences("library-import",0)
        val prior = preferences.getBoolean("enabled",false)
        preferences.edit().putBoolean("enabled",true).commit()
        val store=runBlocking { JournalGraph.store(context) }
        val records=store.climbs.map { it.id }.toSet()
        val scenario = ActivityScenario.launch<MainActivity>(Intent(context,MainActivity::class.java).putExtra("initialTab",1))
        try {
            rule.waitUntil(20_000) { rule.onAllNodesWithTag("libraryImport").fetchSemanticsNodes().isNotEmpty() }
            rule.onNodeWithTag("libraryImport").performScrollTo().performClick()
            rule.onNodeWithTag("libraryImportToggle").assertIsOn()
            rule.onNodeWithText(context.getString(R.string.library_access_error)).assertExists()
            rule.onNodeWithText(context.getString(R.string.library_settings)).assertExists()
            rule.onNodeWithTag("libraryImportStatus").assertDoesNotExist()
            assertThrows(SecurityException::class.java) { runBlocking { LibraryMedia(context).videos() } }
            scenario.moveToState(Lifecycle.State.CREATED)
            scenario.moveToState(Lifecycle.State.RESUMED)
            scenario.onActivity {
                val state=ViewModelProvider(it)[LibraryImport::class.java].state.value
                assertEquals(LibraryAccess.NONE,state.access); assertFalse(state.running); assertEquals(0,state.imported)
            }
            assertEquals(records,store.climbs.map { it.id }.toSet())
        } finally {
            scenario.onActivity { ViewModelProvider(it)[LibraryImport::class.java].setEnabled(false) }
            scenario.close(); preferences.edit().putBoolean("enabled",prior).commit()
        }
    }
    @Test fun optInEntryAndStatusRemainReachable() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val preferences = context.getSharedPreferences("library-import",0)
        val prior = preferences.getBoolean("enabled",false)
        preferences.edit().putBoolean("enabled",false).commit()
        instrumentation.uiAutomation.grantRuntimePermission(context.packageName,if (Build.VERSION.SDK_INT >= 33) Manifest.permission.READ_MEDIA_VIDEO else Manifest.permission.READ_EXTERNAL_STORAGE)
        val scenario = ActivityScenario.launch<MainActivity>(Intent(context,MainActivity::class.java).putExtra("initialTab",1))
        try {
            rule.waitUntil(20_000) { rule.onAllNodesWithTag("libraryImport").fetchSemanticsNodes().isNotEmpty() }
            rule.onNodeWithTag("libraryImport").performScrollTo().performClick()
            rule.onNodeWithTag("libraryImportToggle").assertIsOff()
            val output = File(context.getExternalFilesDir(null),"library-shots").apply { mkdirs() }
            fun shot(name: String) {
                rule.waitForIdle(); instrumentation.waitForIdleSync()
                android.os.SystemClock.sleep(250) // Let the rendered frame catch up with Compose semantics.
                instrumentation.uiAutomation.takeScreenshot()?.let { bitmap ->
                FileOutputStream(File(output,name)).use { bitmap.compress(Bitmap.CompressFormat.PNG,100,it) }; bitmap.recycle()
            } }
            shot("disabled.png")
            rule.onNodeWithTag("libraryImportToggle").performClick()
            rule.onNodeWithTag("libraryImportToggle").assertIsOn()
            rule.waitUntil(60_000) { rule.onAllNodesWithTag("libraryScan").fetchSemanticsNodes().isNotEmpty() }
            rule.onNodeWithTag("libraryScan").performScrollTo().assertIsDisplayed()
            rule.onNodeWithText(context.getString(R.string.library_model_missing)).assertDoesNotExist()
            rule.onNodeWithText(context.getString(R.string.library_scan_error)).assertDoesNotExist()
            shot("ready.png")
            rule.onNodeWithTag("libraryImportToggle").performScrollTo().performClick()
            rule.onNodeWithTag("libraryImportToggle").assertIsOff()
            rule.onNodeWithText(context.getString(R.string.album_done)).performScrollTo().performClick()
            rule.onNodeWithTag("libraryImport").assertExists()
        } finally {
            scenario.onActivity { ViewModelProvider(it)[LibraryImport::class.java].setEnabled(false) }
            scenario.close(); preferences.edit().putBoolean("enabled",prior).commit()
        }
    }
}
