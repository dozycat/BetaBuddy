package com.betabuddy.android.journal

import android.accessibilityservice.AccessibilityServiceInfo
import android.app.UiAutomation
import android.content.Intent
import android.graphics.Bitmap
import android.view.InputDevice
import android.view.InputEvent
import android.view.MotionEvent
import android.view.accessibility.AccessibilityManager
import android.view.accessibility.AccessibilityNodeInfo
import androidx.activity.compose.setContent
import androidx.lifecycle.ViewModelProvider
import androidx.test.core.app.ActivityScenario
import androidx.test.platform.app.InstrumentationRegistry
import com.betabuddy.android.MainActivity
import com.betabuddy.android.R
import com.betabuddy.android.ui.theme.BetaBuddyTheme
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.atomic.AtomicBoolean
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test

// Deliberately has no Compose test rule: TalkBack drives real frames and physical gestures.
class LibraryTalkBackTest {
    /** Enable TalkBack and grant video access on a disposable emulator before running. */
    @Test fun optionalTalkBackSwipesReachAndActivateLibraryControls() {
        assumeTrue(InstrumentationRegistry.getArguments().getString("talkBack") == "1")
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val automation = instrumentation.getUiAutomation(UiAutomation.FLAG_DONT_SUPPRESS_ACCESSIBILITY_SERVICES)
        val manager = context.getSystemService(AccessibilityManager::class.java)
        fun await(message: String, condition: () -> Boolean) {
            val end = android.os.SystemClock.elapsedRealtime()+15_000
            while (!condition() && android.os.SystemClock.elapsedRealtime()<end) android.os.SystemClock.sleep(100)
            assertTrue(message,condition())
        }
        await("TalkBack must remain enabled during this test") {
            manager.isTouchExplorationEnabled && manager.getEnabledAccessibilityServiceList(AccessibilityServiceInfo.FEEDBACK_SPOKEN)
                .any { it.resolveInfo.serviceInfo.packageName == "com.google.android.marvin.talkback" }
        }
        assertTrue("Grant video access before this control-navigation test",LibraryPermission.access(context) != LibraryAccess.NONE)
        // Normal injection bypasses accessibility. Use the platform test hook so TalkBack receives gestures.
        val inject = UiAutomation::class.java.getMethod("injectInputEventToInputFilter",InputEvent::class.java)
        val preferences = context.getSharedPreferences("library-import",0)
        val prior = preferences.getBoolean("enabled",false)
        preferences.edit().putBoolean("enabled",false).commit()
        val closed = AtomicBoolean(false)
        val scenario = ActivityScenario.launch<MainActivity>(Intent(context,MainActivity::class.java).putExtra("initialTab",1))
        val output = File(context.getExternalFilesDir(null),"talkback-library").apply { mkdirs() }
        val focused = mutableListOf<String>()
        fun label(node: AccessibilityNodeInfo?): String = node?.let {
            (listOfNotNull(it.text,it.contentDescription,it.stateDescription).map { value -> value.toString() } +
                (0 until it.childCount).mapNotNull { index -> it.getChild(index)?.let(::label) }).joinToString(" | ")
        } ?: "<none>"
        fun nodes(node: AccessibilityNodeInfo?): List<AccessibilityNodeInfo> = node?.let {
            listOf(it) + (0 until it.childCount).flatMap { index -> nodes(it.getChild(index)) }
        } ?: emptyList()
        fun screenshot(name: String) {
            automation.takeScreenshot()?.let { bitmap ->
                FileOutputStream(File(output,name)).use { bitmap.compress(Bitmap.CompressFormat.PNG,100,it) }; bitmap.recycle()
            }
        }
        fun touch(action: Int, down: Long, at: Long, x: Float, y: Float) {
            val event = MotionEvent.obtain(down,at,action,x,y,0).apply { source = InputDevice.SOURCE_TOUCHSCREEN }
            try { inject.invoke(automation,event) } finally { event.recycle() }
        }
        fun swipe() {
            val bounds = android.graphics.Rect().also { automation.rootInActiveWindow.getBoundsInScreen(it) }
            val down = android.os.SystemClock.uptimeMillis()
            val x = bounds.width()*.35f; val y = bounds.height()*.55f
            touch(MotionEvent.ACTION_DOWN,down,down,x,y)
            for (step in 1..10) {
                android.os.SystemClock.sleep(12)
                touch(MotionEvent.ACTION_MOVE,down,android.os.SystemClock.uptimeMillis(),x+bounds.width()*.3f*step/10,y)
            }
            touch(MotionEvent.ACTION_UP,down,android.os.SystemClock.uptimeMillis(),x+bounds.width()*.3f,y)
            android.os.SystemClock.sleep(450)
            focused += label(automation.rootInActiveWindow?.findFocus(AccessibilityNodeInfo.FOCUS_ACCESSIBILITY))
        }
        fun doubleTap() {
            val bounds = android.graphics.Rect().also { automation.rootInActiveWindow.getBoundsInScreen(it) }
            repeat(2) {
                val down = android.os.SystemClock.uptimeMillis()
                touch(MotionEvent.ACTION_DOWN,down,down,bounds.width()*.5f,bounds.height()*.55f)
                android.os.SystemClock.sleep(40)
                touch(MotionEvent.ACTION_UP,down,android.os.SystemClock.uptimeMillis(),bounds.width()*.5f,bounds.height()*.55f)
                android.os.SystemClock.sleep(70)
            }
        }
        try {
            scenario.onActivity { activity ->
                val model = ViewModelProvider(activity)[LibraryImport::class.java]
                model.setEnabled(false)
                activity.setContent { BetaBuddyTheme { LibraryImportScreen(model) { closed.set(true) } } }
            }
            await("Library switch must be visible") {
                nodes(automation.rootInActiveWindow).any { it.isCheckable && label(it).contains(context.getString(R.string.library_enable)) }
            }
            android.os.SystemClock.sleep(600)
            screenshot("before.png")
            var found = false
            for (step in 0 until 10) {
                swipe()
                val node = automation.rootInActiveWindow?.findFocus(AccessibilityNodeInfo.FOCUS_ACCESSIBILITY)
                if (node?.isCheckable == true && label(node).contains(context.getString(R.string.library_enable))) { found = true; break }
            }
            assertTrue("Swipe navigation must reach the named switch: $focused",found)
            screenshot("switch-focused.png")
            doubleTap()
            await("TalkBack double tap must enable the switch") { preferences.getBoolean("enabled",false) }
            screenshot("enabled.png")
            doubleTap()
            await("TalkBack double tap must disable the switch") { !preferences.getBoolean("enabled",true) }
            found = false
            for (step in 0 until 10) {
                swipe()
                if (label(automation.rootInActiveWindow?.findFocus(AccessibilityNodeInfo.FOCUS_ACCESSIBILITY)).contains(context.getString(R.string.album_done))) { found = true; break }
            }
            assertTrue("Swipe navigation must reach Done: $focused",found)
            doubleTap(); await("Done must activate") { closed.get() }
            assertTrue(manager.isTouchExplorationEnabled)
        } finally {
            File(output,"focus.txt").writeText(focused.joinToString("\n"))
            scenario.onActivity { ViewModelProvider(it)[LibraryImport::class.java].setEnabled(false) }
            scenario.close(); preferences.edit().putBoolean("enabled",prior).commit()
        }
    }

}
