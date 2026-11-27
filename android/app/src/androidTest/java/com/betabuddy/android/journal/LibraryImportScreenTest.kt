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

class LibraryImportScreenTest {
    @get:Rule val rule = createEmptyComposeRule()
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
