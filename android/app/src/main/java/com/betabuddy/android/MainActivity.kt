package com.betabuddy.android

import android.content.Intent
import android.graphics.Color
import android.os.Bundle
import androidx.lifecycle.lifecycleScope
import com.betabuddy.android.journal.AnalysisService
import com.betabuddy.android.journal.JournalGraph
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import androidx.activity.SystemBarStyle
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import com.betabuddy.android.capture.CaptureDebug
import com.betabuddy.android.capture.CaptureFrameInfo
import com.betabuddy.android.capture.GymLocator
import com.betabuddy.android.gym.GymDebugOptions
import java.io.File
import com.betabuddy.android.gym.GymViewModel
import com.betabuddy.android.ui.nav.AppShell
import com.betabuddy.android.ui.nav.Tabs
import com.betabuddy.android.ui.theme.BetaBuddyTheme

/**
 * The single activity. AppCompat is here for one thing: the per-app language setting
 * (`AppCompatDelegate.setApplicationLocales`), which needs an AppCompat activity and theme.
 */
class MainActivity : AppCompatActivity() {
    private val gym: GymViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        // Edge to edge with dark icons: the app is light only, paper under both bars.
        enableEdgeToEdge(
            statusBarStyle = SystemBarStyle.light(Color.TRANSPARENT, Color.TRANSPARENT),
            navigationBarStyle = SystemBarStyle.light(Color.TRANSPARENT, Color.TRANSPARENT),
        )
        super.onCreate(savedInstanceState)

        var initialTab = Tabs.GYM
        var initialCapture = false
        if (BuildConfig.DEBUG && savedInstanceState == null) {
            // Debug builds only: put the app in a given state for screenshots, e.g.
            //   adb shell am start -n com.betabuddy.android/.MainActivity --ei initialTab 1
            //   adb shell am start -n com.betabuddy.android/.MainActivity --ez gymNoticeSeen true \
            //       --es gymQuery "sheet=half&tab=steps" --es gymOpenRoute <id>
            initialTab = intent.intExtra("initialTab") ?: Tabs.GYM
            gym.session.debug = GymDebugOptions(
                query = intent.getStringExtra("gymQuery"),
                openRoute = intent.getStringExtra("gymOpenRoute"),
                openWall = intent.getStringExtra("gymOpenWall"),
            )
            if (intent.flagExtra("gymNoticeSeen")) gym.session.acknowledgeNotice()
            // The in-app camera where there is no camera worth pointing at anything:
            //   --ez openCapture true                     open the capture screen at once (with --ei initialTab 1)
            //   --es debugCaptureVideo <file>             play this file (a path, or a name under the app's files directory) in place of the camera
            //   --es debugCaptureFov <degrees>            the horizontal field of view to report for it
            //   --es debugCaptureAttitude "<pitch>,<roll>,<shake>"   how the phone is held, where there are no motion sensors
            //   --es debugLocation "<lat>,<lon>[,<accuracy m>]"      a position fix; nothing is asked
            initialCapture = intent.flagExtra("openCapture")
            intent.getStringExtra("debugCaptureVideo")?.let { path -> CaptureDebug.video = File(path).takeIf { it.isAbsolute } ?: File(filesDir, path) }
            intent.getStringExtra("debugCaptureFov")?.toDoubleOrNull()?.let { CaptureDebug.fieldOfView = it }
            intent.getStringExtra("debugCaptureAttitude")?.split(",")?.mapNotNull { it.trim().toDoubleOrNull() }?.takeIf { it.size == 3 }
                ?.let { CaptureDebug.attitude = CaptureFrameInfo(pitch = it[0], roll = it[1], shake = it[2]) }
            intent.getStringExtra("debugLocation")?.split(",")?.mapNotNull { it.trim().toDoubleOrNull() }?.takeIf { it.size >= 2 }
                ?.let { CaptureDebug.location = GymLocator.Fix(it[0], it[1], it.getOrNull(2) ?: 30.0) }
        }

        setContent {
            BetaBuddyTheme {
                AppShell(gym = gym.session, initialTab = initialTab, initialCapture = initialCapture)
            }
        }
    }

    override fun onStart() {
        super.onStart()
        lifecycleScope.launch {
            val pending = withContext(Dispatchers.IO) { JournalGraph.store(this@MainActivity).climbs.any { it.pendingAnalysis != null } }
            if (pending) AnalysisService.start(this@MainActivity)
        }
    }

    // `am start` can pass a value as --ei, --ez or --es; accept any of them.
    private fun Intent.intExtra(name: String): Int? =
        if (!hasExtra(name)) null else getStringExtra(name)?.toIntOrNull() ?: getIntExtra(name, 0)

    private fun Intent.flagExtra(name: String): Boolean =
        hasExtra(name) && (getBooleanExtra(name, false) || getIntExtra(name, 0) == 1 || getStringExtra(name) in setOf("1", "true"))
}
