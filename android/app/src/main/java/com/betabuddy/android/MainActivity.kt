package com.betabuddy.android

import android.content.Intent
import android.graphics.Color
import android.os.Bundle
import androidx.activity.SystemBarStyle
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import com.betabuddy.android.gym.GymDebugOptions
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
        }

        setContent {
            BetaBuddyTheme {
                AppShell(gym = gym.session, initialTab = initialTab)
            }
        }
    }

    // `am start` can pass a value as --ei, --ez or --es; accept any of them.
    private fun Intent.intExtra(name: String): Int? =
        if (!hasExtra(name)) null else getStringExtra(name)?.toIntOrNull() ?: getIntExtra(name, 0)

    private fun Intent.flagExtra(name: String): Boolean =
        hasExtra(name) && (getBooleanExtra(name, false) || getIntExtra(name, 0) == 1 || getStringExtra(name) in setOf("1", "true"))
}
