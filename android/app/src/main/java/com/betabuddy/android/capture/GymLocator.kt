package com.betabuddy.android.capture

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.Build
import android.os.Handler
import android.os.Looper
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.core.content.ContextCompat
import androidx.core.content.edit
import com.betabuddy.android.journal.DirectoryGym
import com.betabuddy.android.journal.GymDistance
import com.betabuddy.android.journal.GymSuggestions

/**
 * One position fix, taken only while the capture screen is choosing a gym, from the platform's own
 * location service with the approximate-location permission. The position is used to order the
 * offline directory and is then forgotten: it is not stored and goes nowhere.
 *
 * With approximate location the system reports a position a kilometre or two off, so on most
 * phones distances only order the list and are not shown as numbers (see [GymDistance.USEFUL_ACCURACY]).
 */
class GymLocator(context: Context, private val standIn: StandIn? = null) {
    enum class Access { NOT_ASKED, ALLOWED, REFUSED }
    data class Fix(val latitude: Double, val longitude: Double, val accuracy: Double)
    /** A made-up state in place of the system's, for device tests and debug runs: nothing is asked, no location service is touched. */
    data class StandIn(val access: Access = Access.ALLOWED, val fix: Fix? = null, val locating: Boolean = false)

    private val debugFix: Fix? get() = standIn?.fix

    private val context = context.applicationContext
    private val preferences = this.context.getSharedPreferences("capture", Context.MODE_PRIVATE)
    private val main = Handler(Looper.getMainLooper())

    var access: Access by mutableStateOf(currentAccess())
        private set
    var fix: Fix? by mutableStateOf(standIn?.fix)
        private set
    /** A fix has been asked for and has not arrived yet. */
    var locating: Boolean by mutableStateOf(standIn?.locating ?: false)
        private set

    private fun granted() = ContextCompat.checkSelfPermission(context, Manifest.permission.ACCESS_COARSE_LOCATION) == PackageManager.PERMISSION_GRANTED

    private fun currentAccess(): Access = when {
        standIn != null -> standIn.access
        granted() -> Access.ALLOWED
        // The system does not say whether it has asked before; the app remembers a refusal itself.
        preferences.getBoolean(KEY_REFUSED, false) -> Access.REFUSED
        else -> Access.NOT_ASKED
    }

    /** The user answered the system's question. */
    fun onPermissionResult(allowed: Boolean) {
        if (standIn != null) return
        preferences.edit { putBoolean(KEY_REFUSED, !allowed) }
        access = currentAccess()
        if (allowed) locateIfAllowed()
    }

    /** Takes one fix when the user has already said yes. Never shows the system's question. */
    @SuppressLint("MissingPermission")   // Checked by `granted()`.
    fun locateIfAllowed() {
        access = currentAccess()
        if (standIn != null || access != Access.ALLOWED || fix != null || locating || !granted()) return
        val manager = context.getSystemService(Context.LOCATION_SERVICE) as? LocationManager ?: return
        val enabled = runCatching { manager.getProviders(true) }.getOrDefault(emptyList())
        // The precise-only provider is of no use to approximate permission before Android 12.
        val wanted = listOfNotNull(
            "fused".takeIf { Build.VERSION.SDK_INT >= 31 }, LocationManager.NETWORK_PROVIDER,
            LocationManager.GPS_PROVIDER.takeIf { Build.VERSION.SDK_INT >= 31 }, LocationManager.PASSIVE_PROVIDER,
        )
        val provider = wanted.firstOrNull { it in enabled } ?: return   // No position: the screen goes on with the last gym and the full directory.
        locating = true
        var done = false
        fun finish(location: Location?) {
            if (done) return
            done = true
            locating = false
            val best = location ?: runCatching { manager.getLastKnownLocation(provider) }.getOrNull()
            if (best != null) fix = Fix(best.latitude, best.longitude, if (best.hasAccuracy()) best.accuracy.toDouble() else Double.MAX_VALUE)
        }
        try {
            if (Build.VERSION.SDK_INT >= 30) {
                manager.getCurrentLocation(provider, null, ContextCompat.getMainExecutor(context)) { location -> finish(location) }
            } else {
                val listener = object : LocationListener {
                    override fun onLocationChanged(location: Location) { manager.removeUpdates(this); finish(location) }
                    @Deprecated("Deprecated in Java") override fun onStatusChanged(provider: String?, status: Int, extras: android.os.Bundle?) = Unit
                    override fun onProviderEnabled(provider: String) = Unit
                    override fun onProviderDisabled(provider: String) = Unit
                }
                @Suppress("DEPRECATION")
                manager.requestSingleUpdate(provider, listener, Looper.getMainLooper())
                main.postDelayed({ runCatching { manager.removeUpdates(listener) }; finish(null) }, TIMEOUT_MILLIS)
            }
        } catch (error: Exception) {
            finish(null)
        }
    }

    /** Asks again after a fix could not be had. */
    fun retry() = locateIfAllowed()

    /** Empty without a position. */
    fun suggestions(directory: List<DirectoryGym>): GymSuggestions {
        val fix = fix ?: return GymSuggestions()
        return GymDistance.suggestions(directory, fix.latitude, fix.longitude, fix.accuracy)
    }

    private companion object {
        const val KEY_REFUSED = "locationRefused"
        const val TIMEOUT_MILLIS = 20_000L
    }
}
