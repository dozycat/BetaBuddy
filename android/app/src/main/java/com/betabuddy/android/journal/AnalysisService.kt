package com.betabuddy.android.journal

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import com.betabuddy.android.MainActivity
import com.betabuddy.android.R
import com.betabuddy.android.pose.AnalysisException
import com.betabuddy.android.pose.AnalysisQuality
import com.betabuddy.android.pose.VideoAnalyzer
import java.util.Locale
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/** User-requested, local media processing. The saved archive is the durable work queue. */
class AnalysisService : Service() {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
    private var queue: Job? = null
    private var active: Job? = null
    private var activeId: String? = null
    private var lastPercent = -1

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        val manager = getSystemService(NotificationManager::class.java)
        manager.createNotificationChannel(NotificationChannel(CHANNEL, getString(R.string.journal_analyze_skeleton), NotificationManager.IMPORTANCE_LOW))
        // Pose recognition gives way to the camera: the running one is stopped, its request is kept.
        scope.launch { CameraGate.open.collect { open -> if (open) active?.cancel() } }
        val notification = notification()
        if (Build.VERSION.SDK_INT >= 35) {
            startForeground(NOTIFICATION, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROCESSING)
        } else if (Build.VERSION.SDK_INT >= 29) {
            startForeground(NOTIFICATION, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC)
        } else {
            startForeground(NOTIFICATION, notification)
        }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == CANCEL) {
            val id = intent.getStringExtra(CLIMB_ID) ?: activeId
            scope.launch {
                if (id != null) {
                    try {
                        withContext(Dispatchers.IO) { JournalGraph.store(this@AnalysisService).requestAnalysis(id, null) }
                        if (id == activeId) active?.cancel()
                        mutableState.value = ReanalysisUiState(climbId = id, message = UiMessage(R.string.journal_reanalyze_stopped))
                    } catch (error: Exception) {
                        mutableState.value = ReanalysisUiState(climbId = id, message = UiMessage(R.string.journal_import_failed, error.localizedMessage))
                    }
                }
                startQueue()
            }
        } else startQueue()
        // A killed process leaves requests in the archive; the next foreground resumes them.
        return START_NOT_STICKY
    }

    private fun startQueue() {
        if (queue?.isActive == true) return
        queue = scope.launch {
            val store = withContext(Dispatchers.IO) { JournalGraph.store(this@AnalysisService) }
            try {
                while (true) {
                    // The in-app camera is in use: requests stay in the archive and start again when it lets go.
                    if (CameraGate.open.value) {
                        mutableState.value = ReanalysisUiState()
                        break
                    }
                    val climb = store.climbs.firstOrNull { it.pendingAnalysis != null } ?: break
                    activeId = climb.id
                    mutableState.value = ReanalysisUiState(climbId = climb.id, busy = true)
                    lastPercent = -1
                    updateNotification(0.0)
                    active = launch {
                        try {
                            val quality = AnalysisQuality.entries.firstOrNull { it.name.lowercase(Locale.ROOT) == climb.pendingAnalysis } ?: AnalysisQuality.BALANCED
                            val result = VideoAnalyzer(this@AnalysisService).analyze(store.file(climb.videoName), quality, cameraClip = climb.capture != null) { progress ->
                                scope.launch {
                                    if (activeId == climb.id && mutableState.value.busy) {
                                        mutableState.value = ReanalysisUiState(climbId = climb.id, busy = true, progress = progress)
                                        updateNotification(progress)
                                    }
                                }
                            }
                            ensureActive()
                            withContext(Dispatchers.IO) { store.replaceAnalysis(climb, result) }
                            mutableState.value = ReanalysisUiState(climbId = climb.id, message = UiMessage(R.string.journal_reanalyze_done))
                        } catch (error: CancellationException) {
                            throw error // stop/timeout never destroys the saved video or prior analysis
                        } catch (error: Exception) {
                            try { withContext(Dispatchers.IO) { store.requestAnalysis(climb.id, null) } }
                            catch (_: Exception) { queue?.cancel() }
                            mutableState.value = ReanalysisUiState(climbId = climb.id, message =
                                if (error is AnalysisException) error.error.message()
                                else UiMessage(R.string.journal_import_failed, error.localizedMessage ?: error.javaClass.simpleName))
                        }
                    }
                    active?.join()
                    active = null
                    activeId = null
                }
            } catch (error: Exception) {
                if (error !is CancellationException) {
                    mutableState.value = ReanalysisUiState(climbId = activeId, message = UiMessage(R.string.journal_import_failed, error.localizedMessage))
                }
            } finally {
                stopForeground(STOP_FOREGROUND_REMOVE)
                stopSelf()
            }
        }
    }

    private fun notification(progress: Double? = null): Notification {
        val open = PendingIntent.getActivity(this, 0, Intent(this, MainActivity::class.java), PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE)
        val stop = PendingIntent.getService(this, 1, Intent(this, AnalysisService::class.java).setAction(CANCEL).putExtra(CLIMB_ID, activeId), PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE)
        return Notification.Builder(this, CHANNEL)
            .setSmallIcon(R.drawable.ic_launcher_monochrome)
            .setContentTitle(getString(R.string.journal_analyze_skeleton))
            .setContentText(getString(R.string.journal_analysis_background))
            .setContentIntent(open).setOngoing(true).setOnlyAlertOnce(true)
            .setProgress(100, ((progress ?: 0.0) * 100).toInt(), progress == null)
            .addAction(Notification.Action.Builder(null, getString(R.string.journal_reanalyze_stop), stop).build())
            .build()
    }

    private fun updateNotification(progress: Double) {
        val percent = (progress * 100).toInt()
        if (percent == lastPercent) return
        lastPercent = percent
        if (Build.VERSION.SDK_INT >= 33 && checkSelfPermission(android.Manifest.permission.POST_NOTIFICATIONS) != android.content.pm.PackageManager.PERMISSION_GRANTED) return
        getSystemService(NotificationManager::class.java).notify(NOTIFICATION, notification(progress))
    }

    override fun onTimeout(startId: Int, fgsType: Int) {
        scope.cancel()
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    override fun onDestroy() {
        scope.cancel()
        mutableState.value = mutableState.value.copy(busy = false)
        super.onDestroy()
    }

    companion object {
        private const val CHANNEL = "local-analysis"
        private const val NOTIFICATION = 41
        private const val CANCEL = "com.betabuddy.android.CANCEL_ANALYSIS"
        private const val CLIMB_ID = "climbId"
        private val mutableState = MutableStateFlow(ReanalysisUiState())
        internal val state = mutableState.asStateFlow()

        internal fun reportFailure(id: String, error: Exception) {
            mutableState.value = ReanalysisUiState(climbId = id, message = UiMessage(R.string.journal_import_failed, error.localizedMessage))
        }

        internal fun start(context: Context) {
            if (CameraGate.open.value) return   // Picked up when the camera closes.
            try {
                context.startForegroundService(Intent(context, AnalysisService::class.java))
            } catch (_: RuntimeException) {
                // Starting from background may be refused. The durable request will run on return.
                mutableState.value = ReanalysisUiState(message = UiMessage(R.string.journal_analysis_deferred))
            }
        }

        internal fun cancel(context: Context, id: String) {
            context.startService(Intent(context, AnalysisService::class.java).setAction(CANCEL).putExtra(CLIMB_ID, id))
        }
    }
}
