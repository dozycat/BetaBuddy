package com.betabuddy.android.capture

import android.Manifest
import android.content.Intent
import android.graphics.BitmapFactory
import android.net.Uri
import android.provider.Settings
import android.view.ViewGroup
import android.view.WindowManager
import android.widget.FrameLayout
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.defaultMinSize
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.SideEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.key
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.produceState
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.layout.layout
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.LiveRegionMode
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.clearAndSetSemantics
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.liveRegion
import androidx.compose.ui.semantics.selected
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.semantics.stateDescription
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.Constraints
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import androidx.compose.ui.window.DialogWindowProvider
import androidx.core.view.WindowCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.betabuddy.android.BuildConfig
import com.betabuddy.android.R
import com.betabuddy.android.data.Climb
import com.betabuddy.android.data.clockText
import com.betabuddy.android.gym.GymPack
import com.betabuddy.android.i18n.AppLanguage
import com.betabuddy.android.journal.FilledButton
import com.betabuddy.android.journal.JournalViewModel
import com.betabuddy.android.journal.NearbyGym
import com.betabuddy.android.journal.OutlineButton
import com.betabuddy.android.journal.ProgressBar
import com.betabuddy.android.journal.SwitchRow
import com.betabuddy.android.journal.directoryGyms
import com.betabuddy.android.journal.gradeText
import com.betabuddy.android.journal.gradeValues
import com.betabuddy.android.journal.text
import com.betabuddy.android.review.AnalysisScreen
import com.betabuddy.android.review.AnalysisSession
import com.betabuddy.android.review.rememberMainScope
import com.betabuddy.android.ui.common.BBIcons
import com.betabuddy.android.ui.theme.BB
import java.io.File
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

private val Chip = Color.White.copy(alpha = 0.14f)
private val Warning = Color(0xFFFF9500)
private val RecordRed = Color(0xFFE5332A)

/**
 * The in-app camera, full screen over the logbook: frame the wall, pick the colour, mark start and
 * finish, record, and come back to a clip that is saved before anything is analysed.
 * Counterpart of ios/BetaBuddy/Views/CaptureView.swift.
 *
 * [onImport] leaves the camera for the photo picker import.
 */
@Composable
internal fun CaptureScreen(model: JournalViewModel, pack: GymPack?, onClose: () -> Unit, onImport: () -> Unit) {
    val context = LocalContext.current
    val owner = LocalLifecycleOwner.current
    val language = AppLanguage.current()
    val debug = BuildConfig.DEBUG
    val usedPack = if (debug && CaptureDebug.noPack) null else (if (debug) CaptureDebug.pack else null) ?: pack
    val packFiles = if (debug) CaptureDebug.packFiles else null
    // In the language shown when the clip is saved, not the one the screen was opened in.
    val defaultTitle = rememberUpdatedState(stringResource(R.string.journal_default_title))
    // Made once: the screen is recomposed whenever the logbook behind it changes, and a camera must not be set up each time.
    val session = remember {
        val video = if (debug) CaptureDebug.video else null
        val source = if (video != null) FileCaptureSource(video, CaptureDebug.fieldOfView) else CameraSource(context)
        if (debug) CaptureDebug.lastSource = source
        CaptureSession(context, usedPack, source, language, defaultTitle = { defaultTitle.value }, packFiles = packFiles)
            .also { if (debug) { it.motion.fixed = CaptureDebug.attitude; CaptureDebug.configure?.invoke(it) } }
    }
    val locator = remember { GymLocator(context, if (debug) CaptureDebug.location else null) }
    val bundled = directoryGyms()
    val directory = (if (debug) CaptureDebug.directory else null) ?: bundled
    val climbs by model.climbs.collectAsState()

    var choosingGym by remember { mutableStateOf(false) }
    var confirming by remember { mutableStateOf(false) }
    // What a tap on the picture marks. Starts on the start holds: those are what the page gets wrong most.
    var tool by remember { mutableStateOf(CaptureRole.START_HAND) }
    var viewing by remember { mutableStateOf<String?>(null) }
    // The gym the user declined when it was offered as the one they are standing in.
    var declined by remember { mutableStateOf<String?>(null) }
    var askedCamera by rememberSaveable { mutableStateOf(false) }

    val cameraPermission = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted -> if (granted) session.retry(owner) }
    LaunchedEffect(session) {
        session.start(owner)
        locator.locateIfAllowed()
    }
    // The system's question is asked once per visit; after a refusal the screen explains and offers the import.
    LaunchedEffect(session.camera) {
        if (session.camera == CaptureSession.Camera.DENIED && !askedCamera) {
            askedCamera = true
            cameraPermission.launch(Manifest.permission.CAMERA)
        }
    }
    DisposableEffect(owner, session) {
        val observer = LifecycleEventObserver { _, event ->
            when (event) {
                // Leaving the foreground closes a running recording with what it has.
                Lifecycle.Event.ON_STOP -> session.onBackground()
                // Back from the system's settings, perhaps with the permission.
                Lifecycle.Event.ON_RESUME -> if (askedCamera) session.retry(owner)
                else -> Unit
            }
        }
        owner.lifecycle.addObserver(observer)
        onDispose {
            owner.lifecycle.removeObserver(observer)
            session.close()
        }
    }

    val phase = session.phase
    val busy = phase == CaptureSession.Phase.Recording || phase == CaptureSession.Phase.Saving
    val framing = phase == CaptureSession.Phase.Framing && session.camera == CaptureSession.Camera.RUNNING
    val saved = (phase as? CaptureSession.Phase.Saved)?.let { state -> climbs?.firstOrNull { it.id == state.id } }
    // A gym right where the phone is that is not the one selected. It is offered; the selection never changes by itself.
    val offered = if (framing) locator.suggestions(directory).here?.takeIf { it.gym.id != session.gymId && it.gym.id != declined } else null

    Dialog(
        onDismissRequest = { if (!busy) onClose() },
        properties = DialogProperties(usePlatformDefaultWidth = false, decorFitsSystemWindows = false, dismissOnClickOutside = false),
    ) {
        val view = LocalView.current
        // A window of its own: dark under both bars, light icons, and the screen stays on while the phone stands there.
        DisposableEffect(view) {
            val window = (view.parent as? DialogWindowProvider)?.window
            window?.setDimAmount(0f)
            window?.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
            window?.let { WindowCompat.getInsetsController(it, view).apply { isAppearanceLightStatusBars = false; isAppearanceLightNavigationBars = false } }
            onDispose { window?.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON) }
        }
        Box(Modifier.fillMaxSize().testTag("captureScreen")) {
            // The capture page has nothing to show; it only has to stay alive and keep answering. It is
            // kept in the window at full size, underneath everything, like a page nobody happens to see.
            session.webView?.let { webView ->
                key(webView) {
                    AndroidView(
                        factory = { viewContext ->
                            FrameLayout(viewContext).apply {
                                (webView.parent as? ViewGroup)?.removeView(webView)
                                addView(webView, FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT))
                                importantForAccessibility = android.view.View.IMPORTANT_FOR_ACCESSIBILITY_NO_HIDE_DESCENDANTS
                            }
                        },
                        onRelease = { container -> container.removeAllViews() },
                        modifier = Modifier.fillMaxSize().clearAndSetSemantics { },
                    )
                }
            }
            Column(
                Modifier.fillMaxSize().background(BB.Ink).pointerInput(Unit) { detectTapGestures { } }
                    .windowInsetsPadding(WindowInsets.safeDrawing).padding(horizontal = 12.dp, vertical = 8.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                TopBar(session, busy, onClose = onClose, onChooseGym = { choosingGym = true })
                offered?.let { here ->
                    Offer(here, onUse = { session.selectGym(here.gym.id, here.gym.name) }, onDecline = { declined = here.gym.id })
                }
                Picture(
                    session, framing, tool, saved, model, Modifier.weight(1f).fillMaxWidth(),
                    onImport = onImport, onClose = onClose, onViewAnalysis = { viewing = it },
                )
                // Below the picture, not on it: the start of a problem is at the bottom of the wall.
                if (framing) Colours(session, tool) { tool = it }
                BottomBar(session, framing) { if (session.needsConfirmation) confirming = true else session.startRecording() }
            }
        }
    }

    if (choosingGym) {
        CaptureGymSheet(
            session.gymId, session.gymName, locator, directory,
            onSelect = { id, name -> session.selectGym(id, name); choosingGym = false }, onClose = { choosingGym = false },
        )
    }
    if (confirming) {
        AlertDialog(
            onDismissRequest = { confirming = false }, containerColor = BB.Paper,
            title = { Text(stringResource(R.string.capture_confirm_title), style = BB.Title3, color = BB.Ink) },
            text = { session.assessment?.hint?.let { Text(captureLine(it), style = BB.Body, color = BB.Ink) } },
            confirmButton = {
                TextButton(onClick = { confirming = false; session.startRecording() }, modifier = Modifier.testTag("recordAnyway")) { Text(stringResource(R.string.capture_confirm_record)) }
            },
            dismissButton = { TextButton(onClick = { confirming = false }) { Text(stringResource(R.string.capture_confirm_adjust)) } },
        )
    }
    session.notice?.let { notice ->
        AlertDialog(
            onDismissRequest = { session.notice = null }, containerColor = BB.Paper,
            text = { Text(stringResource(CaptureText.notice(notice)), style = BB.Body, color = BB.Ink, modifier = Modifier.testTag("captureNotice")) },
            confirmButton = { TextButton(onClick = { session.notice = null }) { Text(stringResource(R.string.capture_notice_ok)) } },
        )
    }
    val viewed = viewing?.let { id -> climbs?.firstOrNull { it.id == id } }
    val analysisUrl = usedPack?.analysisUrl(language)
    if (viewed != null && usedPack != null && analysisUrl != null) {
        val scope = rememberMainScope()
        val analysis = remember(viewed.id, analysisUrl) {
            AnalysisSession(
                context = context, pack = usedPack, url = analysisUrl, climbId = viewed.id, video = model.videoFile(viewed), model = model, scope = scope,
                poses = { viewed.analysis.frames }, onSeek = { viewing = null }, packFiles = packFiles,
            )
        }
        AnalysisScreen(analysis, onDone = { viewing = null })
    }
}

// Top: close, gym, grade

@Composable
private fun TopBar(session: CaptureSession, busy: Boolean, onClose: () -> Unit, onChooseGym: () -> Unit) {
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
        val close = stringResource(R.string.capture_close)
        Box(
            Modifier.size(48.dp).alpha(if (busy) 0.3f else 1f).clip(CircleShape).background(Chip)
                .clickable(enabled = !busy, role = Role.Button, onClickLabel = close, onClick = onClose).testTag("captureClose"),
            contentAlignment = Alignment.Center,
        ) { Icon(BBIcons.Close, contentDescription = close, tint = BB.Paper, modifier = Modifier.size(18.dp)) }
        if (session.phase == CaptureSession.Phase.Recording) {
            Spacer(Modifier.weight(1f))
            Row(
                Modifier.defaultMinSize(minHeight = 48.dp).clip(CircleShape).background(Chip).padding(horizontal = 14.dp),
                horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically,
            ) {
                Box(Modifier.size(10.dp).clip(CircleShape).background(RecordRed))
                Text(stringResource(R.string.capture_recording), style = BB.Subheadline, fontWeight = FontWeight.Bold, color = BB.Paper)
                Text(clockText(session.recordedSeconds), style = BB.mono(15.sp, FontWeight.Bold, 0.sp), color = BB.Paper, modifier = Modifier.testTag("captureElapsed"))
            }
            Spacer(Modifier.weight(1f))
            Spacer(Modifier.width(48.dp))
        } else {
            val enabled = session.phase == CaptureSession.Phase.Framing
            Row(
                Modifier.weight(1f).defaultMinSize(minHeight = 48.dp).alpha(if (enabled) 1f else 0.5f).clip(CircleShape).background(Chip)
                    .clickable(enabled = enabled, role = Role.Button, onClick = onChooseGym).padding(horizontal = 12.dp).testTag("captureGym"),
                horizontalArrangement = Arrangement.spacedBy(6.dp), verticalAlignment = Alignment.CenterVertically,
            ) {
                Icon(BBIcons.Pin, contentDescription = null, tint = BB.Paper, modifier = Modifier.size(18.dp))
                Text(
                    session.gymName.ifEmpty { stringResource(R.string.album_choose_gym) }, style = BB.Subheadline, fontWeight = FontWeight.Bold, color = BB.Paper,
                    maxLines = 1, overflow = TextOverflow.Ellipsis, modifier = Modifier.weight(1f, fill = false),
                )
                Icon(BBIcons.ChevronDown, contentDescription = null, tint = BB.Paper, modifier = Modifier.size(12.dp))
            }
            var open by remember { mutableStateOf(false) }
            val label = stringResource(R.string.journal_import_grade_label)
            val grade = gradeText(session.grade)
            Box {
                Row(
                    Modifier.defaultMinSize(minHeight = 48.dp).alpha(if (enabled) 1f else 0.5f).clip(CircleShape).background(Chip)
                        .clickable(enabled = enabled, role = Role.DropdownList, onClickLabel = label) { open = true }.padding(horizontal = 12.dp)
                        .semantics { contentDescription = "$label, $grade" }.testTag("captureGrade"),
                    horizontalArrangement = Arrangement.spacedBy(6.dp), verticalAlignment = Alignment.CenterVertically,
                ) {
                    // A grade the user did not pick is marked as such.
                    Text(grade, style = BB.Subheadline, fontWeight = FontWeight.Bold, color = if (session.gradePrefilled) BB.Lime else BB.Paper, maxLines = 1)
                    Icon(BBIcons.ChevronDown, contentDescription = null, tint = BB.Paper, modifier = Modifier.size(12.dp))
                }
                DropdownMenu(expanded = open, onDismissRequest = { open = false }, containerColor = BB.Paper) {
                    for (value in gradeValues) {
                        DropdownMenuItem(text = { Text(gradeText(value), color = BB.Ink) }, onClick = { session.pickGrade(value); open = false }, modifier = Modifier.testTag("captureGrade-$value"))
                    }
                }
            }
        }
    }
}

@Composable
private fun Offer(here: NearbyGym, onUse: () -> Unit, onDecline: () -> Unit) {
    Row(
        Modifier.fillMaxWidth().clip(RoundedCornerShape(14.dp)).background(Chip).padding(start = 12.dp, end = 4.dp, top = 4.dp, bottom = 4.dp),
        horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically,
    ) {
        Icon(BBIcons.Pin, contentDescription = null, tint = BB.Paper, modifier = Modifier.size(14.dp))
        Text(
            stringResource(R.string.capture_offer, here.gym.name, captureDistance(here).orEmpty()), style = BB.Caption, fontWeight = FontWeight.Bold, color = BB.Paper,
            modifier = Modifier.weight(1f).testTag("captureOffer"),
        )
        Box(
            Modifier.defaultMinSize(minHeight = 48.dp).clip(CircleShape).clickable(role = Role.Button, onClick = onUse).testTag("useNearestGym"),
            contentAlignment = Alignment.Center,
        ) {
            Text(
                stringResource(R.string.capture_offer_use), style = BB.Caption, fontWeight = FontWeight.Bold, color = BB.Ink,
                modifier = Modifier.clip(CircleShape).background(BB.Lime).padding(horizontal = 12.dp, vertical = 10.dp),
            )
        }
        val decline = stringResource(R.string.capture_offer_decline)
        Box(Modifier.size(48.dp).clip(CircleShape).clickable(role = Role.Button, onClickLabel = decline, onClick = onDecline).testTag("declineNearestGym"), contentAlignment = Alignment.Center) {
            Icon(BBIcons.Close, contentDescription = decline, tint = BB.Paper, modifier = Modifier.size(14.dp))
        }
    }
}

// Middle: the picture and what is drawn on it

@Composable
private fun Picture(
    session: CaptureSession, framing: Boolean, tool: CaptureRole, saved: Climb?, model: JournalViewModel, modifier: Modifier,
    onImport: () -> Unit, onClose: () -> Unit, onViewAnalysis: (String) -> Unit,
) {
    val context = LocalContext.current
    BoxWithConstraints(modifier, contentAlignment = Alignment.Center) {
        val (frameWidth, frameHeight) = session.frameSize
        val scale = minOf(maxWidth.value / frameWidth, maxHeight.value / frameHeight)
        val width = (frameWidth * scale).dp
        val height = (frameHeight * scale).dp
        Box(Modifier.size(width, height).clip(RoundedCornerShape(18.dp)).background(Color.Black), contentAlignment = Alignment.Center) {
            when (session.camera) {
                CaptureSession.Camera.DENIED -> Unavailable(
                    R.string.capture_denied_title, R.string.capture_denied_detail,
                    onSettings = {
                        runCatching { context.startActivity(Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS, Uri.fromParts("package", context.packageName, null))) }
                    },
                    onImport = onImport,
                )
                CaptureSession.Camera.UNAVAILABLE -> Unavailable(R.string.capture_unavailable_title, R.string.capture_unavailable_detail, onSettings = null, onImport = onImport)
                CaptureSession.Camera.STARTING, CaptureSession.Camera.RUNNING -> {
                    AndroidView(factory = { session.source.createPreview(it) }, modifier = Modifier.fillMaxSize().clearAndSetSemantics { })
                    val phase = session.phase
                    val holds = session.assessment?.holds
                    if ((phase == CaptureSession.Phase.Framing || phase == CaptureSession.Phase.Recording) && session.color != null && holds != null) {
                        Overlay(holds, width, height, Modifier.alpha(if (phase == CaptureSession.Phase.Recording) 0.45f else 1f))
                    }
                    when {
                        framing -> {
                            // One tap marks the hold there as whatever is selected below the picture.
                            val marks = stringResource(CaptureText.role(tool))
                            Box(
                                Modifier.fillMaxSize().testTag("capturePicture").semantics { contentDescription = marks }
                                    .pointerInput(tool) {
                                        detectTapGestures { offset -> session.correct((offset.x / size.width).toDouble(), (offset.y / size.height).toDouble(), tool) }
                                    },
                            )
                            Box(Modifier.fillMaxSize().padding(10.dp), contentAlignment = Alignment.TopStart) { Guidance(session) }
                        }
                        phase == CaptureSession.Phase.Recording -> Box(Modifier.fillMaxSize().padding(10.dp), contentAlignment = Alignment.BottomCenter) {
                            Text(
                                stringResource(R.string.capture_recording_note), style = BB.Subheadline, fontWeight = FontWeight.Bold, color = BB.Ink,
                                modifier = Modifier.clip(RoundedCornerShape(14.dp)).background(BB.Paper).padding(horizontal = 14.dp, vertical = 10.dp),
                            )
                        }
                        phase == CaptureSession.Phase.Saving -> Column(
                            Modifier.clip(RoundedCornerShape(22.dp)).background(BB.Paper).padding(24.dp),
                            horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(14.dp),
                        ) {
                            CircularProgressIndicator(color = BB.Ink, trackColor = BB.Line, strokeWidth = 3.dp, modifier = Modifier.size(32.dp))
                            Text(stringResource(R.string.capture_saving), style = BB.Headline, color = BB.Ink, modifier = Modifier.testTag("captureSaving"))
                        }
                        phase is CaptureSession.Phase.SaveFailed -> SaveFailed(phase.message, onRetry = session::save, onDiscard = session::discardHeldClip)
                        saved != null -> SavedCard(saved, session, model, onAgain = session::frameAgain, onDone = onClose, onViewAnalysis = { onViewAnalysis(saved.id) })
                    }
                }
            }
        }
    }
}

/**
 * What the capture page recognised, drawn over the picture: the chosen colour's holds, with the
 * start and finish marked. Dashed outlines are the page's own guesses; solid ones were set by the user.
 */
@Composable
private fun Overlay(holds: List<CaptureAssessment.Hold>, width: Dp, height: Dp, modifier: Modifier = Modifier) {
    fun colour(role: CaptureRole) = when (role) {
        CaptureRole.START_HAND, CaptureRole.START_FOOT -> BB.Lime
        CaptureRole.FINISH -> BB.Lavender
        else -> Color.White
    }
    Box(modifier.size(width, height).clearAndSetSemantics { }) {
        Canvas(Modifier.fillMaxSize()) {
            val dash = PathEffect.dashPathEffect(floatArrayOf(5.dp.toPx(), 4.dp.toPx()))
            for (hold in holds) {
                val marked = hold.role != CaptureRole.HOLD
                // Thin for plain holds: a board can have a hundred of them, and the holds must stay visible.
                val under = Stroke(width = (if (marked) 5.5f else 3f).dp.toPx(), join = StrokeJoin.Round)
                val over = Stroke(width = (if (marked) 3f else 1.5f).dp.toPx(), join = StrokeJoin.Round, pathEffect = if (hold.user) null else dash)
                if (hold.poly.size >= 3) {
                    val path = Path().apply {
                        hold.poly.forEachIndexed { index, point ->
                            val x = (point.x * size.width).toFloat(); val y = (point.y * size.height).toFloat()
                            if (index == 0) moveTo(x, y) else lineTo(x, y)
                        }
                        close()
                    }
                    drawPath(path, Color.Black.copy(alpha = 0.65f), style = under)
                    drawPath(path, colour(hold.role), style = over)
                } else {
                    val grow = 2.dp.toPx()
                    val origin = Offset((hold.box.x * size.width).toFloat() - grow, (hold.box.y * size.height).toFloat() - grow)
                    val extent = Size((hold.box.width * size.width).toFloat() + 2 * grow, (hold.box.height * size.height).toFloat() + 2 * grow)
                    val corner = CornerRadius(6.dp.toPx())
                    drawRoundRect(Color.Black.copy(alpha = 0.65f), origin, extent, corner, style = under)
                    drawRoundRect(colour(hold.role), origin, extent, corner, style = over)
                }
            }
        }
        // Start and finish also carry a word: colour alone does not say which is which.
        val density = LocalDensity.current
        for (hold in holds) {
            if (hold.role == CaptureRole.HOLD) continue
            val name = stringResource(CaptureText.role(hold.role))
            // A guess says so; what the user marked carries a tick.
            val text = if (hold.user) name else stringResource(R.string.capture_role_guess, name)
            val centre = with(density) { (hold.box.midX * width.toPx()).toFloat().coerceIn(40.dp.toPx(), maxOf(40.dp.toPx(), width.toPx() - 40.dp.toPx())) }
            val middle = with(density) { maxOf((hold.box.y * height.toPx()).toFloat() - 13.dp.toPx(), 10.dp.toPx()) }
            Row(
                Modifier.layout { measurable, _ ->
                    val placeable = measurable.measure(Constraints())
                    layout(0, 0) { placeable.place((centre - placeable.width / 2f).toInt(), (middle - placeable.height / 2f).toInt()) }
                }.clip(CircleShape).background(colour(hold.role).copy(alpha = if (hold.user) 1f else 0.8f)).padding(horizontal = 6.dp, vertical = 3.dp),
                horizontalArrangement = Arrangement.spacedBy(3.dp), verticalAlignment = Alignment.CenterVertically,
            ) {
                Text(text, style = BB.Caption.copy(fontSize = 10.sp, lineHeight = 12.sp), fontWeight = FontWeight.Bold, color = BB.Ink, maxLines = 1)
                if (hold.user) Icon(BBIcons.Check, contentDescription = null, tint = BB.Ink, modifier = Modifier.size(9.dp))
            }
        }
    }
}

private enum class Tone { PLAIN, BLOCKED, READY }

/** One line: what to change, or that the position is good. Plus an indicator for the score. */
@Composable
private fun Guidance(session: CaptureSession) {
    val assessment = session.assessment
    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        when (session.page) {
            CaptureSession.Page.UNAVAILABLE -> Pill(BBIcons.Video, stringResource(R.string.capture_no_checks), Tone.PLAIN, "captureNoChecks")
            CaptureSession.Page.FAILED -> Pill(BBIcons.Alert, stringResource(R.string.capture_checks_failed), Tone.PLAIN, "captureNoChecks")
            CaptureSession.Page.LOADING -> Pill(null, stringResource(R.string.capture_looking), Tone.PLAIN, "captureLooking")
            CaptureSession.Page.READY -> {
                val hint = assessment?.hint
                when {
                    assessment == null -> Pill(null, stringResource(R.string.capture_looking), Tone.PLAIN, "captureLooking")
                    // A remark that does not block is shown under the picture, see `Colours`.
                    assessment.ready -> Pill(BBIcons.Check, stringResource(R.string.capture_ready), Tone.READY, "captureReady")
                    hint != null -> Pill(if (hint.blocking) BBIcons.Alert else null, captureLine(hint), if (hint.blocking) Tone.BLOCKED else Tone.PLAIN, "captureHint", hint.code)
                    else -> Pill(null, stringResource(R.string.capture_settling), Tone.PLAIN, "captureSettling")
                }
                assessment?.score?.let { Quality(it) }
            }
        }
    }
}

@Composable
private fun Pill(icon: ImageVector?, text: String, tone: Tone, tag: String, state: String? = null) {
    val shape = RoundedCornerShape(14.dp)
    Row(
        Modifier.fillMaxWidth().clip(shape).background(if (tone == Tone.READY) BB.Lime else BB.Paper)
            .then(if (tone == Tone.BLOCKED) Modifier.border(2.5.dp, Warning, shape) else Modifier)
            .padding(horizontal = 14.dp, vertical = 10.dp)
            .semantics(mergeDescendants = true) { liveRegion = LiveRegionMode.Polite; if (state != null) stateDescription = state }.testTag(tag),
        horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically,
    ) {
        if (icon != null) Icon(icon, contentDescription = null, tint = BB.Ink, modifier = Modifier.size(18.dp))
        Text(text, style = BB.Subheadline, fontWeight = FontWeight.Bold, color = BB.Ink)
    }
}

/** Five segments, never a number: the score only orders camera positions. */
@Composable
private fun Quality(score: Double) {
    val level = CaptureText.level(score)
    val label = stringResource(R.string.capture_quality)
    val value = stringResource(if (level >= 4) R.string.capture_quality_good else if (level >= 3) R.string.capture_quality_fair else R.string.capture_quality_poor)
    Row(
        Modifier.clip(CircleShape).background(Color.Black.copy(alpha = 0.55f)).padding(horizontal = 10.dp, vertical = 6.dp)
            .testTag("captureQuality").clearAndSetSemantics { contentDescription = label; stateDescription = value },
        horizontalArrangement = Arrangement.spacedBy(6.dp), verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(label, style = BB.Caption.copy(fontSize = 11.sp), fontWeight = FontWeight.Bold, color = BB.Paper)
        Row(horizontalArrangement = Arrangement.spacedBy(3.dp)) {
            for (index in 1..5) Box(Modifier.size(width = 16.dp, height = 6.dp).clip(CircleShape).background(if (index <= level) BB.Lime else Color.White.copy(alpha = 0.35f)))
        }
    }
}

/**
 * Under the picture: first the colours the page sees on the wall; once one is chosen, what a tap
 * on the picture marks. The start comes first because the page's own guess at it is the weakest.
 */
@Composable
private fun Colours(session: CaptureSession, tool: CaptureRole, onTool: (CaptureRole) -> Unit) {
    if (session.page != CaptureSession.Page.READY || (session.palette.isEmpty() && session.color == null)) return
    val assessment = session.assessment
    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        // Nothing blocks, but the page still has something to say. What it says about the start
        // and the finish is already the business of the line below.
        val hint = assessment?.hint
        if (assessment != null && assessment.ready && hint != null && hint.code != "start_unclear" && hint.code != "finish_unclear") {
            Text(
                captureLine(hint), style = BB.Caption, fontWeight = FontWeight.Bold, color = BB.Lime,
                modifier = Modifier.semantics { stateDescription = hint.code }.testTag("captureHint"),
            )
        }
        val route = session.recognised
        if (session.color != null && route != null) {
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
                val detail = listOfNotNull(route.title, route.grade).filter { it.isNotBlank() }.joinToString(" · ")
                Text(
                    stringResource(R.string.capture_route, detail), style = BB.Caption, fontWeight = FontWeight.Bold, color = BB.Paper,
                    modifier = Modifier.weight(1f, fill = false).testTag("captureRoute"),
                )
                Box(Modifier.defaultMinSize(minHeight = 48.dp).clip(CircleShape).clickable(role = Role.Button, onClick = session::rejectRoute).testTag("captureRejectRoute"), contentAlignment = Alignment.Center) {
                    Text(
                        stringResource(R.string.capture_route_reject), style = BB.Caption, fontWeight = FontWeight.Bold, color = BB.Paper,
                        modifier = Modifier.clip(CircleShape).background(Chip).padding(horizontal = 10.dp, vertical = 8.dp),
                    )
                }
            }
        }
        if (session.gradePrefilled) {
            Text(stringResource(R.string.capture_grade_prefilled), style = BB.Caption, fontWeight = FontWeight.Bold, color = BB.Lime, modifier = Modifier.testTag("captureGradePrefilled"))
        }
        val stage = CaptureText.stage(session.tapMissed, session.color != null, assessment, recognised = session.color != null && route != null)
        Text(
            stringResource(CaptureText.prompt(stage)), style = BB.Caption, fontWeight = FontWeight.Bold, color = BB.Paper,
            modifier = Modifier.semantics { liveRegion = LiveRegionMode.Polite; stateDescription = stage.name.lowercase() }.testTag("capturePrompt"),
        )
        Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
            val chosen = session.color
            if (chosen == null) {
                // Swatches only: the page's name for a colour is right too rarely to print.
                session.palette.forEachIndexed { index, swatch ->
                    val label = stringResource(R.string.capture_colour_n, index + 1)
                    val count = stringResource(R.string.capture_hold_count, swatch.count)
                    Row(
                        Modifier.defaultMinSize(minHeight = 48.dp).clip(CircleShape).background(Chip)
                            .clickable(role = Role.Button) { onTool(CaptureRole.START_HAND); session.selectColor(swatch.key) }.padding(horizontal = 10.dp)
                            .testTag("captureColor-${swatch.key}").clearAndSetSemantics { contentDescription = "$label, $count" },
                        horizontalArrangement = Arrangement.spacedBy(6.dp), verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Swatch(swatch.rgb, 26.dp, Color.White)
                        Text(swatch.count.toString(), style = BB.mono(12.sp, FontWeight.Medium, 0.sp), color = BB.Paper)
                    }
                }
            } else {
                // The chosen colour (tap it to choose again), then what a tap on the picture marks.
                val change = stringResource(R.string.capture_change_colour)
                Row(
                    Modifier.defaultMinSize(minHeight = 48.dp).clip(CircleShape).background(Chip).clickable(role = Role.Button, onClickLabel = change) { session.selectColor(null) }
                        .padding(horizontal = 10.dp).testTag("captureChangeColour").clearAndSetSemantics { contentDescription = change },
                    horizontalArrangement = Arrangement.spacedBy(5.dp), verticalAlignment = Alignment.CenterVertically,
                ) {
                    Swatch(session.palette.firstOrNull { it.key == chosen }?.rgb ?: listOf(128, 128, 128), 26.dp, Color.White)
                    Icon(BBIcons.ChevronDown, contentDescription = null, tint = BB.Paper, modifier = Modifier.size(11.dp))
                }
                for (role in CaptureRole.entries) {
                    val active = tool == role
                    Box(
                        Modifier.defaultMinSize(minHeight = 48.dp).clip(CircleShape).background(if (active) BB.Lime else Chip)
                            .clickable(role = Role.Tab) { onTool(role) }.padding(horizontal = 12.dp).semantics { selected = active }.testTag("captureTool-${role.wire}"),
                        contentAlignment = Alignment.Center,
                    ) { Text(stringResource(CaptureText.role(role)), style = BB.Caption, fontWeight = FontWeight.Bold, color = if (active) BB.Ink else BB.Paper, maxLines = 1) }
                }
                if (assessment?.holds?.any { it.user } == true) {
                    Row(
                        Modifier.defaultMinSize(minHeight = 48.dp).clip(CircleShape).background(Chip).clickable(role = Role.Button, onClick = session::resetCorrections)
                            .padding(horizontal = 12.dp).testTag("captureResetMarks"),
                        horizontalArrangement = Arrangement.spacedBy(5.dp), verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Icon(BBIcons.Undo, contentDescription = null, tint = BB.Paper, modifier = Modifier.size(14.dp))
                        Text(stringResource(R.string.capture_reset_marks), style = BB.Caption, fontWeight = FontWeight.Bold, color = BB.Paper, maxLines = 1)
                    }
                }
            }
        }
    }
}

/** A colour as the capture page reported it: 0–255 red, green and blue. */
@Composable
internal fun Swatch(rgb: List<Int>, size: Dp, ring: Color, modifier: Modifier = Modifier) {
    val parts = if (rgb.size == 3) rgb else listOf(128, 128, 128)
    Box(modifier.size(size).clip(CircleShape).background(Color(parts[0].coerceIn(0, 255), parts[1].coerceIn(0, 255), parts[2].coerceIn(0, 255))).border(1.5.dp, ring, CircleShape))
}

/**
 * The colour chosen on the capture screen, as the swatch the page reported. No name: the page's
 * names for colours are not reliable enough to print.
 */
@Composable
internal fun CaptureColourLabel(climb: Climb, modifier: Modifier = Modifier) {
    val rgb = climb.capture?.colorRGB ?: return
    val label = stringResource(R.string.capture_colour_label)
    Swatch(rgb, 12.dp, BB.Ink.copy(alpha = 0.35f), modifier.testTag("captureColour").clearAndSetSemantics { contentDescription = label })
}

@Composable
private fun Unavailable(title: Int, detail: Int, onSettings: (() -> Unit)?, onImport: () -> Unit) {
    Column(
        Modifier.padding(12.dp).fillMaxWidth().clip(RoundedCornerShape(22.dp)).background(BB.Lavender).padding(22.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        Icon(BBIcons.Video, contentDescription = null, tint = BB.Ink, modifier = Modifier.size(40.dp))
        Text(stringResource(title), style = BB.Title2, color = BB.Ink, modifier = Modifier.testTag("captureUnavailable"))
        Text(stringResource(detail), style = BB.Body, color = BB.Muted)
        if (onSettings != null) FilledButton(stringResource(R.string.capture_open_settings), Modifier.fillMaxWidth().testTag("captureOpenSettings"), onClick = onSettings)
        OutlineButton(stringResource(R.string.capture_import_instead), Modifier.fillMaxWidth().testTag("captureImportInstead"), onClick = onImport)
    }
}

@Composable
private fun SaveFailed(message: String, onRetry: () -> Unit, onDiscard: () -> Unit) {
    Column(
        Modifier.padding(12.dp).fillMaxWidth().clip(RoundedCornerShape(22.dp)).background(BB.Paper).padding(22.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp),
    ) {
        Text(stringResource(R.string.capture_save_failed_title), style = BB.Title3, color = BB.Ink, modifier = Modifier.testTag("captureSaveFailed"))
        // What the system said about the failed write is shown as it is; it has no translation.
        Text(message, style = BB.Subheadline, color = BB.Muted)
        FilledButton(stringResource(R.string.capture_save_retry), Modifier.fillMaxWidth().testTag("captureSaveRetry"), onClick = onRetry)
        OutlineButton(stringResource(R.string.capture_save_discard), Modifier.fillMaxWidth().testTag("captureSaveDiscard"), color = Color(0xFFB3261E), onClick = onDiscard)
    }
}

/** Saved first. What happens to the clip afterwards is reported here but never waited for. */
@Composable
private fun SavedCard(climb: Climb, session: CaptureSession, model: JournalViewModel, onAgain: () -> Unit, onDone: () -> Unit, onViewAnalysis: () -> Unit) {
    val scope = rememberCoroutineScope()
    val thumbnail: File = model.thumbnailFile(climb)
    val image by produceState<ImageBitmap?>(null, thumbnail.path) {
        value = withContext(Dispatchers.IO) { runCatching { BitmapFactory.decodeFile(thumbnail.path)?.asImageBitmap() }.getOrNull() }
    }
    Column(
        Modifier.padding(10.dp).fillMaxWidth().clip(RoundedCornerShape(22.dp)).background(BB.Paper).padding(18.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
            Icon(BBIcons.Check, contentDescription = null, tint = BB.Ink, modifier = Modifier.size(22.dp))
            Text(stringResource(R.string.capture_saved_title), style = BB.Title3, color = BB.Ink, modifier = Modifier.testTag("captureSaved"))
        }
        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            Box(Modifier.size(width = 64.dp, height = 88.dp).clip(RoundedCornerShape(12.dp)).background(BB.Lavender)) {
                image?.let { Image(it, contentDescription = null, contentScale = ContentScale.Crop, modifier = Modifier.fillMaxSize()) }
            }
            Column(verticalArrangement = Arrangement.spacedBy(5.dp)) {
                Text("${gradeText(climb.grade)}  /  ${clockText(climb.analysis.duration)}", style = BB.mono(13.sp, FontWeight.Bold, 0.sp), color = BB.Ink)
                if (!climb.gymName.isNullOrBlank()) Text(climb.gymName, style = BB.Caption, color = BB.Muted, maxLines = 2, overflow = TextOverflow.Ellipsis)
                CaptureColourLabel(climb)
                CaptureText.stop(session.stopReason)?.let { reason ->
                    val text = if (reason == R.string.capture_stop_limit) stringResource(reason, CaptureLimits.MAX_MINUTES) else stringResource(reason)
                    Text(text, style = BB.Caption, color = BB.Muted, modifier = Modifier.testTag("captureStopReason"))
                }
            }
        }
        AnalysisStatus(climb, model, onViewAnalysis)
        SwitchRow(stringResource(R.string.journal_sent_toggle), climb.sent) { value -> scope.launch { model.saveNote(climb, climb.note, value) } }
        Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            FilledButton(stringResource(R.string.capture_again), Modifier.weight(1f).testTag("captureAgain"), onClick = onAgain)
            OutlineButton(stringResource(R.string.journal_import_done), Modifier.weight(1f).testTag("captureDone"), onClick = onDone)
        }
    }
}

@Composable
private fun AnalysisStatus(climb: Climb, model: JournalViewModel, onViewAnalysis: () -> Unit) {
    val results by model.analysisResults.collectAsState()
    val pose by model.reanalysis.collectAsState()
    val messages by model.queueMessages.collectAsState()
    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        when {
            climb.pendingAnalysis != null -> {
                Text(stringResource(R.string.capture_queued), style = BB.Caption, color = BB.Muted, modifier = Modifier.testTag("captureQueued"))
                ProgressBar(if (pose.climbId == climb.id && pose.busy) pose.progress else 0.0)
            }
            climb.capture?.analysisQueued == true -> Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
                CircularProgressIndicator(color = BB.Ink, trackColor = BB.Line, strokeWidth = 2.dp, modifier = Modifier.size(16.dp))
                Text(stringResource(R.string.capture_analysing), style = BB.Caption, color = BB.Muted, modifier = Modifier.testTag("captureAnalysing"))
            }
            climb.id in results -> Box(
                Modifier.fillMaxWidth().defaultMinSize(minHeight = 48.dp).clip(RoundedCornerShape(14.dp)).background(BB.Lime)
                    .clickable(role = Role.Button, onClick = onViewAnalysis).padding(horizontal = 16.dp, vertical = 12.dp).testTag("captureViewAnalysis"),
                contentAlignment = Alignment.Center,
            ) { Text(stringResource(R.string.journal_analysis_view), style = BB.Subheadline, fontWeight = FontWeight.SemiBold, color = BB.Ink) }
            // Pose recognition found nobody, or was stopped: what the service said about this record.
            pose.climbId == climb.id && pose.message != null ->
                Text(pose.message!!.text(), style = BB.Caption, color = BB.Muted, modifier = Modifier.testTag("captureAnalysisMessage"))
            messages[climb.id] != null -> Text(messages.getValue(climb.id).text(), style = BB.Caption, color = BB.Muted, modifier = Modifier.testTag("captureAnalysisMessage"))
        }
    }
}

// Bottom: record / stop

@Composable
private fun BottomBar(session: CaptureSession, framing: Boolean, onRecord: () -> Unit) {
    Box(Modifier.fillMaxWidth().height(78.dp), contentAlignment = Alignment.Center) {
        if (session.phase == CaptureSession.Phase.Recording) {
            val stop = stringResource(R.string.capture_stop_action)
            Box(
                Modifier.size(72.dp).clip(CircleShape).border(4.dp, BB.Paper, CircleShape)
                    .clickable(role = Role.Button, onClickLabel = stop) { session.stop(CaptureStop.USER) }.semantics { contentDescription = stop }.testTag("captureStop"),
                contentAlignment = Alignment.Center,
            ) { Box(Modifier.size(30.dp).clip(RoundedCornerShape(7.dp)).background(RecordRed)) }
        } else {
            val record = stringResource(R.string.capture_record_action)
            Box(
                Modifier.size(72.dp).alpha(if (framing) 1f else 0.3f).clip(CircleShape).border(4.dp, if (session.assessment?.ready == true) BB.Lime else BB.Paper, CircleShape)
                    .clickable(enabled = framing, role = Role.Button, onClickLabel = record, onClick = onRecord).semantics { contentDescription = record }.testTag("captureRecord"),
                contentAlignment = Alignment.Center,
            ) { Box(Modifier.size(58.dp).clip(CircleShape).background(RecordRed)) }
        }
    }
}
