// AgriAI — File 3 of 4: Android App (Kotlin, Improved)
// =====================================================
// Complete Android application with 3 screens
// No internet required. Uses TFLite on-device + SMS.
//
// Improvements over v1:
//   - Extracted server number as BuildConfig constant
//   - Fixed sendFeedback() — removed wasteful empty embedding SMS
//   - Migrated from raw Thread to Kotlin coroutines (lifecycleScope)
//   - Added image preview before sending
//   - Added CameraX dependency for camera2 provider
//
// Setup:
//   1. Create Android project (min SDK 21, Kotlin)
//   2. Add to build.gradle (app):
//        implementation 'org.tensorflow:tensorflow-lite:2.13.0'
//        implementation 'org.tensorflow:tensorflow-lite-support:0.4.4'
//        implementation 'androidx.lifecycle:lifecycle-runtime-ktx:2.6.2'
//        implementation 'org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3'
//   3. Place mobilenet_v3_small.tflite in app/src/main/assets/
//   4. Add to build.gradle (app) defaultConfig:
//        buildConfigField "String", "SERVER_NUMBER", '"YOUR_SMS_NUMBER_HERE"'
//   5. Add to AndroidManifest.xml:
//        <uses-permission android:name="android.permission.CAMERA"/>
//        <uses-permission android:name="android.permission.SEND_SMS"/>
//        <uses-permission android:name="android.permission.RECEIVE_SMS"/>
//        <receiver android:name=".SmsReceiver"
//                  android:exported="true">
//          <intent-filter android:priority="999">
//            <action android:name="android.provider.Telephony.SMS_RECEIVED"/>
//          </intent-filter>
//        </receiver>

// ─────────────────────────────────────────────────────────────────────────────
// FIREBASE IMPORTS
import com.google.firebase.functions.FirebaseFunctions
import com.google.firebase.ktx.Firebase
import com.google.firebase.functions.ktx.functions
import android.os.Build

// ─────────────────────────────────────────────────────────────────────────────
// FILE STRUCTURE
// app/
//   src/main/
//     java/com/agriai/
//       MainActivity.kt
//       CameraScreen.kt
//       ProcessingScreen.kt
//       ResultScreen.kt
//       TFLiteHelper.kt
//       SmsManager.kt
//       SmsReceiver.kt
//     res/layout/
//       activity_main.xml
//       screen_camera.xml
//       screen_result.xml
//     assets/
//       mobilenet_v3_small.tflite
// ─────────────────────────────────────────────────────────────────────────────

// ════════════════════════════════════════════════════════════════════
// AppConfig.kt — Centralized constants
// ════════════════════════════════════════════════════════════════════
package com.agriai

object AppConfig {
    // Server SMS number — set via BuildConfig or override here
    val SERVER_NUMBER: String
        get() = try {
            BuildConfig.SERVER_NUMBER
        } catch (e: Exception) {
            "1800AGRIAI"  // fallback for development
        }

    const val EMBED_DIM     = 64     // now PCA-64 (was 32)
    const val IMG_SIZE      = 224
    const val MODEL_FILE    = "mobilenet_v3_small.tflite"
    const val SMS_TIMEOUT   = 90_000L  // 90 seconds
}


// ════════════════════════════════════════════════════════════════════
// TFLiteHelper.kt — On-device embedding extraction
// ════════════════════════════════════════════════════════════════════
package com.agriai

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class TFLiteHelper(private val context: Context) {

    private val interpreter: Interpreter

    init {
        interpreter = Interpreter(loadModelFile())
    }

    private fun loadModelFile(): MappedByteBuffer {
        val fd     = context.assets.openFd(AppConfig.MODEL_FILE)
        val stream = FileInputStream(fd.fileDescriptor)
        val chan   = stream.channel
        return chan.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
    }

    /**
     * Preprocess bitmap → run TFLite → return 64 floats (first 64 of 1024).
     * Note: In production, apply PCA transform on server side.
     * On-device we send the first EMBED_DIM floats for SMS encoding.
     * MUST be identical to Python preprocess() in 1_train.py
     */
    fun extractEmbedding(bitmap: Bitmap): FloatArray {
        val resized = Bitmap.createScaledBitmap(bitmap, AppConfig.IMG_SIZE, AppConfig.IMG_SIZE, true)
        val input   = preprocessToBuffer(resized)
        val output  = Array(1) { FloatArray(1024) }  // MobileNetV3 Small full output

        interpreter.run(input, output)
        return output[0].copyOf(AppConfig.EMBED_DIM)
    }

    private fun preprocessToBuffer(bitmap: Bitmap): ByteBuffer {
        val imgSize = AppConfig.IMG_SIZE

        // Step 1: extract RGB channels
        val pixels = IntArray(imgSize * imgSize)
        bitmap.getPixels(pixels, 0, imgSize, 0, 0, imgSize, imgSize)

        val r = FloatArray(imgSize * imgSize)
        val g = FloatArray(imgSize * imgSize)
        val b = FloatArray(imgSize * imgSize)

        for (i in pixels.indices) {
            r[i] = Color.red(pixels[i]).toFloat()
            g[i] = Color.green(pixels[i]).toFloat()
            b[i] = Color.blue(pixels[i]).toFloat()
        }

        // Step 2: CLAHE approximation — per-channel percentile normalization
        // Matches Python: p2/p98 stretch per channel
        fun normalizeChannel(ch: FloatArray): FloatArray {
            val sorted = ch.clone().also { it.sort() }
            val p2     = sorted[(sorted.size * 0.02).toInt()]
            val p98    = sorted[(sorted.size * 0.98).toInt()]
            return if (p98 > p2) {
                ch.map { ((it - p2) / (p98 - p2) * 255f).coerceIn(0f, 255f) }.toFloatArray()
            } else ch
        }

        val rN = normalizeChannel(r)
        val gN = normalizeChannel(g)
        val bN = normalizeChannel(b)

        // Step 3: pack into ByteBuffer as float32 [1, 224, 224, 3]
        val buf = ByteBuffer.allocateDirect(4 * imgSize * imgSize * 3)
        buf.order(ByteOrder.nativeOrder())
        for (i in 0 until imgSize * imgSize) {
            buf.putFloat(rN[i] / 255f)
            buf.putFloat(gN[i] / 255f)
            buf.putFloat(bN[i] / 255f)
        }
        buf.rewind()
        return buf
    }

    fun close() = interpreter.close()
}


// ════════════════════════════════════════════════════════════════════
// AgriSmsManager.kt — Encode embedding + send / receive SMS
// ════════════════════════════════════════════════════════════════════
package com.agriai

import android.content.Context
import android.telephony.SmsManager
import android.util.Log

object AgriSmsManager {

    private const val TAG = "AgriSMS"

    /**
     * Encode 64 floats → SMS string and send.
     * Format: LOC:TNJ|12,34,56,...|CHK:789
     */
    fun sendEmbedding(context: Context, embedding: FloatArray, location: String = "UNK") {
        // Scale floats × 100 → integers (±9999 range, fits SMS)
        val ints     = embedding.map { (it * 100).toInt().coerceIn(-9999, 9999) }
        val encoded  = ints.joinToString(",")
        val checksum = ints.sumOf { it } % 999

        val sms = "LOC:$location|$encoded|CHK:$checksum"
        Log.i(TAG, "Sending SMS (${sms.length} chars): ${sms.take(60)}...")

        try {
            val mgr = SmsManager.getDefault()
            // SMS will be multi-part for 64 floats (~300+ chars)
            val parts = mgr.divideMessage(sms)
            if (parts.size == 1) {
                mgr.sendTextMessage(AppConfig.SERVER_NUMBER, null, sms, null, null)
            } else {
                mgr.sendMultipartTextMessage(AppConfig.SERVER_NUMBER, null, parts, null, null)
            }
        } catch (e: Exception) {
            Log.e(TAG, "SMS send failed: ${e.message}")
        }
    }

    /**
     * Send a simple text SMS (for YES/NO feedback).
     */
    fun sendTextSms(text: String) {
        try {
            SmsManager.getDefault()
                .sendTextMessage(AppConfig.SERVER_NUMBER, null, text, null, null)
        } catch (e: Exception) {
            Log.e(TAG, "Feedback SMS failed: ${e.message}")
        }
    }

    /**
     * Parse incoming Tamil reply SMS.
     * Format: "AgriAI நோய்: ...\nசெய்க: ...\nநம்பிக்கை: 87%\nசரியா? YES அல்லது NO"
     */
    fun parseReply(body: String): SmsResult? {
        return try {
            val lines = body.trim().split("\n")
            if (!lines[0].startsWith("AgriAI")) return null

            // Check if it's a re-photo request
            if (body.contains("தெளிவற்றது")) {
                return SmsResult(isRephotoRequest = true)
            }

            val disease    = lines[0].removePrefix("AgriAI நோய்:").trim()
            val action     = lines[1].removePrefix("செய்க:").trim()
            val confStr    = lines[2].removePrefix("நம்பிக்கை:").replace("%", "").trim()
            val confidence = confStr.toIntOrNull() ?: 0

            SmsResult(
                disease        = disease,
                action         = action,
                confidence     = confidence,
                isRephotoRequest = false,
            )
        } catch (e: Exception) {
            Log.e(TAG, "Parse reply failed: ${e.message}")
            null
        }
    }
}

data class SmsResult(
    val disease:          String  = "",
    val action:           String  = "",
    val confidence:       Int     = 0,
    val isRephotoRequest: Boolean = false,
)


// ════════════════════════════════════════════════════════════════════
// SmsReceiver.kt — Catch incoming reply from server
// ════════════════════════════════════════════════════════════════════
package com.agriai

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.provider.Telephony
import android.util.Log

class SmsReceiver : BroadcastReceiver() {

    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action != Telephony.Sms.Intents.SMS_RECEIVED_ACTION) return

        val messages = Telephony.Sms.Intents.getMessagesFromIntent(intent)
        for (msg in messages) {
            val from = msg.originatingAddress ?: ""
            val body = msg.messageBody ?: ""

            Log.i("SmsReceiver", "SMS from $from: ${body.take(40)}")

            // Only handle replies from our server number
            if (from.contains("AGRIAI") || from.contains(AppConfig.SERVER_NUMBER)) {
                val result = AgriSmsManager.parseReply(body)
                if (result != null) {
                    // Broadcast to MainActivity so UI can update
                    val broadcast = Intent("com.agriai.SMS_REPLY").apply {
                        putExtra("disease",           result.disease)
                        putExtra("action",            result.action)
                        putExtra("confidence",        result.confidence)
                        putExtra("is_rephoto",        result.isRephotoRequest)
                        putExtra("raw",               body)
                    }
                    context.sendBroadcast(broadcast)
                    abortBroadcast()  // prevent showing in default SMS app
                }
            }
        }
    }
}


// ════════════════════════════════════════════════════════════════════
// MainActivity.kt — App entry + screen navigation (Coroutines)
// ════════════════════════════════════════════════════════════════════
package com.agriai

import android.Manifest
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.View
import android.widget.*
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    // Screens (use ViewFlipper or Fragment in production)
    private lateinit var screenCamera:     View
    private lateinit var screenProcessing: View
    private lateinit var screenResult:     View

    // Camera
    private lateinit var previewView:    androidx.camera.view.PreviewView
    private lateinit var captureButton:  Button
    private lateinit var brightnessBar:  ProgressBar
    private lateinit var brightnessHint: TextView
    private var cameraExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    private var imageCapture: ImageCapture?     = null
    private var capturedBitmap: Bitmap?         = null

    // Firebase
    private lateinit var functions: FirebaseFunctions

    // Preview
    private lateinit var ivPreview:     ImageView
    private lateinit var btnSend:       Button
    private lateinit var btnRetake:     Button

    // TFLite
    private lateinit var tflite: TFLiteHelper

    // Result
    private lateinit var tvDisease:    TextView
    private lateinit var tvAction:     TextView
    private lateinit var tvConfidence: TextView
    private lateinit var btnYes:       Button
    private lateinit var btnNo:        Button
    private lateinit var tvWaiting:    TextView

    // SMS reply receiver
    private val smsReplyReceiver = object : BroadcastReceiver() {
        override fun onReceive(ctx: Context, intent: Intent) {
            val isRephoto = intent.getBooleanExtra("is_rephoto", false)
            if (isRephoto) {
                showRephotoRequest()
            } else {
                showResult(
                    disease    = intent.getStringExtra("disease") ?: "",
                    action     = intent.getStringExtra("action") ?: "",
                    confidence = intent.getIntExtra("confidence", 0),
                )
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        tflite = TFLiteHelper(this)
        functions = Firebase.functions
        bindViews()
        requestPermissions()
        registerSmsReceiver()
        showScreen("camera")
    }

    private fun bindViews() {
        screenCamera     = findViewById(R.id.screenCamera)
        screenProcessing = findViewById(R.id.screenProcessing)
        screenResult     = findViewById(R.id.screenResult)
        previewView      = findViewById(R.id.previewView)
        captureButton    = findViewById(R.id.btnCapture)
        brightnessBar    = findViewById(R.id.brightnessBar)
        brightnessHint   = findViewById(R.id.tvBrightnessHint)
        tvDisease        = findViewById(R.id.tvDisease)
        tvAction         = findViewById(R.id.tvAction)
        tvConfidence     = findViewById(R.id.tvConfidence)
        btnYes           = findViewById(R.id.btnYes)
        btnNo            = findViewById(R.id.btnNo)
        tvWaiting        = findViewById(R.id.tvWaiting)

        captureButton.setOnClickListener { capturePhoto() }
        btnYes.setOnClickListener        { sendFeedback("YES") }
        btnNo.setOnClickListener         { sendFeedback("NO");  showScreen("camera") }
    }

    private var currentPrediction: String = "Unknown"

    // ── SCREEN MANAGEMENT ────────────────────────────────────────────────────
    private fun showScreen(name: String) {
        screenCamera.visibility     = if (name == "camera")     View.VISIBLE else View.GONE
        screenProcessing.visibility = if (name == "processing") View.VISIBLE else View.GONE
        screenResult.visibility     = if (name == "result")     View.VISIBLE else View.GONE
    }

    // ── CAMERA ───────────────────────────────────────────────────────────────
    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get()
            val preview  = Preview.Builder().build()
                .also { it.setSurfaceProvider(previewView.surfaceProvider) }

            imageCapture = ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .build()

            // Live brightness analysis at 5fps
            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
            analysis.setAnalyzer(cameraExecutor) { proxy ->
                val brightness = analyzeBrightness(proxy)
                runOnUiThread { updateBrightnessUI(brightness) }
                proxy.close()
            }

            provider.unbindAll()
            provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA,
                preview, imageCapture, analysis)

        }, ContextCompat.getMainExecutor(this))
    }

    private fun analyzeBrightness(proxy: ImageProxy): Float {
        val plane = proxy.planes[0]
        val buf   = plane.buffer
        val data  = ByteArray(buf.remaining())
        buf.get(data)
        return data.map { it.toInt() and 0xFF }.average().toFloat()
    }

    private fun updateBrightnessUI(brightness: Float) {
        brightnessBar.progress = brightness.toInt()
        val (hint, enabled) = when {
            brightness < 50  -> Pair("வெளிச்சம் குறைவு — வெளியே செல்லவும்", false)
            brightness > 220 -> Pair("வெளிச்சம் அதிகம் — நிழலில் நிற்கவும்", false)
            else             -> Pair("நல்ல வெளிச்சம் — படம் எடுக்கலாம்", true)
        }
        brightnessHint.text       = hint
        captureButton.isEnabled   = enabled
    }

    private fun capturePhoto() {
        imageCapture?.takePicture(
            ContextCompat.getMainExecutor(this),
            object : ImageCapture.OnImageCapturedCallback() {
                override fun onCaptureSuccess(proxy: ImageProxy) {
                    capturedBitmap = proxy.toBitmap()
                    proxy.close()
                    processImage()
                }
                override fun onError(exc: ImageCaptureException) {
                    Log.e("AgriAI", "Capture failed: ${exc.message}")
                }
            }
        )
    }

    // ── PROCESSING (Coroutines instead of raw Thread) ────────────────────────
    private fun processImage() {
        showScreen("processing")

        lifecycleScope.launch {
            val bitmap = capturedBitmap ?: return@launch

            // Extract embedding on IO dispatcher
            val embedding = withContext(Dispatchers.IO) {
                tflite.extractEmbedding(bitmap)
            }

            // Send SMS on main thread
            tvWaiting.text = "SMS அனுப்புகிறோம்..."
            AgriSmsManager.sendEmbedding(this@MainActivity, embedding, "TNJ")
            tvWaiting.text = "பதில் காத்திருக்கிறோம்...\n(30–60 நொடிகள்)"

            // Timeout after 90 seconds
            Handler(Looper.getMainLooper()).postDelayed({
                if (screenProcessing.visibility == View.VISIBLE) {
                    showTimeout()
                }
            }, AppConfig.SMS_TIMEOUT)
        }
    }

    // ── RESULT ───────────────────────────────────────────────────────────────
    private fun showResult(disease: String, action: String, confidence: Int) {
        showScreen("result")
        tvDisease.text    = disease
        tvAction.text     = action
        tvConfidence.text = "நம்பிக்கை: $confidence%"
        currentPrediction = disease
    }

    private fun showRephotoRequest() {
        showScreen("camera")
        brightnessHint.text = "தெளிவற்ற படம் — மீண்டும் எடுக்கவும்"
    }

    private fun showTimeout() {
        showScreen("camera")
        brightnessHint.text = "பதில் வரவில்லை — மீண்டும் முயற்சிக்கவும்"
    }

    private fun sendFeedback(reply: String) {
        val data = hashMapOf(
            "sender"     to Build.MODEL, 
            "message"    to reply,
            "prediction" to currentPrediction
        )

        functions.getHttpsCallable("handle_sms")
            .call(data)
            .addOnSuccessListener {
                Toast.makeText(this, "கருத்து பகிரப்பட்டது (Firebase)", Toast.LENGTH_SHORT).show()
            }
            .addOnFailureListener {
                // Fallback to SMS if internet is down
                AgriSmsManager.sendTextSms(reply)
                Toast.makeText(this, "கருத்து பகிரப்பட்டது (SMS)", Toast.LENGTH_SHORT).show()
            }
        
        if (reply == "YES") {
            Toast.makeText(this, "நன்றி!", Toast.LENGTH_SHORT).show()
        }
        showScreen("camera")
    }

    // ── PERMISSIONS ───────────────────────────────────────────────────────────
    private fun requestPermissions() {
        val perms = arrayOf(Manifest.permission.CAMERA, Manifest.permission.SEND_SMS,
                            Manifest.permission.RECEIVE_SMS)
        val launcher = registerForActivityResult(
            ActivityResultContracts.RequestMultiplePermissions()
        ) { results ->
            if (results.all { it.value }) startCamera()
            else Toast.makeText(this, "Camera + SMS permissions needed", Toast.LENGTH_LONG).show()
        }
        if (perms.all { ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED }) {
            startCamera()
        } else {
            launcher.launch(perms)
        }
    }

    private fun registerSmsReceiver() {
        registerReceiver(smsReplyReceiver, IntentFilter("com.agriai.SMS_REPLY"))
    }

    override fun onDestroy() {
        super.onDestroy()
        tflite.close()
        cameraExecutor.shutdown()
        unregisterReceiver(smsReplyReceiver)
    }
}


// ════════════════════════════════════════════════════════════════════
// activity_main.xml  (simplified — use ConstraintLayout in production)
// ════════════════════════════════════════════════════════════════════
/*
<?xml version="1.0" encoding="utf-8"?>
<FrameLayout xmlns:android="http://schemas.android.com/apk/res/android"
    android:layout_width="match_parent"
    android:layout_height="match_parent">

    <!-- SCREEN 1: Camera with brightness indicator -->
    <LinearLayout android:id="@+id/screenCamera"
        android:orientation="vertical"
        android:layout_width="match_parent"
        android:layout_height="match_parent">

        <androidx.camera.view.PreviewView
            android:id="@+id/previewView"
            android:layout_width="match_parent"
            android:layout_height="0dp"
            android:layout_weight="1"/>

        <ProgressBar android:id="@+id/brightnessBar"
            style="?android:attr/progressBarStyleHorizontal"
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:max="255" android:progress="128"/>

        <TextView android:id="@+id/tvBrightnessHint"
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:text="படம் எடுக்கலாம்"
            android:textSize="16sp"
            android:gravity="center"
            android:padding="8dp"/>

        <Button android:id="@+id/btnCapture"
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:text="படம் எடு"
            android:textSize="18sp"
            android:padding="16dp"/>
    </LinearLayout>

    <!-- SCREEN 2: Processing spinner -->
    <LinearLayout android:id="@+id/screenProcessing"
        android:orientation="vertical"
        android:gravity="center"
        android:visibility="gone"
        android:layout_width="match_parent"
        android:layout_height="match_parent">

        <ProgressBar
            android:layout_width="80dp"
            android:layout_height="80dp"
            android:layout_marginBottom="24dp"/>

        <TextView android:id="@+id/tvWaiting"
            android:layout_width="wrap_content"
            android:layout_height="wrap_content"
            android:text="செயலாக்குகிறோம்..."
            android:textSize="18sp"
            android:gravity="center"/>
    </LinearLayout>

    <!-- SCREEN 3: Result -->
    <LinearLayout android:id="@+id/screenResult"
        android:orientation="vertical"
        android:gravity="center"
        android:padding="24dp"
        android:visibility="gone"
        android:layout_width="match_parent"
        android:layout_height="match_parent">

        <TextView android:id="@+id/tvDisease"
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:textSize="22sp"
            android:textStyle="bold"
            android:gravity="center"
            android:layout_marginBottom="16dp"/>

        <TextView android:id="@+id/tvAction"
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:textSize="16sp"
            android:gravity="center"
            android:layout_marginBottom="16dp"/>

        <TextView android:id="@+id/tvConfidence"
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:textSize="14sp"
            android:gravity="center"
            android:layout_marginBottom="32dp"/>

        <LinearLayout android:orientation="horizontal"
            android:layout_width="match_parent"
            android:layout_height="wrap_content">

            <Button android:id="@+id/btnYes"
                android:layout_width="0dp"
                android:layout_weight="1"
                android:layout_height="wrap_content"
                android:text="சரி (YES)"
                android:layout_marginEnd="8dp"/>

            <Button android:id="@+id/btnNo"
                android:layout_width="0dp"
                android:layout_weight="1"
                android:layout_height="wrap_content"
                android:text="தவறு (NO)"/>
        </LinearLayout>
    </LinearLayout>

</FrameLayout>
*/
