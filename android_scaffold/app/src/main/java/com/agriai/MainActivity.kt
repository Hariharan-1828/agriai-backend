package com.agriai

import android.Manifest
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Build
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
import com.google.firebase.functions.FirebaseFunctions
import com.google.firebase.ktx.Firebase
import com.google.firebase.functions.ktx.functions
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
