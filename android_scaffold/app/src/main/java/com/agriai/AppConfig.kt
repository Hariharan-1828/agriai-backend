package com.agriai

object AppConfig {
    // Server SMS number — set via BuildConfig or override here
    val SERVER_NUMBER: String
        get() = try {
            BuildConfig.SERVER_NUMBER
        } catch (e: Exception) {
            "+12815008847"  // fallback to your real Twilio number
        }

    const val EMBED_DIM     = 64     // now PCA-64 (was 32)
    const val IMG_SIZE      = 224
    const val MODEL_FILE    = "mobilenet_v3_small.tflite"
    const val SMS_TIMEOUT   = 90_000L  // 90 seconds
}
