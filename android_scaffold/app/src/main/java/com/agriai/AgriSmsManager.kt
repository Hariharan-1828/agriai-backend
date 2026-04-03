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
