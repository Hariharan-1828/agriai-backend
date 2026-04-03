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
