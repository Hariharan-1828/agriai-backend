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
