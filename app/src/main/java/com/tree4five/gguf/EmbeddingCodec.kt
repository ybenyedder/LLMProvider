package com.tree4five.gguf

import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs

/**
 * Quantization format for embeddings crossing the Binder (~8x smaller than
 * float32 and far below the 512 KB per-transaction limit): per vector,
 * [float32 scale][int8 x dim] little-endian, with scale = maxAbs / 127.
 *
 * The exact same layout is implemented in harnessDroid's EmbeddingCodec.
 */
object EmbeddingCodec {

    /** Bytes occupied by one packed vector of the given dimension. */
    fun bytesPerVector(dim: Int): Int = 4 + dim

    /** Packs a single vector. */
    fun encodeOne(vector: FloatArray): ByteArray {
        val out = ByteArray(bytesPerVector(vector.size))
        val buf = ByteBuffer.wrap(out).order(ByteOrder.LITTLE_ENDIAN)
        buf.putFloat(maxAbsScale(vector))
        for (x in vector) buf.put(quantize(x, scaleOf(vector)))
        return out
    }

    /** Unpacks a stream of packed vectors into a flat [count * dim] array. */
    fun decodeFlat(bytes: ByteArray, dim: Int): FloatArray {
        val perVector = bytesPerVector(dim)
        val count = bytes.size / perVector
        val out = FloatArray(count * dim)
        val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        for (t in 0 until count) {
            val scale = buf.float
            val base = t * dim
            for (d in 0 until dim) out[base + d] = buf.get() * scale
        }
        return out
    }

    private fun maxAbsScale(vector: FloatArray): Float {
        var maxAbs = 0f
        for (x in vector) {
            val a = abs(x)
            if (a > maxAbs) maxAbs = a
        }
        return if (maxAbs == 0f) 1f else maxAbs / 127f
    }

    private fun scaleOf(vector: FloatArray): Float = maxAbsScale(vector)

    private fun quantize(x: Float, scale: Float): Byte =
        ((x / scale).toInt().coerceIn(-127, 127)).toByte()
}
