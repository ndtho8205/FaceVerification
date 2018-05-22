package face

import org.bytedeco.javacpp.indexer.UByteBufferIndexer
import org.bytedeco.javacpp.opencv_core.Mat


class FaceQualityComputation
{

    // Calculate brightness of image based on YUV color space
    fun computeBrightnessScore(rgbaMat: Mat): Float
    {
        val brightness: Float

        val brightElem = rgbaMat.createIndexer<UByteBufferIndexer>()
        var sum = 0f
        for (x in 0 until rgbaMat.rows())
        {
            for (y in 0 until rgbaMat.cols())
            {
                val r = brightElem.get(x.toLong(), y.toLong(), 2)
                val g = brightElem.get(x.toLong(), y.toLong(), 1)
                val b = brightElem.get(x.toLong(), y.toLong(), 0)
                val Y = (0.299 * r + 0.587 * g + 0.114 * b).toFloat()
                sum += Y
            }
        }
        brightness = sum / (rgbaMat.rows() * rgbaMat.cols())

        return brightness
    }

    // Calculate contrast of image using the Root Mean Square (RMS) equation
    fun computeContrastScore(grayMat: Mat): Float
    {
        val contrast: Float

        val contrastElem = grayMat.createIndexer<UByteBufferIndexer>()
        var mean = 0f
        for (x in 0 until grayMat.rows())
        {
            for (y in 0 until grayMat.cols())
            {
                mean += contrastElem.get(x.toLong(), y.toLong()).toFloat()
            }
        }
        mean /= (grayMat.rows() * grayMat.cols()).toFloat()

        var temp = 0f
        for (x in 0 until grayMat.rows())
        {
            for (y in 0 until grayMat.cols())
            {
                temp += Math.pow((contrastElem.get(x.toLong(), y.toLong()) - mean).toDouble(), 2.0).toFloat()
            }
        }
        contrast = Math.sqrt((temp / (grayMat.rows() * grayMat.cols())).toDouble()).toFloat()

        return contrast
    }

    //Calculate sharpness of image
    // Based on research of Kryszczuk and Drygajlo (Paper name: "On combining evidence for reliability estimation in face verification")
    fun computeSharpnessScore(grayMat: Mat): Float
    {
        val sharpness: Float

        val sharpElem = grayMat.createIndexer<UByteBufferIndexer>()

        var sumX = 0f
        for (x in 0 until grayMat.rows())
        {
            for (y in 0 until grayMat.cols() - 1)
            {
                sumX += Math.abs(sharpElem.get(x.toLong(), y.toLong()) - sharpElem.get(x.toLong(), (y + 1).toLong()))
                        .toFloat()
            }
        }
        sumX /= (grayMat.rows() * (grayMat.cols() - 1)).toFloat()

        var sumY = 0f
        for (x in 0 until grayMat.rows() - 1)
        {
            for (y in 0 until grayMat.cols())
            {
                sumY += Math.abs(sharpElem.get(x.toLong(), y.toLong()) - sharpElem.get((x + 1).toLong(), y.toLong()))
                        .toFloat()
            }
        }
        sumY /= (grayMat.cols() * (grayMat.rows() - 1)).toFloat()

        sharpness = (0.5 * (sumX + sumY)).toFloat()

        return sharpness
    }

}
