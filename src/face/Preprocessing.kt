package face

import org.bytedeco.javacpp.opencv_core
import org.bytedeco.javacpp.opencv_core.Mat
import org.bytedeco.javacpp.opencv_imgproc.createCLAHE

object Preprocessing {
    fun equalizeHist(image: Mat): Mat {
        val clipLimit = 2.0
        val tileGridSize = opencv_core.Size(8, 8)
        val clahe = createCLAHE(clipLimit, tileGridSize)

        val result = Mat()
        clahe.apply(image, result)
        return result
    }

    fun changeBrightness(image: Mat, beta: Double): Mat {
        val result = Mat()
        image.convertTo(result, -1, 1.0, beta)
        return result
    }
}