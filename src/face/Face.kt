package face

import org.bytedeco.javacpp.opencv_core.Mat
import org.bytedeco.javacpp.opencv_imgproc.COLOR_RGB2BGR
import org.bytedeco.javacpp.opencv_imgproc.cvtColor

data class Face(val containerImage: Mat,
                val faceImageName: String,
                val faceImage: Mat)
{

    val alignedFaceImage: Mat

    init
    {
        alignedFaceImage = preprocessing(faceImage)
    }

    fun save(path: String)
    {
        val faceImagePath = "$path/${faceImageName}_face"
        val containerImagePath = "$path/${faceImageName}_container"

        JavaCvUtils.imsave(faceImagePath, alignedFaceImage)

        val bgrImage = Mat()
        cvtColor(containerImage, bgrImage, COLOR_RGB2BGR)
        JavaCvUtils.imsave(containerImagePath, bgrImage)
    }

    private fun preprocessing(faceImage: Mat): Mat
    {
        val preprocessor = ImagePreprocessing()
        val aligned = preprocessor.scaleToStandardSize(faceImage)
        return aligned
    }
}
