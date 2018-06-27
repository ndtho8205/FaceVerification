package face

import org.bytedeco.javacpp.DoublePointer
import org.bytedeco.javacpp.IntPointer
import org.bytedeco.javacpp.opencv_core.CV_32SC1
import org.bytedeco.javacpp.opencv_core.Mat
import org.bytedeco.javacpp.opencv_face.EigenFaceRecognizer
import java.nio.IntBuffer

class FaceVerification
{

    companion object
    {

        const val FACE_IMAGE_QUANTITY = 25
    }

    private val TAG = "FaceVerification"

    private val MODEL_FILENAME = "model_face.yml"

    private val mRecognizer = EigenFaceRecognizer.create(128, 10000.0)

    private var mIsTrained = false

    fun loadTrainedModel(modelPath: String)
    {
        mIsTrained = try
        {
            mRecognizer.read(modelPath)
            true
        }
        catch (e: Exception)
        {
            e.printStackTrace()
            false
        }
    }

    fun saveTrainedModel(modelPath: String)
    {
        mRecognizer.save("$modelPath/$MODEL_FILENAME")
    }

    fun train(faces: List<Face>): Boolean
    {
        return try
        {
            val faceImageList = faces.map { it.alignedFaceImage; }
            if (faceImageList.isEmpty())
                return false

            val labels = generateLabels(faces)

            mRecognizer.train(JavaCvUtils.list2MatVector(faceImageList), labels)
            mIsTrained = true
            true
        }
        catch (e: Exception)
        {
            e.printStackTrace()
            false
        }
    }

    private fun generateLabels(faces: List<Face>): Mat
    {
        val labels = Mat(faces.size, 1, CV_32SC1)
        val label = 100
        val labelsBuf: IntBuffer = labels.createBuffer()

        for ((counter, _) in faces.withIndex())
        {
            labelsBuf.put(counter, label)
        }

        return labels
    }

    fun predict(testFace: Face): Double
    {
        if (mIsTrained)
        {
            val label = IntPointer(1)
            val confidence = DoublePointer(1)

            mRecognizer.predict(testFace.alignedFaceImage, label, confidence)

            val predictedLabel = label.get(0)
            val predictedConfidence = confidence.get(0)


            return predictedConfidence
        }
        else
        {
            throw Exception("FaceVerification is not trained.")
        }
    }

    fun predict(testFaces: List<Face>): List<Double>
    {
        return testFaces.map {
            predict(it)
        }
    }
}
