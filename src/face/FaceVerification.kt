package face

import org.bytedeco.javacpp.DoublePointer
import org.bytedeco.javacpp.IntPointer
import org.bytedeco.javacpp.opencv_core.CV_32SC1
import org.bytedeco.javacpp.opencv_core.Mat
import org.bytedeco.javacpp.opencv_face.createEigenFaceRecognizer
import java.nio.IntBuffer


class FaceVerification {
    val FACE_IMAGE_QUANTITY = 25
    val MODEL_FILENAME = "model.yml"

    private val mRecognizer = createEigenFaceRecognizer(128, 10000.0)

    fun load(trainedResult: String) {
        mRecognizer.load(trainedResult)
    }

    fun save(path: String) {
        mRecognizer.save(path)
    }


    fun train(faces: List<Face>): Boolean {
        val labels = generateLabels(faces)
        val faceImageList = faces.map { it.alignedImage }
        if (faceImageList.isEmpty())
            return false
        return try
        {
            mRecognizer.train(JavaCvUtils.list2MatVector(faceImageList), labels)
            true
        } catch (e: Exception) {
            println(e)
            false
        }
    }

    private fun generateLabels(faces: List<Face>): Mat {
        val labels = Mat(faces.size, 1, CV_32SC1)
        val label = 100
        val labelsBuf: IntBuffer = labels.createBuffer()

        for ((counter, _) in faces.withIndex()) {
            labelsBuf.put(counter, label)
        }

        return labels
    }

    fun predict(testFace: Face) {
        val label = IntPointer(1)
        val confidence = DoublePointer(1)
        mRecognizer.predict(testFace.image, label, confidence)

        val predictedLabel = label.get(0)
        val predictedConfidence = confidence.get(0)
        println("Face: ${testFace.containerImageName}")
        println("Predicted label: $predictedLabel")
        println("Confidence: $predictedConfidence")
    }

    fun predict(testFaces: List<Face>) {
        testFaces.forEach {
            predict(it)
            println()
        }
    }
}