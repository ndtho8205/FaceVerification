package face

import org.bytedeco.javacpp.opencv_core.Mat
import org.bytedeco.javacpp.opencv_core.RectVector
import org.bytedeco.javacpp.opencv_objdetect.CascadeClassifier


object Detection {
    private const val CASCADE_CLASSIFIER_PATH = "/home/ndtho/Documents/Projects/FaceVerification/resources/xml/haarcascade_frontalface_default.xml"
    private val mFaceCascade = CascadeClassifier(CASCADE_CLASSIFIER_PATH)

    fun detect(imgGray: Mat): Mat? {
        val faces = RectVector()

        mFaceCascade.detectMultiScale(imgGray, faces)
        println("Number of face detected in images: " + faces.size())

        val face = when {
            faces.size() == 1L -> Mat(imgGray, faces.get(0))
            faces.size() > 1 -> {
                println("More than one face detected.")
                var maxFaceIndex: Long = -1
                var maxFaceWidth = -1
                for (i in 0 until faces.size()) {
                    if (faces.get(i).width() > maxFaceWidth) {
                        maxFaceWidth = faces.get(i).width()
                        maxFaceIndex = i
                    }
                }
                Mat(imgGray, faces.get(maxFaceIndex))
            }
            else -> {
                println("No face detected.")
                null
            }
        }

        return face
    }
}
