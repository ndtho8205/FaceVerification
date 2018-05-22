package face

import org.bytedeco.javacpp.opencv_core.*
import org.bytedeco.javacpp.opencv_objdetect.CascadeClassifier


object FaceDetection {
    private const val MIN_FACE_WIDTH_THRESHOLD = 160

    private val CASCADE_CLASSIFIER_PATH = javaClass.getResource("/src/face/haarcascade_frontalface_alt.xml").file

    private val mFaceCascade = CascadeClassifier(CASCADE_CLASSIFIER_PATH)

    private fun findBoundingBox(imgGray: Mat): Face.BoundingBox?
    {
        val facesRect = RectVector()

        mFaceCascade.detectMultiScale(imgGray, facesRect, 1.25, 5, 0,
                Size(MIN_FACE_WIDTH_THRESHOLD, MIN_FACE_WIDTH_THRESHOLD),
                Size(4 * MIN_FACE_WIDTH_THRESHOLD,
                        4 * MIN_FACE_WIDTH_THRESHOLD))

        val faceIndex = when
        {
            facesRect.size() == 1L -> if (facesRect.get(0).width() >= MIN_FACE_WIDTH_THRESHOLD) 0L else -1L
            facesRect.size() > 1   ->
            {
                var maxFaceIndex: Long = -1
                var maxFaceWidth = -1
                for (i in 0 until facesRect.size())
                {
                    if (facesRect.get(i).width() > maxFaceWidth)
                    {
                        maxFaceWidth = facesRect.get(i).width()
                        maxFaceIndex = i
                    }
                }
                if (maxFaceWidth >= MIN_FACE_WIDTH_THRESHOLD) maxFaceIndex else -1
            }
            else                   -> -1
        }

        if (faceIndex < 0)
            return null
        val faceRect = facesRect.get(faceIndex)

        return Face.BoundingBox(faceRect.x(), faceRect.y(), faceRect.width(), faceRect.height())
    }

    fun detect(imgGray: Mat, imageName: String = ""): Face?
    {
        val faceBoundingBox = findBoundingBox(imgGray) ?: return null
        val faceRect =
                Rect(faceBoundingBox.x, faceBoundingBox.y, faceBoundingBox.w, faceBoundingBox.h)
        val face = Mat(imgGray, faceRect)

        return Face(face, faceBoundingBox, imageName)
    }
}
