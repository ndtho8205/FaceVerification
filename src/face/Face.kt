package face

import org.bytedeco.javacpp.opencv_core.Mat

data class Face(val image: Mat, val boundingBox: BoundingBox, val containerImageName: String)
{

    var alignedImage: Mat = image

    fun save(filePath: String)
    {
        JavaCvUtils.imsave(filePath, image)
    }

    data class BoundingBox(val x: Int, val y: Int, val w: Int, val h: Int)
}
