package face

import org.bytedeco.javacpp.opencv_core.Mat
import utils.JavaCvUtils

data class Face(val image: Mat, val containerImageName: String) {
    fun save(filePath: String) {
        JavaCvUtils.imsave(filePath, image)
    }
}
