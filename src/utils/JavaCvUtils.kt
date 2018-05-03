package utils

import org.bytedeco.javacpp.opencv_core
import org.bytedeco.javacpp.opencv_core.Mat
import org.bytedeco.javacpp.opencv_core.MatVector
import org.bytedeco.javacpp.opencv_imgcodecs.imread
import org.bytedeco.javacpp.opencv_imgcodecs.imwrite
import org.bytedeco.javacpp.opencv_imgproc.*
import org.bytedeco.javacv.CanvasFrame
import org.bytedeco.javacv.OpenCVFrameConverter
import java.io.File
import java.io.FilenameFilter
import javax.swing.WindowConstants


object JavaCvUtils {
    fun imshow(txt: String, img: Mat) {
        val canvasFrame = CanvasFrame(txt)
        canvasFrame.defaultCloseOperation = WindowConstants.EXIT_ON_CLOSE
        canvasFrame.setCanvasSize(img.cols(), img.rows())
        canvasFrame.showImage(OpenCVFrameConverter.ToMat().convert(img))
    }

    fun imreadGray(imagePath: String): Mat {
        val img = imread(imagePath)
        cvtColor(img, img, COLOR_BGR2GRAY)
        return img
    }

    fun getAllFilesInDirectory(filesDir: String): Array<File>? {
        val root = File(filesDir)

        val imageFilter = FilenameFilter { _, name ->
            name.toLowerCase().endsWith(".jpg") || name.endsWith(".png") || name.endsWith(".pgm")
        }

        return root.listFiles(imageFilter)
    }

    fun list2MatVector(list: List<Mat>): MatVector? {
        if (list.isEmpty())
            return null

        val matVector = MatVector(list.size.toLong())
        for ((counter, mat) in list.withIndex()) {
            matVector.put(counter.toLong(), mat)
        }
        return matVector
    }

    fun imresize(src: Mat): Mat {
        val dest = Mat()
        resize(src, dest, opencv_core.Size(160, 160))
        return dest
    }

    fun imsave(path: String, image: Mat) {
        imwrite(path, image)
    }
}
