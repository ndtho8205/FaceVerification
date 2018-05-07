import face.Detection
import face.Face
import face.Preprocessing
import utils.JavaCvUtils

fun main(args: Array<String>) {
    val inputDir = args[0]
    val outputDir = args[1]

    val imageFiles = JavaCvUtils.getAllFilesInDirectory(inputDir)

    imageFiles?.forEach {
        val image = JavaCvUtils.imreadGray(it.absolutePath)
        val faceImageDetected = Detection.detect(image)
        if (faceImageDetected != null) {
            val scaledFaceImage = Preprocessing.scaleToStandardSize(faceImageDetected)

            val face = Face(scaledFaceImage, it.name)
            face.save("$outputDir/aligned_${face.containerImageName}")
        }
    }

}

