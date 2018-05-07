import face.Detection
import face.Face
import face.Preprocessing
import utils.JavaCvUtils

fun main(args: Array<String>) {
    val inputDir = args[0]
    val outputDir = args[1]

    val imageFiles = JavaCvUtils.getAllFilesInDirectory(inputDir)
    val faces = ArrayList<Face>()

    imageFiles?.forEach {
        val image = JavaCvUtils.imreadGray(it.absolutePath)
        val faceImagesDetected = Detection.detect(image)
        if (faceImagesDetected != null) {
            var preprocessedFaceImage = Preprocessing.scaleToStandardSize(faceImagesDetected)
            faces.add(Face(preprocessedFaceImage, it.name))
        }
    }

    faces.forEach {
        it.save("$outputDir/aligned_${it.containerImageName}")
    }
}

