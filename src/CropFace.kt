import face.FaceDetection
import face.FacePreprocessing
import face.JavaCvUtils

fun main(args: Array<String>)
{
    val inputDir = args[0]
    val outputDir = args[1]

    val imageFiles = JavaCvUtils.getAllImageFilesInDirectory(inputDir)

    imageFiles?.forEach {
        val image = JavaCvUtils.imreadGray(it.absolutePath)
        val faceImageDetected = FaceDetection.detect(image)
        if (faceImageDetected != null)
        {
            FacePreprocessing.scaleToStandardSize(faceImageDetected)

            faceImageDetected.save("$outputDir/aligned_${faceImageDetected.containerImageName}")
        }
    }

}

