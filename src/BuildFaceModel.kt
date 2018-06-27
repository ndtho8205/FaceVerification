import face.Face
import face.FaceVerification
import face.ImagePreprocessing
import face.JavaCvUtils
import java.nio.file.Files
import java.nio.file.Paths

private val DIR = "/home/ndtho/Desktop/data/Day 3"
private val MODEL_NAME = "model_face.yml"

fun main(args: Array<String>)
{
    val faceVerifier = FaceVerification()

    val path = Paths.get(DIR)
    if (Files.isDirectory(path))
    {
        val directories = Files.list(path).filter { Files.isDirectory(it) }
        directories.forEach {
            println(it.fileName)

            val faceDirectoryPath = Paths.get(it.toString(), "Face").toString()
            val modelPath = it.toString()
            println(modelPath)

            processFaceDirectory(faceDirectoryPath, modelPath, faceVerifier)
        }
    }
}

private fun processFaceDirectory(facePath: String, modelPath: String, verifier: FaceVerification)
{
    val imageFiles = JavaCvUtils.getAllImageFilesInDirectory(facePath)

    val containerImageFiles = imageFiles.filter { it.nameWithoutExtension.endsWith("_container") }
    val faceImageFiles = imageFiles.filter { it.nameWithoutExtension.endsWith("_face") }

    val faceImages = faceImageFiles.map {
        var img = JavaCvUtils.imreadGray(it.absolutePath)
        img = ImagePreprocessing().equalizeHist(img)
        Face(img, "img", img)
    }

    verifier.train(faceImages)
    verifier.saveTrainedModel(modelPath)

    val testImage =
            ImagePreprocessing().equalizeHist(JavaCvUtils.imreadGray("/home/ndtho/Desktop/data/Day 1/Hung_1/Face/0_face.png"))
    println(verifier.predict(Face(testImage, "test", testImage)))
}
