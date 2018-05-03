import face.Face
import face.Verification
import utils.JavaCvUtils

fun main(args: Array<String>) {
    val trainingDir = args[0]
    val testDir = args[1]

    val trainingFaces = getAllFaceImagesFromDir(trainingDir)
    val testFaces = getAllFaceImagesFromDir(testDir)

    val verifier = Verification()
    verifier.train(trainingFaces)
    verifier.predict(testFaces)
}

fun getAllFaceImagesFromDir(imageDir: String): List<Face> {
    val faceImageFiles = JavaCvUtils.getAllFilesInDirectory(imageDir)
    val faces = ArrayList<Face>()

    faceImageFiles?.forEach {
        val faceImage = JavaCvUtils.imreadGray(it.absolutePath)
        val face = Face(faceImage, it.name)
        faces.add(face)
    }

    return faces
}
