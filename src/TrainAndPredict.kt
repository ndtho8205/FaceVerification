import face.Face
import face.FaceVerification
import face.JavaCvUtils

fun main(args: Array<String>) {
    val trainingDir = args[0]
    val testDir = args[1]

    val trainingFaces = getAllFaceImagesFromDir(trainingDir)
    val testFaces = getAllFaceImagesFromDir(testDir)

    val verifier = FaceVerification()
    verifier.train(trainingFaces)
    verifier.save("/home/ndtho/Desktop/eigenfaces_trained.yml")
    verifier.predict(testFaces)
}

fun getAllFaceImagesFromDir(imageDir: String): List<Face> {
    val faceImageFiles = JavaCvUtils.getAllImageFilesInDirectory(imageDir)
    val faces = ArrayList<Face>()

    faceImageFiles?.forEach {
        val faceImage = JavaCvUtils.imreadGray(it.absolutePath)
        val face = Face(faceImage, Face.BoundingBox(0,0,0,0), it.name)
        faces.add(face)
    }

    return faces
}
