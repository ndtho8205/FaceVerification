import face.FaceVerification

fun main(args: Array<String>) {
    val trainedResult = args[0]
    val testDir = args[1]

    val testFaces = getAllFaceImagesFromDir(testDir)

    val verifier = FaceVerification()
    verifier.load(trainedResult)
    verifier.predict(testFaces)
}

