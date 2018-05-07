import face.Verification

fun main(args: Array<String>) {
    val trainedResult = args[0]
    val testDir = args[1]

    val testFaces = getAllFaceImagesFromDir(testDir)

    val verifier = Verification()
    verifier.load(trainedResult)
    verifier.predict(testFaces)
}

