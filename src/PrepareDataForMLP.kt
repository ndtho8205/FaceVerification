import edu.bk.thesis.biodiary.core.voice.math.vq.Codebook
import face.Face
import face.FaceQualityComputation
import face.FaceVerification
import face.JavaCvUtils
import models.FaceData
import voice.SerializeArray
import voice.VoiceAuthenticator
import java.nio.file.Files
import java.nio.file.Paths
import java.util.*

val MODEL_DIR = "/home/ndtho8205/Desktop/BioDiaryData/train"

/*
    .../train/
        |- classname_1/
            |- label_1/
                |- Face/
                |- Voice/
            |- label_2/
            |- codebook.yml
            |- model_face.yml
        |- classname_2/
 */

fun main(args: Array<String>)
{
    val faceVerifier = FaceVerification()
    val voiceVerifier = VoiceAuthenticator()

    val path = Paths.get(MODEL_DIR)
    if (Files.isDirectory(path))
    {
        val directories = Files.list(path).filter { Files.isDirectory(it) }
        directories.forEach {
            println(it.fileName)

            val classname = it.fileName.toString().split("_")[0]

            val classFaceModelPath = Paths.get(it.toString(), "model_face.yml").toString()
            faceVerifier.loadTrainedModel(classFaceModelPath)
            println("\tLoad face verification model: $classFaceModelPath")

            val classVoiceModelPath = Paths.get(it.toString(), "codebook.yml").toString()
            voiceVerifier.codeBook = SerializeArray.loadArray(classVoiceModelPath) as ArrayList<Codebook>
            println("\tLoad voice verification model: $classVoiceModelPath")

            val rawLabels = Files.list(it).filter { Files.isDirectory(it) }
            rawLabels.forEach {
                println("  ${it.fileName}")

                val label = it.fileName.toString().split("_")[0]

                val labelFaceDirectoryPath = Paths.get(it.toString(), "Face").toString()
                processFaceDirectory(labelFaceDirectoryPath, faceVerifier, if (label == classname) 1 else 0)

                val labelVoiceirectoryPath = Paths.get(it.toString(), "Voice").toString()
                processVoiceDirectory(labelVoiceirectoryPath, voiceVerifier, if (label == classname) 1 else 0)

            }
        }
    }

}

fun processFaceDirectory(path: String, verifier: FaceVerification, label: Int)
{
    val qualityComputation = FaceQualityComputation()

    val imageFiles = JavaCvUtils.getAllImageFilesInDirectory(path)

    val containerImageFiles = imageFiles.filter { it.nameWithoutExtension.endsWith("_container") }
    val faceImageFiles = imageFiles.filter { it.nameWithoutExtension.endsWith("_face") }

    val faceData = (containerImageFiles zip faceImageFiles).map {
        val face = Face(JavaCvUtils.imreadRgb(it.first.absolutePath),
                        it.first.absolutePath,
                        JavaCvUtils.imreadGray(it.second.absolutePath))

        val distance = verifier.predict(face)
        val qualityBrightness = qualityComputation.computeBrightnessScore(face.containerImage)
        val qualityContrast = qualityComputation.computeContrastScore(face.faceImage)
        val qualitySharpness = qualityComputation.computeSharpnessScore(face.faceImage)

        val data = FaceData(label,
                            distance,
                            qualityBrightness,
                            qualityContrast,
                            qualitySharpness)
        println("\t  ${it.first.nameWithoutExtension} - ${it.second.nameWithoutExtension}: $data")

        data
    }
}

fun processVoiceDirectory(path: String, verifier: VoiceAuthenticator, label: Int)
{
    val audioFiles = JavaCvUtils.getAllAudioFilesInDirectory(path)

    audioFiles.forEach {
        verifier.readWav(it.absolutePath)
        val featureVector = verifier.currentFeatureVector
        val distance = verifier.identifySpeaker(featureVector)
        println("\t  ${it.name}: $distance")
    }

//    val containerImageFiles = imageFiles.filter { it.nameWithoutExtension.endsWith("_container") }
//    val faceImageFiles = imageFiles.filter { it.nameWithoutExtension.endsWith("_face") }
//
//    val faceData = (containerImageFiles zip faceImageFiles).map {
//        val face = Face(JavaCvUtils.imreadRgb(it.first.absolutePath),
//                        it.first.absolutePath,
//                        JavaCvUtils.imreadGray(it.second.absolutePath))
//
//        val distance = verifier.predict(face)
//        val qualityBrightness = qualityComputation.computeBrightnessScore(face.containerImage)
//        val qualityContrast = qualityComputation.computeContrastScore(face.faceImage)
//        val qualitySharpness = qualityComputation.computeSharpnessScore(face.faceImage)
//
//        val data = FaceData(label,
//                            distance,
//                            qualityBrightness,
//                            qualityContrast,
//                            qualitySharpness)
//        println("\t  ${it.first.nameWithoutExtension} - ${it.second.nameWithoutExtension}: $data")
//
//        data
//    }
}


