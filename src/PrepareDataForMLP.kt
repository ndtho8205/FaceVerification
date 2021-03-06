import edu.bk.thesis.biodiary.core.voice.math.vq.Codebook
import face.*
import models.FaceData
import models.VoiceData
import voice.SerializeArray
import voice.VoiceAuthenticator
import java.io.File
import java.nio.file.Files
import java.nio.file.Paths
import java.util.*

private val DIR = "/home/ndtho/Desktop/data/train"
private val OUTPUT_PATH = "$DIR/train_data.txt"



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

    val path = Paths.get(DIR)
    if (Files.isDirectory(path))
    {
        val trainingDataFile = File(OUTPUT_PATH).printWriter()

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
                val faceData =
                        processFaceDirectory(labelFaceDirectoryPath, faceVerifier, if (label == classname) 1 else 0)

                val labelVoiceDirectoryPath = Paths.get(it.toString(), "Voice").toString()
                val voiceData =
                        processVoiceDirectory(labelVoiceDirectoryPath, voiceVerifier, if (label == classname) 1 else 0)

                for (i in faceData.indices)
                    for (j in voiceData.indices)
                    {
                        val line = faceData[i].toString() + voiceData[j].toString().drop(1)
                        trainingDataFile.println(line)
                    }

            }
        }
        trainingDataFile.close()
    }
}

private fun processFaceDirectory(path: String, verifier: FaceVerification, label: Int): List<FaceData>
{
    val qualityComputation = FaceQualityComputation()

    val imageFiles = JavaCvUtils.getAllImageFilesInDirectory(path)

    val containerImageFiles = imageFiles.filter { it.nameWithoutExtension.endsWith("_container") }
    val faceImageFiles = imageFiles.filter { it.nameWithoutExtension.endsWith("_face") }

    val faceData = (containerImageFiles zip faceImageFiles).map {
        val face = Face(JavaCvUtils.imreadRgb(it.first.absolutePath),
                        it.first.absolutePath,
                        ImagePreprocessing().equalizeHist(JavaCvUtils.imreadGray(it.second.absolutePath)))

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
    return faceData
}

fun processVoiceDirectory(path: String, verifier: VoiceAuthenticator, label: Int): List<VoiceData>
{
    val audioFiles = JavaCvUtils.getAllAudioFilesInDirectory(path)

    val voiceData = audioFiles.map {
        verifier.readWav(it.absolutePath)
        val featureVector = verifier.currentFeatureVector
        val distance = verifier.identifySpeaker(featureVector).toDouble()

        val qualityEnvAmplitude = it.nameWithoutExtension.split("_")[1].toDouble()

        val data = VoiceData(label, distance, qualityEnvAmplitude)

        println("\t  ${it.nameWithoutExtension}: $data")

        data
    }
    return voiceData
}


