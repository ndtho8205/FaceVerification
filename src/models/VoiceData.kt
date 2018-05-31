package models

data class VoiceData(val label: Int,
                     val distance: Double,
                     val qualityEnvAmplitude: Double)
{

    override fun toString(): String
    {
        return "$label;$distance;$qualityEnvAmplitude"
    }

}
