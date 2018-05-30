package models

data class FaceData(val label: Int,
                    val distance: Double,
                    val qualityBrightness: Double,
                    val qualityContrast: Double,
                    val qualitySharpness: Double)
{

    override fun toString(): String
    {
        return "$label;$distance;$qualityBrightness;$qualityContrast;$qualitySharpness"
    }

}
