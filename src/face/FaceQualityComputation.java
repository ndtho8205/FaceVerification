package face;

import org.bytedeco.javacpp.indexer.UByteRawIndexer;
import org.bytedeco.javacpp.opencv_core.Mat;


public class FaceQualityComputation
{

    // Calculate brightness of image based on YUV color space
    public double computeBrightnessScore(Mat rgbMat)
    {
        double brightness;

        UByteRawIndexer brightElem = rgbMat.createIndexer();
        double             sum        = 0;
        for (int x = 0; x < rgbMat.rows(); x++) {
            for (int y = 0; y < rgbMat.cols(); y++) {
                int    r = brightElem.get(x, y, 2);
                int    g = brightElem.get(x, y, 1);
                int    b = brightElem.get(x, y, 0);
                double Y = 0.299 * r + 0.587 * g + 0.114 * b;
                sum += Y;
            }
        }
        brightness = sum / (rgbMat.rows() * rgbMat.cols());

        return brightness;
    }

    // Calculate contrast of image using the Root Mean Square (RMS) equation
    public double computeContrastScore(Mat grayMat)
    {
        double contrast;

        UByteRawIndexer contrastElem = grayMat.createIndexer();
        double             mean         = 0;
        for (int x = 0; x < grayMat.rows(); x++) {
            for (int y = 0; y < grayMat.cols(); y++) {
                mean += contrastElem.get(x, y);
            }
        }
        mean /= (grayMat.rows() * grayMat.cols());

        double temp = 0;
        for (int x = 0; x < grayMat.rows(); x++) {
            for (int y = 0; y < grayMat.cols(); y++) {
                temp += Math.pow(contrastElem.get(x, y) - mean, 2);
            }
        }
        contrast = Math.sqrt(temp / (grayMat.rows() * grayMat.cols()));

        return contrast;
    }

    //Calculate sharpness of image
    // Based on research of Kryszczuk and Drygajlo (Paper name: "On combining evidence for reliability estimation in face verification")
    public double computeSharpnessScore(Mat grayMat)
    {
        double sharpness;

        UByteRawIndexer sharpElem = grayMat.createIndexer();

        double sumX = 0;
        for (int x = 0; x < grayMat.rows(); x++) {
            for (int y = 0; y < grayMat.cols() - 1; y++) {
                sumX += Math.abs(sharpElem.get(x, y) - sharpElem.get(x, y + 1));
            }
        }
        sumX /= grayMat.rows() * (grayMat.cols() - 1);

        double sumY = 0;
        for (int x = 0; x < grayMat.rows() - 1; x++) {
            for (int y = 0; y < grayMat.cols(); y++) {
                sumY += Math.abs(sharpElem.get(x, y) - sharpElem.get(x + 1, y));
            }
        }
        sumY /= grayMat.cols() * (grayMat.rows() - 1);

        sharpness = 0.5 * (sumX + sumY);

        return sharpness;
    }
}
