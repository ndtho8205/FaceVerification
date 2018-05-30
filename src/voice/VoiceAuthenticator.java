package voice;

import edu.bk.thesis.biodiary.core.voice.math.matrix.Matrix;
import edu.bk.thesis.biodiary.core.voice.math.mfcc.FeatureVector;
import edu.bk.thesis.biodiary.core.voice.math.mfcc.MFCC;
import edu.bk.thesis.biodiary.core.voice.math.vq.ClusterUtil;
import edu.bk.thesis.biodiary.core.voice.math.vq.Codebook;
import edu.bk.thesis.biodiary.core.voice.math.vq.KMeans;
import voice.wav.WavReader;

import java.util.ArrayList;


public class VoiceAuthenticator
{

    private static final String LOG_TAG        = "VoiceAuthenticator";
    private static final int    calibrate_time = 5000;

    private WavReader           mVoiceRecorder;
    private ArrayList<Codebook> codeBook;

    public VoiceAuthenticator()
    {
        codeBook = new ArrayList<>();
    }

    public void readWav(String wavPath)
    {
        mVoiceRecorder = new WavReader(wavPath);
    }

    public ArrayList<Codebook> getCodeBook()
    {
        return new ArrayList<>(codeBook);
    }

    public void setCodeBook(ArrayList<Codebook> cb)
    {
        codeBook = new ArrayList<>(cb);
    }

    public FeatureVector getCurrentFeatureVector()
    {
        double[] samples = readSamplesFromBuffer();

        double[][] mfcc = calculateMfcc(samples);

        FeatureVector result = createFeatureVector(mfcc);

        return result;
    }

    public float identifySpeaker(FeatureVector featureVector)
    {
        float result = -1f;
        if (featureVector != null && codeBook.size() > 0) {
            result = 0f;

            for (int i = 0; i < codeBook.size(); i++) {
                double averageDistortion = ClusterUtil.calculateAverageDistortion(featureVector,
                                                                                  codeBook.get(i));

                result += averageDistortion;

            }

            result = result / codeBook.size();
        }
        else {
        }
        return result;
    }

    public FeatureVector computeFeature(FeatureVector featureVector)
    {
        double[] samples = readSamplesFromBuffer();

        if (samples.length < 1) {
            return null;
        }

        double[][] mfcc = calculateMfcc(samples);


        if (featureVector == null) {
            featureVector = createFeatureVector(mfcc);
        }
        else {
            featureVector = mergeFeatureVector(featureVector, mfcc);
        }

        return featureVector;
    }

    public boolean train(FeatureVector featureVector)
    {
        if (featureVector != null) {
            KMeans kmeans = doClustering(featureVector);

            Codebook cb = createCodebook(kmeans);

            insertFeature(cb);

            return true;
        }
        else {
            return false;
        }
    }

    private FeatureVector createFeatureVector(double[][] mfcc)
    {
        int vectorSize  = mfcc[0].length;
        int vectorCount = mfcc.length;
        FeatureVector pl = new FeatureVector(vectorSize, vectorCount);
        for (int i = 0; i < vectorCount; i++) {
            pl.add(mfcc[i]);
        }
        return pl;
    }

    private FeatureVector mergeFeatureVector(FeatureVector featureVector, double[][] mfcc)
    {
        int vectorCount = mfcc.length;
        for (int i = 0; i < vectorCount; i++) {
            featureVector.add(mfcc[i]);
        }
        return featureVector;
    }

    private short createSample(byte[] buffer)
    {
        short sample = 0;
        // hardcoded two bytes here
        short b1 = buffer[0];
        short b2 = buffer[1];
        b2 <<= 8;
        sample = (short) (b1 | b2);
        return sample;
    }

    private double[][] calculateMfcc(double[] samples)
    {
        MFCC mfccCalculator = new MFCC(Constants.SAMPLERATE,
                                       Constants.WINDOWSIZE,
                                       Constants.COEFFICIENTS,
                                       false,
                                       Constants.MINFREQ + 1,
                                       Constants.MAXFREQ,
                                       Constants.FILTERS);

        int        hopSize   = Constants.WINDOWSIZE / 2;
        int        mfccCount = (samples.length / hopSize) - 1;
        double[][] mfcc      = new double[mfccCount][Constants.COEFFICIENTS];
        long       start     = System.currentTimeMillis();
        for (int i = 0, pos = 0; pos < samples.length - hopSize; i++, pos += hopSize) {
            mfcc[i] = mfccCalculator.processWindow(samples, pos);
            if (i % 50 == 0) {
            }
        }

        return mfcc;
    }

    private double[] readSamplesFromBuffer()
    {
        int sampleSize = mVoiceRecorder.getBlockAlign();

//        int sampleCount = mVoiceRecorder.getPayloadSize() / sampleSize;
        int sampleCount = mVoiceRecorder.getPayloadSize() / sampleSize;

        int windowCount = (int) Math.floor(sampleCount / Constants.WINDOWSIZE);

        double[] samples    = new double[windowCount * Constants.WINDOWSIZE];
        byte[]   buffer     = new byte[sampleSize];
        int      currentPos = 0;

        for (int i = 0; i < samples.length; i++) {
            currentPos = mVoiceRecorder.read(buffer, currentPos, sampleSize);
            samples[i] = createSample(buffer);
        }

        return samples;
    }

    private void insertFeature(Codebook cb)
    {
        // Save password in list
        codeBook.add(cb);
    }

    private Codebook createCodebook(KMeans kmeans)
    {
        int      numberClusters = kmeans.getNumberClusters();
        Matrix[] centers        = new Matrix[numberClusters];
        for (int i = 0; i < numberClusters; i++) {
            centers[i] = kmeans.getCluster(i).getCenter();
        }
        Codebook cb = new Codebook();
        cb.setLength(numberClusters);
        cb.setCentroids(centers);
        return cb;
    }

    private KMeans doClustering(FeatureVector pl)
    {
        long   start;
        KMeans kmeans = new KMeans(Constants.CLUSTER_COUNT, pl, Constants.CLUSTER_MAX_ITERATIONS);
        start = System.currentTimeMillis();

        kmeans.run();
        return kmeans;
    }
}
