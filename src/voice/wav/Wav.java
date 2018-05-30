package voice.wav;

import java.io.IOException;
import java.io.RandomAccessFile;


public abstract class Wav
{

    // http://www.topherlee.com/software/pcm-tut-wavformat.html
    // http://soundfile.sapp.org/doc/WaveFormat/

    int   mFileSize;
    short mNumChannels; //
    int   mSampleRate; //
    int   mByteRate;
    int   mBlockAlign; //
    short mBitsPerSample; //
    int   mPayloadSize;

    RandomAccessFile mWavFile;
    String           mWavFilePath;

    public int getFileSize()
    {
        return mFileSize;
    }

    public short getNumChannels()
    {
        return mNumChannels;
    }

    public int getSampleRate()
    {
        return mSampleRate;
    }

    public int getByteRate()
    {
        return mByteRate;
    }

    public int getBlockAlign()
    {
        return mBlockAlign;
    }

    public short getBitsPerSample()
    {
        return mBitsPerSample;
    }

    public int getPayloadSize()
    {
        return mPayloadSize;
    }

    public abstract int read(byte[] buffer, int offset, int count) throws IOException;
}
