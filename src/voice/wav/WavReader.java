package voice.wav;

import java.io.IOException;
import java.io.RandomAccessFile;


public class WavReader extends Wav
{

    private static final String TAG = WavReader.class.getSimpleName();

    private long mCurrentPosition;

    public WavReader(String filePath)
    {
        init(filePath);
    }

    @Override
    public int read(byte[] buffer, int offset, int count)
    {
        int read = 0;
        try {
            read = mWavFile.read(buffer, 0, count);
            mCurrentPosition += read;
            mWavFile.seek(mCurrentPosition);
            return 0;
        }
        catch (IOException e) {
            e.printStackTrace();
            return -1;
        }
    }

    public void open(String filePath)
    {
        init(filePath);
    }

    public void close()
    {
        try {
            mWavFile.close();
        }
        catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void init(String filePath)
    {
        try {
            mWavFile = new RandomAccessFile(filePath, "r");

            // read file size and payload size (data) fields
            mWavFile.seek(4);
            mFileSize = Integer.reverseBytes(mWavFile.readInt());
            mWavFile.seek(40);
            mPayloadSize = Integer.reverseBytes(mWavFile.readInt());

            // get other metadata
            mWavFile.seek(22);
            mNumChannels = Short.reverseBytes(mWavFile.readShort());
            mSampleRate = Integer.reverseBytes(mWavFile.readInt());
            mByteRate = Integer.reverseBytes(mWavFile.readInt());
            mBlockAlign = Short.reverseBytes(mWavFile.readShort());
            mBitsPerSample = Short.reverseBytes(mWavFile.readShort());

            // set at start of data part
            mCurrentPosition = 44;
            mWavFile.seek(mCurrentPosition);
        }
        catch (IOException e) {
            e.printStackTrace();
        }
    }
}
