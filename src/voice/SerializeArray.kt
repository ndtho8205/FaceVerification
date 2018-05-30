package voice

import edu.bk.thesis.biodiary.core.voice.math.vq.Codebook
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.ObjectInputStream
import java.io.ObjectOutputStream


object SerializeArray
{

    fun saveArray(arr: List<Codebook>, filepath: String)
    {
        val os = ObjectOutputStream(FileOutputStream(filepath))
        os.writeObject(arr)
        os.close()
    }

    fun loadArray(filepath: String): List<Codebook>
    {
        val oin = ObjectInputStream(FileInputStream(filepath))
        val arr = oin.readObject() as List<Codebook>
        oin.close()
        return arr
    }
}
