package com.example.myapplication

import android.content.Context
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import com.google.android.material.textfield.TextInputEditText
import org.pytorch.*
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.math.RoundingMode
import java.text.DecimalFormat
import kotlin.math.sqrt

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        setPredictionBtnListener()
    }

    private fun setPredictionBtnListener() {
        val predictionBtn = findViewById<Button>(R.id.predictBtn)
        predictionBtn.setOnClickListener {
            val iVal: Float
            val jVal: Float
            val kVal: Float

            val iInput = findViewById<TextInputEditText>(R.id.iInput)
            try {
                iVal = iInput.text.toString().toFloat()
            } catch (e: Exception) {
                Toast.makeText(this, "Set a number input for i", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            val jInput = findViewById<TextInputEditText>(R.id.jInput)
            try {
                jVal = jInput.text.toString().toFloat()
            } catch (e: Exception) {
                Toast.makeText(this, "Set a number input for j", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            val kInput = findViewById<TextInputEditText>(R.id.kInput)
            try {
                kVal = kInput.text.toString().toFloat()
            } catch (e: Exception) {
                Toast.makeText(this, "Set a number input for j", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            val prediction = getPrediction(iVal, jVal, kVal)
            val predictionText = findViewById<TextView>(R.id.predictionText)
            predictionText.text = prediction.toString()

            val trueResultText = findViewById<TextView>(R.id.trueResultText)
            trueResultText.text = getTrueResult(iVal, jVal, kVal).toString()
        }
    }

    private fun getPrediction(i: Float, j: Float, k: Float): Float? {
        val filePath = assetFilePath(this, "model.ptl")
        try {
            val module = LiteModuleLoader.load(filePath)

            // Actual input
            val contInput : FloatArray = floatArrayOf(i, j, k)
            val contInputShape : LongArray = longArrayOf(1, 3)
            val contInputTensor = Tensor.fromBlob(contInput, contInputShape)
            val contIValue = IValue.from(contInputTensor)

            // Unused categorical label input- blank
            val catInput : FloatArray = floatArrayOf()
            val catInputShape : LongArray = longArrayOf(0)
            val catInputTensor = Tensor.fromBlob(catInput, catInputShape)
            val catIValue = IValue.from(catInputTensor)

            val predictionTensor = module.forward(catIValue, contIValue).toTensor()
            val prediction = predictionTensor.dataAsFloatArray[0]
            return roundOffDecimal(prediction)
        } catch (e: Exception) {
            Log.e("MLTabular", "Prediction Error", e)
            Toast.makeText(this, "Prediction error", Toast.LENGTH_SHORT).show()
            return null
        }
    }

    private fun getTrueResult(i: Float, j: Float, k: Float): Float? {
        return roundOffDecimal(i*i + 3*j + sqrt(k))
    }

    private fun roundOffDecimal(number: Float): Float? {
        val df = DecimalFormat("#.##")
        df.roundingMode = RoundingMode.CEILING
        return df.format(number).toFloat()
    }

    private fun assetFilePath(context: Context, asset: String): String {
        val file = File(context.filesDir, asset)

        try {
            val inpStream: InputStream = context.assets.open(asset)
            try {
                val outStream = FileOutputStream(file, false)
                val buffer = ByteArray(4 * 1024)
                var read: Int

                while (true) {
                    read = inpStream.read(buffer)
                    if (read == -1) {
                        break
                    }
                    outStream.write(buffer, 0, read)
                }
                outStream.flush()
            } catch (ex: Exception) {
                ex.printStackTrace()
            }
            return file.absolutePath
        } catch (e: Exception) {
            e.printStackTrace()
        }
        return ""
    }
}