/*
 * Copyright 2021 Shubham Panchal
 * Licensed under the Apache License, Version 2.0 (the "License");
 * You may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.shubham0204.ml.depthestimation

import android.content.Context
import android.graphics.Bitmap
import android.os.Build
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.TensorOperator
import org.tensorflow.lite.support.common.TensorProcessor
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import org.tensorflow.lite.support.tensorbuffer.TensorBufferFloat
import org.tensorflow.lite.support.common.ops.CastOp
import org.tensorflow.lite.support.tensorbuffer.TensorBufferUint8
import java.nio.ByteBuffer

// Helper class for the MiDAS TFlite model
class MiDASModel( context: Context ) {

    // See the `app/src/main/assets` folder for the TFLite model
    private val modelFileName = "quantized_depth_model.tflite"
    private var interpreter : Interpreter
    private val NUM_THREADS = 4

    // These values are taken from the Python file ->
    // https://github.com/isl-org/MiDaS/blob/master/mobile/android/models/src/main/assets/run_tflite.py
    private val inputImageDim = 256

    // Modified input tensor processor for quantized model
    private val inputTensorProcessor = ImageProcessor.Builder()
        .add(ResizeOp(inputImageDim, inputImageDim, ResizeOp.ResizeMethod.BILINEAR))
        .add(CastOp(DataType.UINT8))
        .build()

    init {
        val interpreterOptions = Interpreter.Options().apply {
            if (CompatibilityList().isDelegateSupportedOnThisDevice) {
                Logger.logInfo("GPU Delegate is supported on this device.")
                addDelegate(GpuDelegate(CompatibilityList().bestOptionsForThisDevice))
            } else {
                setNumThreads(NUM_THREADS)
            }
        }
        interpreter = Interpreter(FileUtil.loadMappedFile(context, modelFileName), interpreterOptions)
        Logger.logInfo("TFLite interpreter created.")
    }



    fun getDepthMap(inputImage: Bitmap): Bitmap {
        return run(inputImage)
    }

    private fun run(inputImage: Bitmap): Bitmap {
        var inputTensor = TensorImage.fromBitmap(inputImage)

        val t1 = System.currentTimeMillis()
        inputTensor = inputTensorProcessor.process(inputTensor)

        Logger.logInfo("Input tensor shape: ${inputTensor.tensorBuffer.shape.contentToString()}")
        Logger.logInfo("Input tensor buffer size: ${inputTensor.tensorBuffer.buffer.capacity()} bytes")

        val outputTensor = TensorBufferUint8.createFixedSize(
            intArrayOf(inputImageDim, inputImageDim, 1), DataType.UINT8)

        Logger.logInfo("Output tensor shape: ${outputTensor.shape.contentToString()}")
        Logger.logInfo("Output tensor buffer size: ${outputTensor.buffer.capacity()} bytes")

        try {
            Logger.logInfo("Starting interpreter.run")
            interpreter.run(inputTensor.buffer, outputTensor.buffer)
            Logger.logInfo("Finished interpreter.run")
        } catch (e: IllegalArgumentException) {
            Logger.logError("Interpreter run failed: ${e.message}")
            e.printStackTrace()
            return Bitmap.createBitmap(inputImageDim, inputImageDim, Bitmap.Config.ARGB_8888)
        }

        Logger.logInfo("MiDaS inference speed: ${System.currentTimeMillis() - t1}")

        // Convert UINT8 to float for bitmap creation
        val floatArray = outputTensor.intArray.map { it.toFloat() }.toFloatArray()
        return BitmapUtils.byteBufferToBitmap(floatArray, inputImageDim)
    }


    // Post processing operation for MiDAS
    // Apply min-max scaling to the outputs of the model and bring them in the range [ 0 , 255 ].
    // Also, we apply a transformation which changes the data type from `int` to `uint` in Python.
    // As unsigned integers aren't supported in Java, we add 255 + pixel if pixel < 0
    class MinMaxScalingOp : TensorOperator {

        override fun apply( input : TensorBuffer?): TensorBuffer {
            val values = input!!.floatArray
            // Compute min and max of the output
            val max = values.maxOrNull()!!
            val min = values.minOrNull()!!
            for ( i in values.indices ) {
                // Normalize the values and scale them by a factor of 255
                var p = ((( values[ i ] - min ) / ( max - min )) * 255).toInt()
                if ( p < 0 ) {
                    p += 255
                }
                values[ i ] = p.toFloat()
            }
            // Convert the normalized values to the TensorBuffer and load the values in it.
            val output = TensorBufferFloat.createFixedSize( input.shape , DataType.FLOAT32 )
            output.loadArray( values )
            return output
        }

    }



}