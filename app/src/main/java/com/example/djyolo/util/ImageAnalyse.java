package com.example.djyolo.util;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.Log;


import com.example.djyolo.detector.tflite.ITFliteDetector;
import com.example.djyolo.detector.tflite.Recognition;
import com.example.djyolo.detector.tflite.Yolov11TFLiteDetector;
import com.example.djyolo.detector.tflite.Yolov5TFLiteDetector;

import java.util.ArrayList;

/**
 * @Author：
 * @Description:
 * @CreateDate：
 */
public class ImageAnalyse {
    private static final String TAG = "ImageAnalyse";
    private int previewHeight = 0;
    private int previewWidth = 0;
    private int rotation = 0;
    ImageProcess imageProcess;

    private ITFliteDetector yolov5TFLiteDetector;
    Matrix modelToPreviewTransform = null;

    public ImageAnalyse(int previewHeight, int previewWidth, int rotation) {
        this.previewHeight = previewHeight;
        this.previewWidth = previewWidth;
        this.rotation = rotation;
        imageProcess = new ImageProcess();

    }

    public void initModel(String modelName, Context context) {
        // 加载模型
        try {
            if(modelName.startsWith("yolov5")){
                this.yolov5TFLiteDetector = new Yolov5TFLiteDetector();
            }else{
                this.yolov5TFLiteDetector = new Yolov11TFLiteDetector();
            }
            this.yolov5TFLiteDetector.setModelFile(modelName);
//            this.yolov5TFLiteDetector.addNNApiDelegate();
            this.yolov5TFLiteDetector.addGPUDelegate();
            this.yolov5TFLiteDetector.initialModel(context);
            Log.i("model", "Success loading model" + this.yolov5TFLiteDetector.getModelFile());
        } catch (Exception e) {
            Log.e("image", "load model error: " + e.getMessage() + e.toString());
        }
    }

    public Matrix getPreviewTransform2(int prevWidth,int prevHeight){
        Matrix previewToModelTransform =
                imageProcess.getTransformationMatrix(
                        prevWidth, prevHeight,
                        yolov5TFLiteDetector.getInputSize().getWidth(),
                        yolov5TFLiteDetector.getInputSize().getHeight(),
                        0, false);
//        if(modelToPreviewTransform == null){
            modelToPreviewTransform = new Matrix();
            previewToModelTransform.invert(modelToPreviewTransform);
//        }
        return modelToPreviewTransform;
    }

    public ArrayList<Recognition>  analyze(Bitmap imageBitmap, int imagewWidth, int imageHeight){
        long start = System.currentTimeMillis();
        // 图片适应屏幕fill_start格式的bitmap
        double scale = Math.max(
                previewHeight / (double) (rotation % 180 == 0 ? imagewWidth : imageHeight),
                previewWidth / (double) (rotation % 180 == 0 ? imageHeight : imagewWidth)
        );
        Matrix fullScreenTransform = imageProcess.getTransformationMatrix(
                imagewWidth, imageHeight,
                (int) (scale * imageHeight), (int) (scale * imagewWidth),
                rotation % 180 == 0 ? 90 : 0, false
        );
        //1
        // 适应preview的全尺寸bitmap
        Bitmap fullImageBitmap = Bitmap.createBitmap(imageBitmap, 0, 0, imagewWidth, imageHeight, fullScreenTransform, false);
//        Log.d(TAG, "analyze: fullImgSize width="+fullImageBitmap.getWidth()+" height="+fullImageBitmap.getHeight()+" scale="+scale);
        // 裁剪出跟preview在屏幕上一样大小的bitmap
        Bitmap cropImageBitmap = Bitmap.createBitmap(fullImageBitmap, 0, 0, previewWidth, previewHeight);
        //2、缩放成模型输入的bitmap
        // 模型输入的bitmap
        Matrix previewToModelTransform =
                imageProcess.getTransformationMatrix(
                        cropImageBitmap.getWidth(), cropImageBitmap.getHeight(),
                        yolov5TFLiteDetector.getInputSize().getWidth(),
                        yolov5TFLiteDetector.getInputSize().getHeight(),
                        0, false);
        Bitmap modelInputBitmap = Bitmap.createBitmap(cropImageBitmap, 0, 0,
                cropImageBitmap.getWidth(), cropImageBitmap.getHeight(),
                previewToModelTransform, false);
        long startDec = System.currentTimeMillis();
        ArrayList<Recognition> recognitions = yolov5TFLiteDetector.detect(modelInputBitmap);
        long detecEnd =  System.currentTimeMillis();
        long end = System.currentTimeMillis();
        long costTime = (end - start);
        long dectTime = (detecEnd - startDec);
        Log.d(TAG, "analyze: costTime="+costTime+" detectTime="+dectTime+" size="+recognitions.size());
        // Recycle intermediate bitmaps to prevent memory leak
        if (fullImageBitmap != null && !fullImageBitmap.isRecycled()) {
            fullImageBitmap.recycle();
        }
        if (modelInputBitmap != null && !modelInputBitmap.isRecycled()) {
            modelInputBitmap.recycle();
        }
        if(cropImageBitmap != null && !cropImageBitmap.isRecycled()){
            cropImageBitmap.recycle();
        }
        return recognitions;
    }

}
