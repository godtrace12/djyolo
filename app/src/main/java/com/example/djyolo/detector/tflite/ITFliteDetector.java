package com.example.djyolo.detector.tflite;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Size;

import java.util.ArrayList;

/**
 * @Author：
 * @Description: tflite检测器接口
 * @CreateDate：
 */
public interface ITFliteDetector {
    public void initialModel(Context activity);

    public void setModelFile(String modelFile);
    public Size getInputSize();

    public String getModelFile();

    public void addGPUDelegate();
    public void addNNApiDelegate();
    public ArrayList<Recognition> detect(Bitmap bitmap);


}
