package com.example.djyolo.detector.tflite;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.Build;
import android.util.Log;
import android.util.Size;
import android.widget.Toast;


import com.example.djyolo.util.PostProcessUtils;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.gpu.GpuDelegateFactory;
import org.tensorflow.lite.nnapi.NnApiDelegate;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.common.ops.CastOp;
import org.tensorflow.lite.support.common.ops.DequantizeOp;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.common.ops.QuantizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.metadata.MetadataExtractor;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;


public class Yolov11TFLiteDetector implements ITFliteDetector {
    private static final String TAG = "Yolov11TFLiteDetector";
    private final Size INPNUT_SIZE = new Size(320, 320);
    // yolo11 输出 [1, 84, 2100],yolov5输出是[1, 6300, 85]
    private int[] OUTPUT_SIZE = new int[]{1, 84, 2100}; //[1, 84, 2100] 取值逻辑等要做修改。

    private Boolean IS_INT8 = false;
    private final float DETECT_THRESHOLD = 0.3f;
    private final float IOU_THRESHOLD = 0.45f;
    private final float IOU_CLASS_DUPLICATED_THRESHOLD = 0.7f;
    //替换为自己训练、导出的模型文件
    private final String MODEL_YOLOV5S = "yolo11s_float16.tflite"; //yolov11模型
    private final String MODEL_YOLOV5N =  "yolo11n_float16.tflite";
    private String LABEL_FILE = "coco_label.txt";
    private final int NUM_CLASSES = 80;
    private String MODEL_FILE;
    private Interpreter tflite;
    private List<String> associatedAxisLabels;
    Interpreter.Options options = new Interpreter.Options();

    public String getModelFile() {
        return this.MODEL_FILE;
    }

    @Override
    public void setModelFile(String modelFile){
        switch (modelFile) {
            case "yolov11s":
                IS_INT8 = false;
                MODEL_FILE = MODEL_YOLOV5S;
                Log.d("dj", "setModelFile: "+MODEL_FILE);
                break;
            case "yolov11n":
                IS_INT8 = false;
                MODEL_FILE = MODEL_YOLOV5N;
                break;
            default:
                Log.i("tfliteSupport", "Only yolov5s/n/m/sint8 can be load!");
        }
    }

    public String getLabelFile() {
        return this.LABEL_FILE;
    }

    @Override
    public Size getInputSize(){return this.INPNUT_SIZE;}
    public int[] getOutputSize(){return this.OUTPUT_SIZE;}

    /**
     * 初始化模型, 可以通过 addNNApiDelegate(), addGPUDelegate()提前加载相应代理
     *
     * @param activity
     */
    @Override
    public void initialModel(Context activity) {
        // Initialise the model
        try {
            Log.d(TAG, "initialModel: begin");
            ByteBuffer tfliteModel = FileUtil.loadMappedFile(activity, MODEL_FILE);
            tflite = new Interpreter(tfliteModel, options);
            Log.i("tfliteSupport", "Success reading model: " + MODEL_FILE);
            associatedAxisLabels = FileUtil.loadLabels(activity, LABEL_FILE);
            int numOutputs = tflite.getOutputTensorCount();
            Log.d(TAG, "initialModel: out tensorCount="+numOutputs);
            for(int i=0; i<numOutputs; i++){
                Tensor outputTensor=tflite.getOutputTensor(i);
                int[] outputshapeTmp = outputTensor.shape();
                DataType outptType = outputTensor.dataType();
                Log.d(TAG, "initialModel: out tensor= outType="+outptType+" shape="+Arrays.toString(outputshapeTmp));
            }
            int[] outputShape = tflite.getOutputTensor(0).shape();
            OUTPUT_SIZE = tflite.getOutputTensor(0).shape();
            Log.d("tflite", "detect: "+Arrays.toString(outputShape));

        } catch (IOException e) {
            Log.e("tfliteSupport", "Error reading model or label: ", e);
            Toast.makeText(activity, "load model error: " + e.getMessage(), Toast.LENGTH_LONG).show();
        }
    }


    @Override
    public ArrayList<Recognition> detect(Bitmap bitmap) {

        // yolov5s-tflite的输入是:[1, 320, 320,3], 摄像头每一帧图片需要resize,再归一化
        TensorImage yolov5sTfliteInput;
        ImageProcessor imageProcessor;
        imageProcessor =
                    new ImageProcessor.Builder()
                            .add(new ResizeOp(INPNUT_SIZE.getHeight(), INPNUT_SIZE.getWidth(), ResizeOp.ResizeMethod.BILINEAR))
                            .add(new NormalizeOp(0, 255))
//                            .add(new CastOp(DataType.FLOAT32))
                            .build();
        yolov5sTfliteInput = new TensorImage(DataType.FLOAT32);
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, INPNUT_SIZE.getWidth(), INPNUT_SIZE.getHeight(), false);

        yolov5sTfliteInput.load(resizedBitmap);
        yolov5sTfliteInput = imageProcessor.process(yolov5sTfliteInput);
        // yolov5s-tflite的输出是:[1, 6300, 85], 可以从v5的GitHub release处找到相关tflite模型, 输出是[0,1], 处理到320.
        TensorBuffer probabilityBuffer = TensorBuffer.createFixedSize(OUTPUT_SIZE, DataType.FLOAT32);

        // 推理计算
        if (null != tflite) {
            // 这里tflite默认会加一个batch=1的纬度
            tflite.run(yolov5sTfliteInput.getBuffer(), probabilityBuffer.getBuffer());
        }

        // 输出数据被平铺了出来
        float[] recognitionArray = probabilityBuffer.getFloatArray();
        ArrayList<Recognition> allRecognitions = new ArrayList<>();
        for (int c = 0; c < OUTPUT_SIZE[2]; c++) {
            float maxConf = DETECT_THRESHOLD;
            int maxIdx = -1;
            int j = 4;
            int arrayIdx = c + OUTPUT_SIZE[2] * j;

            while (j < OUTPUT_SIZE[1]) {
                if (recognitionArray[arrayIdx] > maxConf) {
                    maxConf = recognitionArray[arrayIdx];
                    maxIdx = j - 4;
                }
                j++;
                arrayIdx += OUTPUT_SIZE[2];
            }

            if (maxConf > DETECT_THRESHOLD) {
                float cx = recognitionArray[c];
                float cy = recognitionArray[c + OUTPUT_SIZE[2]];
                float w = recognitionArray[c + OUTPUT_SIZE[2] * 2];
                float h = recognitionArray[c + OUTPUT_SIZE[2] * 3];
                float x1 = cx - (w / 2f);
                float y1 = cy - (h / 2f);
                float x2 = cx + (w / 2f);
                float y2 = cy + (h / 2f);

                if (x1 < 0f || x1 > 1f) continue;
                if (y1 < 0f || y1 > 1f) continue;
                if (x2 < 0f || x2 > 1f) continue;
                if (y2 < 0f || y2 > 1f) continue;
                x1 = x1 * INPNUT_SIZE.getWidth();
                x2 = x2 * INPNUT_SIZE.getWidth();
                y1 = y1 * INPNUT_SIZE.getHeight();
                y2 = y2 * INPNUT_SIZE.getHeight();
                String labelName = associatedAxisLabels.get(maxIdx);
                Recognition r = new Recognition(
                        maxIdx,
                        labelName,
                        maxConf,
                        maxConf,
                        new RectF(x1, y1, x2, y2));
                allRecognitions.add(r);
                if(maxConf >1.0f){
                    Log.d(TAG, "detect3: 之心度异常="+maxConf);
                }
            }
        }

        if (allRecognitions.isEmpty()) return null;
        ArrayList<Recognition> result =  PostProcessUtils.applyNMS(allRecognitions, 0.5f);
        Log.d(TAG, "detect3: 抑制前="+allRecognitions.size()+" 抑制后="+result.size());
        for(Recognition r:result){
            Log.d(TAG, "detect3: object="+r.getLabelName()+" score="+r.getLabelScore()+" rect="+r.getLocation());
        }
        return result;
    }

    /**
     * 非极大抑制
     *
     * @param allRecognitions
     * @return
     */
    protected ArrayList<Recognition> nms(ArrayList<Recognition> allRecognitions) {
        ArrayList<Recognition> nmsRecognitions = new ArrayList<Recognition>();

        // 遍历每个类别, 在每个类别下做nms
        for (int i = 0; i < OUTPUT_SIZE[1]-4; i++) {
            // 这里为每个类别做一个队列, 把labelScore高的排前面
            PriorityQueue<Recognition> pq =
                    new PriorityQueue<Recognition>(
                            OUTPUT_SIZE[2],
                            new Comparator<Recognition>() {
                                @Override
                                public int compare(final Recognition l, final Recognition r) {
                                    // Intentionally reversed to put high confidence at the head of the queue.
                                    return Float.compare(r.getLabelScore(), l.getLabelScore());
                                }
                            });

            // 相同类别的过滤出来, 且obj要大于设定的阈值
            for (int j = 0; j < allRecognitions.size(); ++j) {
//                if (allRecognitions.get(j).getLabelId() == i) {
                if (allRecognitions.get(j).getLabelId() == i && allRecognitions.get(j).getLabelScore() > DETECT_THRESHOLD) {
                    pq.add(allRecognitions.get(j));
//                    Log.i("tfliteSupport", allRecognitions.get(j).toString());
                }
            }

            // nms循环遍历
            while (pq.size() > 0) {
                // 概率最大的先拿出来
                Recognition[] a = new Recognition[pq.size()];
                Recognition[] detections = pq.toArray(a);
                Recognition max = detections[0];
                nmsRecognitions.add(max);
                pq.clear();

                for (int k = 1; k < detections.length; k++) {
                    Recognition detection = detections[k];
                    if (boxIou(max.getLocation(), detection.getLocation()) < IOU_THRESHOLD) {
                        pq.add(detection);
                    }
                }
            }
        }
        return nmsRecognitions;
    }

    /**
     * 对所有数据不区分类别做非极大抑制
     *
     * @param allRecognitions
     * @return
     */
    protected ArrayList<Recognition> nmsAllClass(ArrayList<Recognition> allRecognitions) {
        ArrayList<Recognition> nmsRecognitions = new ArrayList<Recognition>();

        PriorityQueue<Recognition> pq =
                new PriorityQueue<Recognition>(
                        100,
                        new Comparator<Recognition>() {
                            @Override
                            public int compare(final Recognition l, final Recognition r) {
                                // Intentionally reversed to put high confidence at the head of the queue.
                                return Float.compare(r.getLabelScore(), l.getLabelScore());
                            }
                        });

        // 相同类别的过滤出来, 且obj要大于设定的阈值
        for (int j = 0; j < allRecognitions.size(); ++j) {
            if (allRecognitions.get(j).getLabelScore() > DETECT_THRESHOLD) {
                pq.add(allRecognitions.get(j));
            }
        }

        while (pq.size() > 0) {
            // 概率最大的先拿出来
            Recognition[] a = new Recognition[pq.size()];
            Recognition[] detections = pq.toArray(a);
            Recognition max = detections[0];
            nmsRecognitions.add(max);
            pq.clear();

            for (int k = 1; k < detections.length; k++) {
                Recognition detection = detections[k];
                if (boxIou(max.getLocation(), detection.getLocation()) < IOU_CLASS_DUPLICATED_THRESHOLD) {
                    pq.add(detection);
                }
            }
        }
        return nmsRecognitions;
    }


    protected float boxIou(RectF a, RectF b) {
        float intersection = boxIntersection(a, b);
        float union = boxUnion(a, b);
        if (union <= 0) return 1;
        return intersection / union;
    }

    protected float boxIntersection(RectF a, RectF b) {
        float maxLeft = a.left > b.left ? a.left : b.left;
        float maxTop = a.top > b.top ? a.top : b.top;
        float minRight = a.right < b.right ? a.right : b.right;
        float minBottom = a.bottom < b.bottom ? a.bottom : b.bottom;
        float w = minRight -  maxLeft;
        float h = minBottom - maxTop;

        if (w < 0 || h < 0) return 0;
        float area = w * h;
        return area;
    }

    protected float boxUnion(RectF a, RectF b) {
        float i = boxIntersection(a, b);
        float u = (a.right - a.left) * (a.bottom - a.top) + (b.right - b.left) * (b.bottom - b.top) - i;
        return u;
    }

    /**
     * 添加NNapi代理
     */
    @Override
    public void addNNApiDelegate() {
        NnApiDelegate nnApiDelegate = null;
        // Initialize interpreter with NNAPI delegate for Android Pie or above
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
//            NnApiDelegate.Options nnApiOptions = new NnApiDelegate.Options();
//            nnApiOptions.setAllowFp16(true);
//            nnApiOptions.setUseNnapiCpu(true);
            //ANEURALNETWORKS_PREFER_LOW_POWER：倾向于以最大限度减少电池消耗的方式执行。这种设置适合经常执行的编译。
            //ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER：倾向于尽快返回单个答案，即使这会耗费更多电量。这是默认值。
            //ANEURALNETWORKS_PREFER_SUSTAINED_SPEED：倾向于最大限度地提高连续帧的吞吐量，例如，在处理来自相机的连续帧时。
//            nnApiOptions.setExecutionPreference(NnApiDelegate.Options.EXECUTION_PREFERENCE_SUSTAINED_SPEED);
//            nnApiDelegate = new NnApiDelegate(nnApiOptions);
            nnApiDelegate = new NnApiDelegate();
            options.addDelegate(nnApiDelegate);
            Log.i("tfliteSupport", "using nnapi delegate.");
        }
    }

    /**
     * 添加GPU代理
     */
    @Override
    public void addGPUDelegate() {
//        addThread(4);
        CompatibilityList compatibilityList = new CompatibilityList();
        if(compatibilityList.isDelegateSupportedOnThisDevice()){
            GpuDelegateFactory.Options delegateOptions = compatibilityList.getBestOptionsForThisDevice();
            GpuDelegate gpuDelegate = new GpuDelegate(delegateOptions);
            options.addDelegate(gpuDelegate);
            Log.i("tfliteSupport", "using gpu delegate.");
        } else {
            addThread(4);
        }
    }

    /**
     * 添加线程数
     * @param thread
     */
    public void addThread(int thread) {
        options.setNumThreads(thread);
    }

}
