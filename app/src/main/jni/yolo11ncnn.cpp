//
// Created by daijun on 2025/12/25.
//
#include <android/asset_manager_jni.h>
#include <android/native_window_jni.h>
#include <android/native_window.h>
#include <android/bitmap.h>
#include "yolo11.h"
#include <android/log.h>

#include <jni.h>

#include <string>
#include <vector>
#include <sys/time.h>

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, "yolo11ncnn", ##__VA_ARGS__);

// FIXME DeleteGlobalRef is missing for objCls
static jclass objCls = NULL;
static jmethodID constructortorId;
static jfieldID xId;
static jfieldID yId;
static jfieldID wId;
static jfieldID hId;
static jfieldID labelId;
static jfieldID probId;

static YOLO11 *g_yolo11 = 0;
static ncnn::Mutex lock;

extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM * vm , void *reserved )
{
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn" , "JNI_OnLoad" ) ;
    ncnn::create_gpu_instance();
    return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM * vm, void * reserved) {
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "JNI_OnUnload");

    {
        ncnn::MutexLockGuard g(lock);

        delete g_yolo11;
        g_yolo11 = 0;
    }

    ncnn::destroy_gpu_instance();
}

JNIEXPORT jboolean JNICALL Java_com_example_djyolo_detector_YOLO11Ncnn_loadModel(JNIEnv* env, jobject thiz, jobject assetManager, jstring model, jstring param, jint cpugpu,jint taskid)
{
//    if (taskid < 0 || taskid > 4 || modelid < 0 || modelid > 8 || cpugpu < 0 || cpugpu > 2)
//    {
//        return JNI_FALSE;
//    }

    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "loadModel %p", mgr);

//    std::string parampath = std::string("yolo11") + modeltypes[(int)modelid] + tasknames[(int)taskid] + ".ncnn.param";
//    std::string modelpath = std::string("yolo11") + modeltypes[(int)modelid] + tasknames[(int)taskid] + ".ncnn.bin";
    const char* modelStr = env->GetStringUTFChars(model, nullptr);
    const char* paramStr = env->GetStringUTFChars(param, nullptr);
    std::string parampath = std::string(paramStr);
    std::string modelpath = std::string(modelStr);
//    std::string parampath = std::string("yolov5_hand_ges.param");
//    std::string modelpath = std::string("yolov5_hand_ges.bin");
    LOGD("modelpath=%s", modelpath.c_str())
    LOGD("parampath=%s", parampath.c_str())
    bool use_gpu = (int)cpugpu == 1;
    bool use_turnip = (int)cpugpu == 2;

    // reload
    {
        ncnn::MutexLockGuard g(lock);

        {
            static int old_taskid = 0;
            static std::string old_modelid = "";
            static int old_cpugpu = 0;
            if (taskid != old_taskid || modelpath.compare(old_modelid) !=0 || cpugpu != old_cpugpu)
            {
                // taskid or model or cpugpu changed
                delete g_yolo11;
                g_yolo11 = 0;
            }
            old_taskid = taskid;
            old_modelid = modelpath;
            old_cpugpu = cpugpu;

            ncnn::destroy_gpu_instance();

            if (use_turnip)
            {
                ncnn::create_gpu_instance("libvulkan_freedreno.so");
            }
            else if (use_gpu)
            {
                ncnn::create_gpu_instance();
                int gpu_count = ncnn::get_gpu_count();
                LOGD("gpu_count=%d", gpu_count)
            }

            if (!g_yolo11)
            {
                LOGD("new yolo11")
//                g_yolo11 = new YOLO11;
                if(taskid == 0) g_yolo11 = new Yolo5HandGes;
                if(taskid == 1) g_yolo11 = new YOLO11_stripcv_det;
                g_yolo11->load(mgr, parampath.c_str(), modelpath.c_str(), use_gpu || use_turnip);
            }
            int target_size = 320;
            target_size = 320;
            g_yolo11->set_det_target_size(target_size);
        }
    }

    // init jni glue
    jclass localObjCls = env->FindClass("com/example/djyolo/bean/NNDetObj");
    objCls = reinterpret_cast<jclass>(env->NewGlobalRef(localObjCls));

//    constructortorId = env->GetMethodID(objCls, "<init>", "(Lcom/tencent/yolo11ncnn/YOLO11Ncnn;)V");

    // 注意：NNDetObj 是一个简单类，没有构造函数参数
    constructortorId = env->GetMethodID(objCls, "<init>", "()V");

    xId = env->GetFieldID(objCls, "x", "F");
    yId = env->GetFieldID(objCls, "y", "F");
    wId = env->GetFieldID(objCls, "w", "F");
    hId = env->GetFieldID(objCls, "h", "F");
    labelId = env->GetFieldID(objCls, "label", "Ljava/lang/String;");
    probId = env->GetFieldID(objCls, "prob", "F");

    return JNI_TRUE;
}



JNIEXPORT jobjectArray JNICALL Java_com_example_djyolo_detector_YOLO11Ncnn_Detect(JNIEnv* env, jobject thiz, jobject bitmap, jboolean use_gpu){
    AndroidBitmapInfo info;
    AndroidBitmap_getInfo(env, bitmap, &info);
    const int img_w = info.width;
    const int img_h = info.height;
//    const int target_size = 320;//640;
    const int target_size = g_yolo11->get_det_target_size();//640;

    // letterbox pad to multiple of max_stride
    int w = img_w;
    int h = img_h;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }
    ncnn::Mat in = ncnn::Mat::from_android_bitmap_resize(env, bitmap, ncnn::Mat::PIXEL_RGB, w,h);
//    YOLO11_stripcv_det* yolo_stripcv_det = new YOLO11_stripcv_det;
    YOLO11* yolo_stripcv_det = g_yolo11;
    std::vector<Recognition> objects;
    LOGD("detect %d %d %d %d %f targetSize=%d",w,h,img_w,img_h,scale,target_size)
    // 记录开始时间
    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);
    yolo_stripcv_det->detect(in,img_w,img_h,w,h,scale,g_yolo11->getNet(),objects);
    LOGD("detect size= %d",objects.size())
    // 记录结束时间并计算耗时
    gettimeofday(&end_time, NULL);

    long execution_time = (end_time.tv_sec - start_time.tv_sec) * 1000000 + (end_time.tv_usec - start_time.tv_usec);
    LOGD("detect method execution time: %ld microseconds, =%0.1f", execution_time, execution_time / 1000.0);

//    yolo_stripcv_det->draw(objects);

    //保存成java列表返回
    jobjectArray jObjArray = env->NewObjectArray(objects.size(), objCls, NULL);
    for (size_t i=0; i<objects.size(); i++)
    {
        jobject jObj = env->NewObject(objCls, constructortorId, thiz);

        env->SetFloatField(jObj, xId, objects[i].x);
        env->SetFloatField(jObj, yId, objects[i].y);
        env->SetFloatField(jObj, wId, objects[i].width);
        env->SetFloatField(jObj, hId, objects[i].height);
        env->SetObjectField(jObj, labelId, env->NewStringUTF(yolo_stripcv_det->get_label(objects[i].label).c_str()));
//        env->SetObjectField(jObj, labelId, env->NewStringUTF(class_names[objects[i].label]));
        env->SetFloatField(jObj, probId, objects[i].prob);
        env->SetObjectArrayElement(jObjArray, i, jObj);
    }
    return jObjArray;
//    return nullptr;
}


}


