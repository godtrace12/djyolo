package com.example.djyolo.util;

import android.content.Context;
import android.graphics.ImageFormat;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.CaptureResult;
import android.hardware.camera2.TotalCaptureResult;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.Image;
import android.media.ImageReader;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Log;
import android.util.Size;
import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.TextureView;


import java.util.Arrays;

import androidx.annotation.NonNull;
import io.netty.buffer.ByteBuf;
import io.netty.buffer.Unpooled;

/**
 * @Author：
 * @Description: camera2工具类
 * @CreateDate：
 */
public class Camera2Util {
    private static final String TAG = "RkCamera2Util";
    private int cameraID = 0;
    private ImageReader mImageReader;
    // 配置摄像头图像的宽高
    public static int mWidth = 1280;//1280  640
    public static int mHeight = 720;//720 480
    private SurfaceHolder mSurfaceHolder;
    public static Context mContext;
    public static CameraPreviewCallback2 mCallbck2 = null;


    public boolean initCamera(Context context){
        mContext = context;
        return true;
    }

    public void setCameraCallback2(CameraPreviewCallback2 callbck) {
        this.mCallbck2 = callbck;
    }


    public boolean startCamera2(Context context) throws CameraAccessException {
        final CameraManager manager = (CameraManager) context.getSystemService(Context.CAMERA_SERVICE);
        String[] cameraIds = manager.getCameraIdList();
        for(String cameraId: cameraIds){
            CameraCharacteristics cameraCharacteristics = manager.getCameraCharacteristics(cameraId);
            int facing  = cameraCharacteristics.get(CameraCharacteristics.LENS_FACING);
            Log.d(TAG, "startCamera2: dj--cameraId="+cameraId+" facing="+facing);
        }
        try {
            String cameraId = ""+cameraID;
            CameraCharacteristics characteristics = manager.getCameraCharacteristics(cameraId);
            StreamConfigurationMap map = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
            assert map != null;
            searchSupportSize(manager);
//            mPreviewSize = chooseOptimalSize(map.getOutputSizes(Surface.class),
//                    width, height, width, height,
//                    map.getOutputSizes(Surface.class)[0]);

//            mImageReader = ImageReader.newInstance(mPreviewSize.getWidth(), mPreviewSize.getHeight(),
//                    ImageFormat.JPEG, 2);
            mImageReader = ImageReader.newInstance(mWidth, mHeight,
                    ImageFormat.YUV_420_888, 2);
            mImageReader.setOnImageAvailableListener(mOnImageAvailableListener, mBackgroundHandler);

            manager.openCamera(cameraId, mStateCallback, null);
        } catch (CameraAccessException e) {
            e.printStackTrace();
            Log.e(TAG, "startCamera2: e="+e);
        }
        return true;
    }

    private CameraDevice mCameraDevice;
    private final CameraDevice.StateCallback mStateCallback = new CameraDevice.StateCallback() {

        @Override
        public void onOpened(@NonNull CameraDevice cameraDevice) {
            Log.d(TAG, "onOpened: ");
            mCameraDevice = cameraDevice;
            createCameraPreviewSession();
        }

        @Override
        public void onDisconnected(@NonNull CameraDevice cameraDevice) {
            Log.d(TAG, "onDisconnected: ");
            cameraDevice.close();
            mCameraDevice = null;
        }

        @Override
        public void onError(@NonNull CameraDevice cameraDevice, int error) {
            Log.e(TAG, "onError: >>>>>>>>>>>>>>>>>");
            cameraDevice.close();
            mCameraDevice = null;
        }
    };


    private void searchSupportSize(CameraManager manager){
        try {
            for (String cameraId : manager.getCameraIdList()) {
                CameraCharacteristics characteristics = manager.getCameraCharacteristics(cameraId);
                // 获取流式传输格式，这里我们选择YUV_420_888作为示例
                StreamConfigurationMap map = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
                if (map != null) {
                    // 获取所有支持的预览尺寸
//                    Size[] outputSizes = map.getOutputSizes(SurfaceTexture.class);
                    Size[] outputSizes = map.getOutputSizes(ImageFormat.YUV_420_888);
                    if(outputSizes == null){
                        Log.e(TAG, "searchSupportSize: 未获取到支持的预览尺寸");
                        return;
                    }
                    //预览尺寸
                    for (Size size : outputSizes) {
                        Log.i(TAG, "Camera " + cameraId + " supports output size: w" + size.getWidth() + " h=" + size.getHeight());
                    }
                }
            }
        } catch (CameraAccessException e) {
            Log.e(TAG, "searchSupportSize: e="+e);
        }
    }

    private CameraCaptureSession mCaptureSession;
    private CaptureRequest.Builder mPreviewRequestBuilder;
    private CaptureRequest mPreviewRequest;
    private HandlerThread mBackgroundThread;
    private Handler mBackgroundHandler;
    SurfaceTexture texture;

    public void createPreviewView(SurfaceView surfaceView) {
        mSurfaceHolder = surfaceView.getHolder();
    }

    public void createPreviewView(TextureView surfaceView) {
        texture = surfaceView.getSurfaceTexture();
        texture.setDefaultBufferSize(mWidth,mHeight);
    }

    private void createCameraPreviewSession() {
        try {
//            Surface surface = mSurfaceHolder.getSurface();
            Surface surface = new Surface(texture);

            mPreviewRequestBuilder = mCameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
            mPreviewRequestBuilder.addTarget(surface);
            mPreviewRequestBuilder.addTarget(mImageReader.getSurface());
            mPreviewRequestBuilder.set(CaptureRequest.JPEG_ORIENTATION,0);
            mPreviewRequestBuilder.set(CaptureRequest.CONTROL_MODE, CaptureRequest.CONTROL_MODE_AUTO);
//            mPreviewRequestBuilder.set(CaptureRequest.);

            mCameraDevice.createCaptureSession(Arrays.asList(surface, mImageReader.getSurface()),
                    new CameraCaptureSession.StateCallback() {

                        @Override
                        public void onConfigured(@NonNull CameraCaptureSession cameraCaptureSession) {
                            if (null == mCameraDevice) {
                                return;
                            }
                            Log.d(TAG, "onConfigured: ");

                            mCaptureSession = cameraCaptureSession;
                            try {
                                mPreviewRequestBuilder.set(CaptureRequest.CONTROL_AF_MODE,
                                        CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE);

                                mPreviewRequest = mPreviewRequestBuilder.build();
                                mCaptureSession.setRepeatingRequest(mPreviewRequest,
                                        mCaptureCallback, mBackgroundHandler);
                            } catch (CameraAccessException e) {
                                e.printStackTrace();
                                Log.e(TAG, "onConfigured: "+e);
                            }
                        }

                        @Override
                        public void onConfigureFailed(
                                @NonNull CameraCaptureSession cameraCaptureSession) {
                            Log.e(TAG, "onConfigureFailed: !!!!");
                        }
                    }, null
            );
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
    }


    private final CameraCaptureSession.CaptureCallback mCaptureCallback = new CameraCaptureSession.CaptureCallback() {

        private void process(CaptureResult result) {
        }

        @Override
        public void onCaptureProgressed(@NonNull CameraCaptureSession session,
                                        @NonNull CaptureRequest request,
                                        @NonNull CaptureResult partialResult) {
        }

        @Override
        public void onCaptureCompleted(@NonNull CameraCaptureSession session,
                                       @NonNull CaptureRequest request,
                                       @NonNull TotalCaptureResult result) {
            process(result);
        }
    };

    private final ImageReader.OnImageAvailableListener mOnImageAvailableListener = new ImageReader.OnImageAvailableListener() {

        @Override
        public void onImageAvailable(ImageReader reader) {
            ThreadUtil.runInThreadPool(() -> {
            Image image = reader.acquireLatestImage();
            if(image == null) return;
            Image.Plane[] planes =image.getPlanes();
            int width = image.getWidth();
            int height = image.getHeight();
            int yRowStride = planes[0].getRowStride();
            int uvRowStride = planes[1].getRowStride();
            int uvPixelStride = planes[1].getPixelStride();
//            Log.d(TAG, "onImageAvailable: width="+width+" height="+height+" yRowStride="+yRowStride+
//                    " uvRowStride="+uvRowStride+" uvPixelStride="+uvPixelStride);

            //发3
            ByteBuf yBuffer;
            ByteBuf uBuffer;
            ByteBuf vBuffer;
            ByteBuf yuvCompositeBuffer;
            byte[] yuvData = null;
            byte[] yuvDataR = null;
            yBuffer = Unpooled.wrappedBuffer(planes[0].getBuffer());
            uBuffer = Unpooled.wrappedBuffer(planes[1].getBuffer());
            vBuffer = Unpooled.wrappedBuffer(planes[2].getBuffer());
            yuvCompositeBuffer = Unpooled.wrappedBuffer(yBuffer, uBuffer);
            // TODO: 2025/10/11 没有完全拷贝yuv，只拷贝了部分？---存储时，数据结构只用了2个panel

            if (yuvData == null || yuvData.length < yuvCompositeBuffer.capacity()) {
//                Log.d(TAG, "YUV_420_888toNV21: yuvData new length="+yuvCompositeBuffer.capacity()+ "  yPnaleLen="+yBuffer.capacity()+" uPanel="+uBuffer.capacity());
                yuvData = new byte[yuvCompositeBuffer.capacity()];
//                yuvDataR = new byte[yuvCompositeBuffer.capacity()+1];
            }
            yuvCompositeBuffer.readBytes(yuvData);
            //发4
            byte[] yData = null;
            byte[] uData = null;
            byte[] vData = null;
            if (yData == null || yData.length < planes[0].getBuffer().capacity()) {
                yData = new byte[planes[0].getBuffer().capacity()];
                uData = new byte[planes[1].getBuffer().capacity()];
                vData = new byte[planes[2].getBuffer().capacity()];
                yBuffer.readBytes(yData);
                uBuffer.readBytes(uData);
                vBuffer.readBytes(vData);

            }

            if(mCallbck2 != null){
                mCallbck2.onCameraPreview(yData,uData,vData,yRowStride,uvRowStride,uvPixelStride,image);
            }

            image.close();
            });
        }
    };


    public interface CameraPreviewCallback2 {
        void onCameraPreview(byte[] y, byte[] u, byte[] v,int yRowStride,int uvRowStride,int uvPixelStride,Image image);
//        void onCameraPreview(byte[][] yu,int yRowStride,int uvRowStride,int uvPixelStride,Image image);

    }



}
