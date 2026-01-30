package com.example.djyolo

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.SurfaceTexture
import android.media.Image
import android.os.Bundle
import android.util.Log
import android.view.SurfaceHolder
import android.view.TextureView
import android.view.TextureView.SurfaceTextureListener
import android.widget.Button
import android.widget.ImageView
import android.widget.Toast
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import com.example.djyolo.detector.tflite.Recognition
import com.example.djyolo.util.ImageAnalyse
import com.example.djyolo.util.ImageProcess
import com.example.djyolo.util.Camera2Util
import com.example.djyolo.util.ThreadUtil
import java.util.concurrent.ConcurrentLinkedQueue

class CameraTfliteActivity : AppCompatActivity(), SurfaceHolder.Callback {
    private val CAMERA_PERMISSION_REQUEST_CODE: Int = 100
    lateinit var textureView:TextureView
    lateinit var labelView:ImageView
    private lateinit var camera2Util: Camera2Util
    private val TAG = "CameraTfliteActivity"
    var imageProcess: ImageProcess = ImageProcess()
    var isWorking: Boolean = false
    var imageAnalyse: ImageAnalyse = ImageAnalyse(Camera2Util.mWidth, Camera2Util.mHeight, 0)
    private val yuvDataQueue = ConcurrentLinkedQueue<IntArray>()
    var mPreviewHeight:Int = 1280
    var mPreviewWidth:Int = 720
    var isCamPrivilegeGranted = false


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_camera_tflite)
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }
        initModel()
        initView()

        // 检查相机权限
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED
        ) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf<kotlin.String>(Manifest.permission.CAMERA),
                CAMERA_PERMISSION_REQUEST_CODE
            )
            return
        }
        isCamPrivilegeGranted = true
        initCameraPreview()
    }

    fun initView(){
        findViewById<Button>(R.id.btn_back).setOnClickListener({
            finish()
        })
        textureView = findViewById<TextureView>(R.id.textureView)
        labelView = findViewById<ImageView>(R.id.ivLabel)
        textureView.post({
            mPreviewWidth = textureView.width
            mPreviewHeight = textureView.height
        })
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if(requestCode == CAMERA_PERMISSION_REQUEST_CODE){
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                isCamPrivilegeGranted = true
                // 只有在TextureView准备好时才初始化相机
                if (textureView.isAvailable) {
                    initCameraPreview()
                    camera2Util.createPreviewView(textureView)
                    camera2Util.startCamera2(applicationContext)
                } else {
                    // TextureView尚未准备就绪，等待其回调
                    textureView.surfaceTextureListener = textureListener
                }
            } else {
                Toast.makeText(this, "Camera permission denied", Toast.LENGTH_SHORT).show()
            }
        }
    }

    fun initCameraPreview(){
        camera2Util = Camera2Util()
        camera2Util.initCamera(applicationContext)
        textureView.setSurfaceTextureListener(textureListener)
        camera2Util.setCameraCallback2(object : Camera2Util.CameraPreviewCallback2 {
            override fun onCameraPreview(
                y: ByteArray?,
                u: ByteArray?,
                v: ByteArray?,
                yRowStride: Int,
                uvRowStride: Int,
                uvPixelStride: Int,
                image: Image?
            ) {
                if(!isWorking)return
                var width = Camera2Util.mWidth;
                var height = Camera2Util.mHeight;
                val rgbBytes = IntArray(width * height)
                imageProcess.YUV420ToARGB8888(
                    y,
                    u,
                    v,
                    width,
                    height,
                    yRowStride,
                    uvRowStride,
                    uvPixelStride,
                    rgbBytes
                )
                if(yuvDataQueue.size <6){
                    yuvDataQueue.add(rgbBytes)
                }else{
                    yuvDataQueue.clear()
                }

                //                    Log.d(TAG, "onCameraPreview: width="+bitmapGD.getWidth()+" height="+bitmapGD.getHeight());
                ThreadUtil.runInSingleThreadPool({
                    if (yuvDataQueue.isEmpty())return@runInSingleThreadPool
                    val rgbData = yuvDataQueue.poll()
                    // 创建Bitmap对象
                    val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
                    // 将RGB数据设置到Bitmap中
                    bitmap.setPixels(rgbData, 0, width, 0, 0, width, height)
                    val bimapGes = imageAnalyse.analyze(bitmap, width, height)
                    ThreadUtil.runInUIThread({
                        var labelBitmap = drawRecognitions(bimapGes,imageAnalyse.getPreviewTransform2(mPreviewWidth,mPreviewHeight))
                        labelView.setImageBitmap(labelBitmap)
                    })
                })

            }
        })
    }

    var textureListener:SurfaceTextureListener = object:SurfaceTextureListener{
        override fun onSurfaceTextureAvailable(p0: SurfaceTexture, p1: Int, p2: Int) {
            if (isCamPrivilegeGranted){
                camera2Util.createPreviewView(textureView)
                camera2Util.startCamera2(applicationContext)
            }
        }

        override fun onSurfaceTextureSizeChanged(p0: SurfaceTexture, p1: Int, p2: Int) {
        }

        override fun onSurfaceTextureDestroyed(p0: SurfaceTexture): Boolean {
            return false;
        }

        override fun onSurfaceTextureUpdated(p0: SurfaceTexture) {
        }
    }

    override fun surfaceCreated(p0: SurfaceHolder) {

    }

    override fun surfaceChanged(p0: SurfaceHolder, p1: Int, p2: Int, p3: Int) {

    }

    override fun surfaceDestroyed(p0: SurfaceHolder) {

    }


    fun initModel(){
        ThreadUtil.runInThreadPool({
            //yolov5s、yolov5n、yolov11s、yolov11n
            imageAnalyse.initModel("yolov11s", applicationContext)
            isWorking = true;
        })
    }



    private fun drawRecognitions(
        recognitions: ArrayList<Recognition>,
        modelToPreviewTransform: Matrix
    ): Bitmap {
        // 绘制识别结果
        val emptyCropSizeBitmap =
            Bitmap.createBitmap(mPreviewWidth, mPreviewHeight, Bitmap.Config.ARGB_8888)
        val cropCanvas = Canvas(emptyCropSizeBitmap)
        // 边框画笔
        val boxPaint = Paint()
        boxPaint.strokeWidth = 5f
        boxPaint.style = Paint.Style.STROKE
        boxPaint.color = Color.RED
        // 字体画笔
        val textPain = Paint()
        textPain.textSize = 50f
        textPain.color = Color.RED
        textPain.style = Paint.Style.FILL

        for (res in recognitions) {
            val location = res.location
            val label = res.labelName
            val confidence = res.confidence
            modelToPreviewTransform.mapRect(location)
            cropCanvas.drawRect(location, boxPaint)
            cropCanvas.drawText(
                label + ":" + kotlin.String.format("%.2f", confidence),
                location.left,
                location.top,
                textPain
            )
        }
        return emptyCropSizeBitmap
    }




}