package com.example.djyolo

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.SurfaceTexture
import android.media.ExifInterface
import android.media.Image
import android.os.Bundle
import android.util.Log
import android.view.SurfaceHolder
import android.view.SurfaceView
import android.view.TextureView
import android.view.TextureView.SurfaceTextureListener
import android.view.View
import android.widget.AdapterView
import android.widget.Button
import android.widget.ImageView
import android.widget.Spinner
import android.widget.Toast
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import com.example.djyolo.bean.NNDetObj
import com.example.djyolo.detector.YOLO11Ncnn
import com.example.djyolo.util.ImageProcess
import com.example.djyolo.util.Camera2Util
import com.example.djyolo.util.ThreadUtil
import java.io.IOException
import java.lang.String

class CameraNcnnActivity : AppCompatActivity(), SurfaceHolder.Callback {
    private val CAMERA_PERMISSION_REQUEST_CODE: Int = 100
    private val yolo11ncnn: YOLO11Ncnn = YOLO11Ncnn()
    private var current_task = 0 // 0-手势 1-目标检测
    private var current_model = 1 //0-nano 1-s
    private val current_cpugpu = 0 //1-使用gpu加速，0-不使用
    var mSelBitmap: Bitmap? = null
    lateinit var btnPicDetect: Button
    val SELECT_IMAGE: Int = 1
    lateinit var surfaceView: SurfaceView
    lateinit var textureView:TextureView
    lateinit var spinnerModel:Spinner
    lateinit var spinnerTask:Spinner
    lateinit var labelView:ImageView
    private lateinit var camera2Util: Camera2Util
    private val TAG = "CameraActivity"
    var imageProcess: ImageProcess = ImageProcess()
    var isWorking: Boolean = false
    var isCamPrivilegeGranted = false



    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_camera)
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
        surfaceView = findViewById<SurfaceView>(R.id.cameraView)
        textureView = findViewById<TextureView>(R.id.textureView)
        labelView = findViewById<ImageView>(R.id.ivLabel)
        btnPicDetect = findViewById<Button>(R.id.btnPicDetect)
        btnPicDetect.setOnClickListener(View.OnClickListener { v: View? ->
            val i = Intent(Intent.ACTION_PICK)
            i.setType("image/*")
            startActivityForResult(i, SELECT_IMAGE)
        })
        spinnerModel=findViewById<Spinner>(R.id.spinnerModel)
        spinnerTask=findViewById<Spinner>(R.id.spinnerTask)
        spinnerModel.setOnItemSelectedListener(object : AdapterView.OnItemSelectedListener{
            override fun onItemSelected(p0: AdapterView<*>?, p1: View?, position: Int, p3: Long) {
                isWorking = false;
                ThreadUtil.runInUIThreadDelayed(Runnable {
                    if(position != current_model){
                        current_model = position
                        initModel()
                        isWorking = true;
                    }
                },2000)
            }

            override fun onNothingSelected(p0: AdapterView<*>?) {

            }

        })
        spinnerTask.setOnItemSelectedListener(object : AdapterView.OnItemSelectedListener{
            override fun onItemSelected(p0: AdapterView<*>?, p1: View?, position: Int, p3: Long) {
                if(position != current_task){
                    current_task = position
                    if(current_task == 0){
                        spinnerModel.setSelection(0)
                    }else{
                        spinnerModel.setSelection(1)
                    }
//                    initModel()
                }
            }

            override fun onNothingSelected(p0: AdapterView<*>?) {

            }

        })
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out kotlin.String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if(requestCode == CAMERA_PERMISSION_REQUEST_CODE){
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                Log.d(TAG, "onRequestPermissionsResult: granted")
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
        surfaceView.holder.addCallback(this)
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

                // 创建Bitmap对象
                val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
                // 将RGB数据设置到Bitmap中
                bitmap.setPixels(rgbBytes, 0, width, 0, 0, width, height)
//                ThreadUtil.runInThreadPool(Runnable {
                    var bitmapTrans = decodeUri(bitmap)
                    var timePre = System.currentTimeMillis();
                    var detecResults = yolo11ncnn.Detect(bitmapTrans,true)
                    var timeAfter = System.currentTimeMillis()
                    if(detecResults != null){
                        for (i in detecResults.indices){
                            var nnDetObj = detecResults[i]
                            Log.d(TAG, "onCameraPreview: "+nnDetObj.toString()+" timeDiff="+(timeAfter-timePre))
                        }
                    }
                    ThreadUtil.runInUIThread({
                        showObjects(detecResults)
                    })
//                })

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
        var modelArray = resources.getStringArray(R.array.model_array)
        var paramArray = resources.getStringArray(R.array.model_param_array);
        var taskArray = resources.getStringArray(R.array.task_type_array)
        var selModel = modelArray[current_model]
        var selParam = paramArray[current_model]
        Log.d(TAG, "initModel: taskID="+taskArray[current_task]+" model="+selModel)
        val ret_init = yolo11ncnn.loadModel(assets, selModel, selParam, current_cpugpu,current_task)
        if (!ret_init) {
            Log.e("MainActivity", "yolo11ncnn loadModel failed")
        }
        isWorking = true;
    }


    private fun showObjects(objects: Array<NNDetObj>?) {
        if (objects == null) {
//            surfaceView.setImageBitmap(mSelBitmap)
            return
        }
        // draw objects on bitmap
        var rgba = Bitmap.createBitmap(Camera2Util.mHeight, Camera2Util.mWidth, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(rgba)

        val paint = Paint()
        paint.style = Paint.Style.STROKE
        paint.strokeWidth = 4f

        val textbgpaint = Paint()
        textbgpaint.color = Color.WHITE
        textbgpaint.style = Paint.Style.FILL

        val textpaint = Paint()
        textpaint.color = Color.BLACK
        textpaint.textSize = 20f
        textpaint.textAlign = Paint.Align.LEFT

        val colors = intArrayOf(
            Color.rgb(54, 67, 244),
            Color.rgb(99, 30, 233),
            Color.rgb(176, 39, 156),
            Color.rgb(183, 58, 103),
            Color.rgb(181, 81, 63),
            Color.rgb(243, 150, 33),
            Color.rgb(244, 169, 3),
            Color.rgb(212, 188, 0),
            Color.rgb(136, 150, 0),
            Color.rgb(80, 175, 76),
            Color.rgb(74, 195, 139)
        )

        for (i in objects.indices) {
            paint.color = colors[i % 11]

            canvas.drawRect(
                objects[i].x,
                objects[i].y,
                objects[i].x + objects[i].w,
                objects[i].y + objects[i].h,
                paint
            )

            // draw filled text inside image
            run {
                val text =
                    objects[i].label + " = " + String.format("%.1f", objects[i].prob * 100) + "%"
                val text_width = textpaint.measureText(text)
                val text_height = -textpaint.ascent() + textpaint.descent()

                var x = objects[i].x
                var y = objects[i].y - text_height
                if (y < 0) y = 0f
                if (x + text_width > rgba.width) x = rgba.width - text_width

                canvas.drawRect(x, y, x + text_width, y + text_height, textbgpaint)
                canvas.drawText(text, x, y - textpaint.ascent(), textpaint)
            }
        }
        labelView.setImageBitmap(rgba)
//        surfaceView.setImageBitmap(rgba)
    }


    private fun decodeUri(selectedImage: Bitmap): Bitmap {
        // Decode image size

        // The new size we want to scale to
        val REQUIRED_SIZE = 640

        // Find the correct scale value. It should be the power of 2.
        var width_tmp = selectedImage.width
        var height_tmp = selectedImage.height
        var scale = 1
        while (true) {
            if (width_tmp / 2 < REQUIRED_SIZE
                || height_tmp / 2 < REQUIRED_SIZE
            ) {
                break
            }
            width_tmp /= 2
            height_tmp /= 2
            scale *= 2
        }

        // Decode with inSampleSize
        val o2 = BitmapFactory.Options()
        o2.inSampleSize = scale
//        val bitmap = BitmapFactory.decodeStream(contentResolver.openInputStream(selectedImage), null, o2)
        val bitmap = selectedImage
        // Rotate according to EXIF
        var rotate = 0
        try {
//            val exif = ExifInterface(contentResolver.openInputStream(selectedImage)!!)
//            val orientation = exif.getAttributeInt(
//                ExifInterface.TAG_ORIENTATION,
//                ExifInterface.ORIENTATION_NORMAL
//            )
            val orientation = ExifInterface.ORIENTATION_ROTATE_90
            when (orientation) {
                ExifInterface.ORIENTATION_ROTATE_270 -> rotate = 270
                ExifInterface.ORIENTATION_ROTATE_180 -> rotate = 180
                ExifInterface.ORIENTATION_ROTATE_90 -> rotate = 90
            }
        } catch (e: IOException) {
            Log.e("MainActivity", "ExifInterface IOException")
        }

        val matrix = Matrix()
        matrix.postRotate(rotate.toFloat())
        val result = Bitmap.createBitmap(bitmap!!, 0, 0, bitmap!!.width, bitmap!!.height, matrix, true)
        return result
    }




}