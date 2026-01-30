package com.example.djyolo

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.media.ExifInterface
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.Toast
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import com.example.djyolo.bean.NNDetObj
import com.example.djyolo.detector.YOLO11Ncnn
import com.example.djyolo.mediapipe.facemesh.FaceMeshMainActivity
import com.example.djyolo.mediapipe.hands.MainActivity
import com.example.djyolo.util.ThreadUtil
import java.io.FileNotFoundException
import java.io.IOException
import java.lang.String

class MainActivity : AppCompatActivity() {
    private val yolo11ncnn: YOLO11Ncnn = YOLO11Ncnn()
    private val current_task = 0 // 0-手势 1-目标检测
    private val current_model = 0
    private val current_cpugpu = 0
    var mSelBitmap: Bitmap? = null
    lateinit var btnPicDetect: Button
    val SELECT_IMAGE: Int = 1
    lateinit var imageView: ImageView
    lateinit var btnMpHands: Button




    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }
        initModel()
        initView()
    }

    fun initView(){
        imageView = findViewById<ImageView>(R.id.cameraView)
        btnPicDetect = findViewById<Button>(R.id.btnPicDetect)
        btnPicDetect.setOnClickListener(View.OnClickListener { v: View? ->
            val i = Intent(Intent.ACTION_PICK)
            i.setType("image/*")
            startActivityForResult(i, SELECT_IMAGE)
        })
        findViewById<Button>(R.id.btnCamera).setOnClickListener(View.OnClickListener { v: View? ->
            val intent = Intent(this, CameraNcnnActivity::class.java)
            startActivity(intent)
        })
        btnMpHands = findViewById(R.id.btn_mp_hands)
        btnMpHands.setOnClickListener(View.OnClickListener { v: View? ->
            var intent = Intent(this, MainActivity::class.java)
            startActivity(intent)
        })
        findViewById<Button>(R.id.btn_mp_face).setOnClickListener({
            var intent = Intent(this, FaceMeshMainActivity::class.java)
            startActivity(intent)
        })
        findViewById<Button>(R.id.btn_yolo_tflite).setOnClickListener({
            var intent = Intent(this, CameraTfliteActivity::class.java)
            startActivity(intent)
        })
    }


    fun initModel(){
        var modelArray = resources.getStringArray(R.array.model_array)
        var paramArray = resources.getStringArray(R.array.model_param_array);
        var selModel = modelArray[current_model]
        var selParam = paramArray[current_model]
        val ret_init = yolo11ncnn.loadModel(assets, selModel, selParam, current_cpugpu,current_task)
        if (!ret_init) {
            Log.e("MainActivity", "yolo11ncnn loadModel failed")
        }
    }


    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (resultCode == RESULT_OK && null != data) {
            val selectedImage = data.data
            try {
                if (requestCode == SELECT_IMAGE) {
                    val bitmap: Bitmap = decodeUri(selectedImage!!)
                    mSelBitmap = bitmap
                    val yourSelectedImage = bitmap.copy(Bitmap.Config.ARGB_8888, true)
                    Log.d(
                        "MainActivity",
                        "onActivityResult: 选择了图片width=" + yourSelectedImage.width + "  height=" + yourSelectedImage.height
                    )
                    var timebefore = System.currentTimeMillis()
                    ThreadUtil.runInThreadPool({
                        val result: Array<NNDetObj> = yolo11ncnn.Detect(yourSelectedImage, true)
                        var timeafter = System.currentTimeMillis()
                        var timeDiff = timeafter - timebefore
                        if(result.isNullOrEmpty()){
                            Toast.makeText(this, "未检测到目标", Toast.LENGTH_SHORT)
                            return@runInThreadPool
                        }
                        Log.d(
                            "MainActivity",
                            "onActivityResult: " + result.size + " result=" + result[0].toString()+" 耗时="+timeDiff
                        )
                        ThreadUtil.runInUIThread({
                            showObjects(result)
                        })
                    })

                }
            } catch (e: FileNotFoundException) {
                Log.e("MainActivity", "FileNotFoundException")
                return
            }
        }
    }

    private fun showObjects(objects: Array<NNDetObj>?) {
        if (objects == null) {
            imageView.setImageBitmap(mSelBitmap)
            return
        }
        // draw objects on bitmap
        val rgba = mSelBitmap!!.copy(Bitmap.Config.ARGB_8888, true)
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
            paint.color = colors[i % 19]

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
        imageView.setImageBitmap(rgba)
    }


    @Throws(FileNotFoundException::class)
    private fun decodeUri(selectedImage: Uri): Bitmap {
        // Decode image size
        val o = BitmapFactory.Options()
        o.inJustDecodeBounds = true
        BitmapFactory.decodeStream(contentResolver.openInputStream(selectedImage), null, o)

        // The new size we want to scale to
        val REQUIRED_SIZE = 640

        // Find the correct scale value. It should be the power of 2.
        var width_tmp = o.outWidth
        var height_tmp = o.outHeight
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
        val bitmap =
            BitmapFactory.decodeStream(contentResolver.openInputStream(selectedImage), null, o2)

        // Rotate according to EXIF
        var rotate = 0
        try {
            val exif = ExifInterface(contentResolver.openInputStream(selectedImage)!!)
            val orientation = exif.getAttributeInt(
                ExifInterface.TAG_ORIENTATION,
                ExifInterface.ORIENTATION_NORMAL
            )
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
        return Bitmap.createBitmap(bitmap!!, 0, 0, bitmap!!.width, bitmap!!.height, matrix, true)
    }

}