//
// Created by daijun on 2025/12/30.
//
#include "yolo11.h"
#include "cpu.h"
#include <net.h>

static inline float intersection_area(const Recognition& a, const Recognition& b)
{
//    cv::Rect_<float> inter = a.rect & b.rect;
//    return inter.area();

    // 计算交集矩形的边界
    float x1 = std::max(a.x, b.x);
    float y1 = std::max(a.y, b.y);
    float x2 = std::min(a.x + a.width, b.x + b.width);
    float y2 = std::min(a.y + a.height, b.y + b.height);

    // 如果矩形不相交，返回0
    if (x2 <= x1 || y2 <= y1) {
        return 0.0f;
    }

    // 计算交集面积
    float width = x2 - x1;
    float height = y2 - y1;
    return width * height;
}

static void qsort_descent_inplace(std::vector<Recognition>& objects, int left, int right)
{
    int i = left;
    int j = right;
    float p = objects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (objects[i].prob > p)
            i++;

        while (objects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(objects[i], objects[j]);

            i++;
            j--;
        }
    }

    // #pragma omp parallel sections
    {
        // #pragma omp section
        {
            if (left < j) qsort_descent_inplace(objects, left, j);
        }
        // #pragma omp section
        {
            if (i < right) qsort_descent_inplace(objects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Recognition>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Recognition>& objects, std::vector<int>& picked, float nms_threshold, bool agnostic = false)
{
    picked.clear();

    const int n = objects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
//        areas[i] = objects[i].rect.area();
        float areaTmp = objects[i].width*objects[i].height;
        areas[i] =areaTmp;
    }

    for (int i = 0; i < n; i++)
    {
        const Recognition& a = objects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Recognition& b = objects[picked[j]];

            if (!agnostic && a.label != b.label)
                continue;

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

static inline float sigmoid(float x) {
    return static_cast<float>(1.f / (1.f + exp(-x)));
}

static const char *class_names[] = {
        "call", "dislike", "fist", "four", "like", "mute", "ok", "one", "palm",
        "peace",
        "rock", "stop", "stop_inverted", "three", "two_up", "two_up_inverted", "three2", "peace_inverted",
        "no_gesture"
};


static void generate111_proposals(const ncnn::Mat& anchors, int stride, const ncnn::Mat& in_pad, const ncnn::Mat& feat_blob, float prob_threshold, std::vector<Recognition>& objects)
{
    const int num_grid = feat_blob.h;

    int num_grid_x;
    int num_grid_y;
    if (in_pad.w > in_pad.h)
    {
        num_grid_x = in_pad.w / stride;
        num_grid_y = num_grid / num_grid_x;
    }
    else
    {
        num_grid_y = in_pad.h / stride;
        num_grid_x = num_grid / num_grid_y;
    }

    const int num_class = feat_blob.w - 5;

    const int num_anchors = anchors.w / 2;

    for (int q = 0; q < num_anchors; q++)
    {
        const float anchor_w = anchors[q * 2];
        const float anchor_h = anchors[q * 2 + 1];

        const ncnn::Mat feat = feat_blob.channel(q);


        for (int i = 0; i < num_grid_y; i++)
        {
            for (int j = 0; j < num_grid_x; j++)
            {
                const float* featptr = feat.row(i * num_grid_x + j);

                // find class index with max class score
                int class_index = 0;
                float class_score = -FLT_MAX;
                for (int k = 0; k < num_class; k++)
                {
                    float score = featptr[5 + k];
                    if (score > class_score)
                    {
                        class_index = k;
                        class_score = score;
                    }
                }

                float box_score = featptr[4];


                float confidence = sigmoid(box_score) * sigmoid(class_score);


                if (confidence >= prob_threshold)
                {
                    // yolov5/models/yolo.py Detect forward
                    // y = x[i].sigmoid()
                    // y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                    // y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh

                    float dx = sigmoid(featptr[0]);
                    float dy = sigmoid(featptr[1]);
                    float dw = sigmoid(featptr[2]);
                    float dh = sigmoid(featptr[3]);


                    float pb_cx = (dx * 2.f - 0.5f + j) * stride;
                    float pb_cy = (dy * 2.f - 0.5f + i) * stride;

                    float pb_w = pow(dw * 2.f, 2) * anchor_w;
                    float pb_h = pow(dh * 2.f, 2) * anchor_h;

                    float x0 = pb_cx - pb_w * 0.5f;
                    float y0 = pb_cy - pb_h * 0.5f;
                    float x1 = pb_cx + pb_w * 0.5f;
                    float y1 = pb_cy + pb_h * 0.5f;

                    Recognition obj;
                    obj.x = x0;
                    obj.y = y0;
                    obj.width = x1 - x0;
                    obj.height = y1 - y0;
                    obj.label = class_index;
                    obj.prob = confidence;

                    objects.push_back(obj);
                }
            }
        }
    }
}


int Yolo5HandGes::detect(ncnn::Mat in,const int img_w,const int img_h, int w,int h,float scale,ncnn::Net& yolo11,std::vector<Recognition>& objects) {
    const float prob_threshold = 0.25f;
    const float nms_threshold = 0.45f;
    const int max_stride = 64;
    // pad to target_size rectangle
    int wpad = (w + max_stride - 1) / max_stride * max_stride - w;
    int hpad = (h + max_stride - 1) / max_stride * max_stride - h;
//    NCNN_LOGE("wpad = %d",wpad);
//    NCNN_LOGE("hpad = %d",hpad);
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2,
                           ncnn::BORDER_CONSTANT, 114.f);
    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in_pad.substract_mean_normalize(0, norm_vals);
//    NCNN_LOGE("in_pad.w = %d",in_pad.w);
//    NCNN_LOGE("in_pad.h = %d",in_pad.h);
//    NCNN_LOGE("in_pad.d = %d",in_pad.d);
//    NCNN_LOGE("in_pad.c = %d",in_pad.c);
//    NCNN_LOGE("in_pad.dims = %d",in_pad.dims);

    ncnn::Extractor ex = yolo11.create_extractor();
    ex.input("images", in_pad);

    std::vector<Recognition> proposals;

    // stride 8
    {
        ncnn::Mat out;
        ex.extract("/model.24/Transpose_output_0", out);
        ncnn::Mat anchors(6);
        anchors[0] = 12.f;
        anchors[1] = 16.f;
        anchors[2] = 19.f;
        anchors[3] = 36.f;
        anchors[4] = 40.f;
        anchors[5] = 28.f;

        std::vector<Recognition> objects8;
        generate111_proposals(anchors, 8, in_pad, out, prob_threshold, objects8);
        proposals.insert(proposals.end(), objects8.begin(), objects8.end());
    }

    // stride 16
    {
        ncnn::Mat out;
        ex.extract("/model.24/Transpose_1_output_0", out);
//        ncnn::Mat out2;
//        out2 =  out.reshape(40, 40, 72);
        ncnn::Mat anchors(6);
        anchors[0] = 36.f;
        anchors[1] = 75.f;
        anchors[2] = 76.f;
        anchors[3] = 55.f;
        anchors[4] = 72.f;
        anchors[5] = 146.f;

        std::vector<Recognition> objects16;
        generate111_proposals(anchors, 16, in_pad, out, prob_threshold, objects16);

        proposals.insert(proposals.end(), objects16.begin(), objects16.end());
    }

    // stride 32
    {
        ncnn::Mat out;
        ex.extract("/model.24/Transpose_2_output_0", out);
//        ncnn::Mat out3;
//        out3 =  out.reshape(20, 20, 72);
        ncnn::Mat anchors(6);
        anchors[0] = 142.f;
        anchors[1] = 110.f;
        anchors[2] = 192.f;
        anchors[3] = 243.f;
        anchors[4] = 459.f;
        anchors[5] = 401.f;

        std::vector<Recognition> objects32;
        generate111_proposals(anchors, 32, in_pad, out, prob_threshold, objects32);

        proposals.insert(proposals.end(), objects32.begin(), objects32.end());
    }

    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposals);

    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    NCNN_LOGE("proposals.size=%d", proposals.size());
    int count = picked.size();
    NCNN_LOGE("count = %d", count);

    objects.resize(count);
    for (int i = 0; i < count; i++) {
        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (objects[i].x - (wpad / 2)) / scale;
        float y0 = (objects[i].y - (hpad / 2)) / scale;
        float x1 = (objects[i].x + objects[i].width - (wpad / 2)) / scale;
        float y1 = (objects[i].y + objects[i].height - (hpad / 2)) / scale;

        // clip
        x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

        objects[i].x = x0;
        objects[i].y = y0;
        objects[i].width = x1 - x0;
        objects[i].height = y1 - y0;
    }

    return 0;
}

std::string Yolo5HandGes::get_label(int label) {
    return class_names[label];
}















