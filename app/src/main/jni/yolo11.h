// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2025 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#ifndef YOLO11_H
#define YOLO11_H

#include <net.h>


struct Recognition
{
    float x, y, width, height;
    int label;
    float prob;
    int gindex;
//    std::vector<KeyPoint> keypoints;
};

class YOLO11
{
public:
    virtual ~YOLO11();

    int load(const char* parampath, const char* modelpath, bool use_gpu = false);
    int load(AAssetManager* mgr, const char* parampath, const char* modelpath, bool use_gpu = false);

    void set_det_target_size(int target_size);
    int get_det_target_size(){return det_target_size;}
    ncnn::Net& getNet(){return yolo11;}

    virtual std::string get_label(int label) = 0;
    virtual int detect(ncnn::Mat in,const int img_w,const int img_h, int w,int h,float scale,ncnn::Net& yolo11,std::vector<Recognition>& objects) = 0;


protected:
    ncnn::Net yolo11;
    int det_target_size;
    ncnn::UnlockedPoolAllocator blob_pool_allocator;
    ncnn::PoolAllocator workspace_pool_allocator;
};


class YOLO11_stripcv_det : public YOLO11
{
public:
    virtual std::string get_label(int label);
    virtual int detect(ncnn::Mat imgIn, const int img_w,const int img_h, int w,int h,float scale,ncnn::Net& yolo11,std::vector<Recognition>& objects);
};


class Yolo5HandGes : public YOLO11
{
public:
    virtual std::string get_label(int label);
    virtual int detect(ncnn::Mat in,const int img_w,const int img_h, int w,int h,float scale,ncnn::Net& yolo11,std::vector<Recognition>& objects);

};

#endif // YOLO11_H
