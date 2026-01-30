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

package com.example.djyolo.detector;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.view.Surface;
import com.example.djyolo.bean.NNDetObj;


public class YOLO11Ncnn
{
    public class Obj
    {
        public float x;
        public float y;
        public float w;
        public float h;
        public String label;
        public float prob;
        @Override
        public String toString()
        {
            return label + " = " + prob+" x="+x+" y="+y+" w="+w+" h="+h;
        }
    }

    public native NNDetObj[] Detect(Bitmap bitmap, boolean use_gpu);
    public native boolean loadModel(AssetManager mgr, String model, String param, int cpugpu,int taskid);

    static {
        System.loadLibrary("yolo11ncnn");
    }
}
