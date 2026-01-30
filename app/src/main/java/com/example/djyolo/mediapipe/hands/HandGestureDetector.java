package com.example.djyolo.mediapipe.hands;

import android.util.Log;

import com.google.mediapipe.formats.proto.LandmarkProto;

/**
 * @Author：daijun
 * @Description:
 * @CreateDate：2026/1/11 20:23
 */
public class HandGestureDetector {
    private static final String TAG = "HandGestureDetector";
    // 判断是否为"胜利"手势（食指和中指伸出）左手
    public static boolean isVictoryGesture(LandmarkProto.NormalizedLandmarkList landmarks) {
        // 获取关键点
        LandmarkProto.NormalizedLandmark wrist = landmarks.getLandmark(0);  // 手腕
        LandmarkProto.NormalizedLandmark indexTip = landmarks.getLandmark(8);  // 食指尖
        LandmarkProto.NormalizedLandmark middleTip = landmarks.getLandmark(12);  // 中指尖
        LandmarkProto.NormalizedLandmark ringTip = landmarks.getLandmark(16);  // 无名指尖
        LandmarkProto.NormalizedLandmark pinkyTip = landmarks.getLandmark(20);  // 小指尖

        // 判断食指和中指是否伸直（指尖高于指根）
        boolean indexStraight = indexTip.getY() < landmarks.getLandmark(6).getY();
        boolean middleStraight = middleTip.getY() < landmarks.getLandmark(10).getY();

        // 判断无名指和小指是否弯曲
        boolean ringBent = ringTip.getY() > landmarks.getLandmark(14).getY();
        boolean pinkyBent = pinkyTip.getY() > landmarks.getLandmark(18).getY();
        //dj左手，大拇指x坐标第4个节点比第2个节点大
        boolean thumbBent = landmarks.getLandmark(4).getX() > landmarks.getLandmark(2).getX();
        Log.d(TAG, "isVictoryGesture: indexStraight="+indexStraight+" middleStraight="+middleStraight+" ringBent="+ringBent
        +" pinkyBent"+pinkyBent+" thumbBent="+thumbBent);
        return indexStraight && middleStraight && ringBent && pinkyBent && thumbBent;
    }

    // 判断是否为"点赞"手势（拇指竖起）
    public static boolean isThumbsUp(LandmarkProto.NormalizedLandmarkList landmarks) {
        LandmarkProto.NormalizedLandmark thumbTip = landmarks.getLandmark(4);
        LandmarkProto.NormalizedLandmark wrist = landmarks.getLandmark(0);

        // 拇指在Y轴上高于手腕（表示竖起）
        boolean thumbUp = thumbTip.getY() < wrist.getY();

        // 其他手指弯曲
        boolean otherFingersBent = true;
        for (int i = 8; i <= 20; i += 4) {  // 检查其他四指的指尖
            LandmarkProto.NormalizedLandmark fingertip = landmarks.getLandmark(i);
            LandmarkProto.NormalizedLandmark fingerBase = landmarks.getLandmark(i - 2);
            if (fingertip.getY() < fingerBase.getY()) {  // 指尖高于指根表示伸直
                otherFingersBent = false;
                break;
            }
        }

        return thumbUp && otherFingersBent;
    }
}
