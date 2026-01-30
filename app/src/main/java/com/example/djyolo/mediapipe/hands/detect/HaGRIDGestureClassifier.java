package com.example.djyolo.mediapipe.hands.detect;

import com.google.mediapipe.formats.proto.ClassificationProto;
import com.google.mediapipe.formats.proto.ClassificationProto.Classification;
import com.google.mediapipe.formats.proto.LandmarkProto;
import com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmark;
import com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmarkList;

import static com.example.djyolo.mediapipe.hands.detect.HandGestureFeatureExtractor.distance;


/**
 * @Author：daijun
 * @Description:
 * @CreateDate：2026/1/12 12:50
 */
public class HaGRIDGestureClassifier {

    // 手势判定阈值
    private static final float GESTURE_CONFIDENCE_THRESHOLD = 0.7f;

    /**
     * 识别 HaGRID 18个手势
     */
    public static GestureResult recognizeGesture(LandmarkProto.NormalizedLandmarkList landmarks,
                                                 ClassificationProto.Classification handedness) {
        if (landmarks == null) {
            return new GestureResult(HaGRIDGesture.NO_GESTURE, 0f, handedness);
        }

        // 提取特征
//        float[] features = HandGestureFeatureExtractor.extractFeatures(landmarks);

        // 使用规则方法识别（机器学习方法见后续）
        GestureResult result = recognizeByRules(landmarks, handedness);

        // 如果规则方法置信度低，尝试机器学习方法
//        if (result.confidence < GESTURE_CONFIDENCE_THRESHOLD) {
//            GestureResult mlResult = recognizeByMLModel(features, handedness);
//            if (mlResult.confidence > result.confidence) {
//                return mlResult;
//            }
//        }

        return result;
    }

    /**
     * 规则方法识别（快速但覆盖有限）
     */
    private static GestureResult recognizeByRules(LandmarkProto.NormalizedLandmarkList landmarks,
                                                  ClassificationProto.Classification handedness) {
        // 1. 检查握拳
        if (isFist(landmarks)) {
            return new GestureResult(HaGRIDGesture.FIST, 0.95f, handedness);
        }

        // 2. 检查手掌
        if (isPalm(landmarks)) {
            return new GestureResult(HaGRIDGesture.PALM, 0.92f, handedness);
        }

        // 3. 检查点赞/喜欢
        if (isLike(landmarks)) {
            return new GestureResult(HaGRIDGesture.LIKE, 0.90f, handedness);
        }

        // 4. 检查不喜欢
        if (isDislike(landmarks)) {
            return new GestureResult(HaGRIDGesture.DISLIKE, 0.88f, handedness);
        }

        // 5. 检查OK手势
        if (isOK(landmarks)) {
            return new GestureResult(HaGRIDGesture.OK, 0.93f, handedness);
        }

        // 6. 检查和平手势
        if (isPeace(landmarks)) {
            return new GestureResult(HaGRIDGesture.PEACE, 0.91f, handedness);
        }

        // 7. 检查数字手势
        GestureResult numberResult = recognizeNumber(landmarks);
        if (numberResult.gesture != HaGRIDGesture.NO_GESTURE) {
            return numberResult;
        }

        // 8. 检查打电话手势
        if (isCall(landmarks)) {
            return new GestureResult(HaGRIDGesture.CALL, 0.85f, handedness);
        }

        // 9. 检查摇滚手势
        if (isRock(landmarks)) {
            return new GestureResult(HaGRIDGesture.ROCK, 0.86f, handedness);
        }

        // 10. 检查停止手势
        if (isStop(landmarks)) {
            return new GestureResult(HaGRIDGesture.STOP, 0.89f, handedness);
        }

        // 11. 检查静音手势
        if (isMute(landmarks)) {
            return new GestureResult(HaGRIDGesture.MUTE, 0.87f, handedness);
        }

        return new GestureResult(HaGRIDGesture.NO_GESTURE, 0.1f, handedness);
    }

    // ========== 具体手势判断规则 ==========

    private static boolean isFist(LandmarkProto.NormalizedLandmarkList landmarks) {
        // 所有指尖都靠近手掌中心
        float palmCenterX = (landmarks.getLandmark(0).getX() +
                landmarks.getLandmark(9).getX()) / 2;
        float palmCenterY = (landmarks.getLandmark(0).getY() +
                landmarks.getLandmark(9).getY()) / 2;

        int[] fingertipIndices = {4, 8, 12, 16, 20};
        int closeCount = 0;

        for (int index : fingertipIndices) {
            LandmarkProto.NormalizedLandmark tip = landmarks.getLandmark(index);
            float distance = (float) Math.sqrt(
                    Math.pow(tip.getX() - palmCenterX, 2) +
                            Math.pow(tip.getY() - palmCenterY, 2)
            );

            if (distance < 0.15f) { // 阈值可调整
                closeCount++;
            }
        }

        return closeCount >= 4; // 至少4个手指靠近手掌
    }

    private static boolean isPalm(LandmarkProto.NormalizedLandmarkList landmarks) {
        // 所有手指都伸直且张开
        int[] baseIndices = {2, 5, 9, 13, 17};
        int[] midIndices = {3, 6, 10, 14, 18};
        int[] tipIndices = {4, 8, 12, 16, 20};

        int straightCount = 0;

        for (int i = 0; i < 5; i++) {
            float angle = HandGestureFeatureExtractor.calculateFingerAngle(
                    landmarks.getLandmark(baseIndices[i]),
                    landmarks.getLandmark(midIndices[i]),
                    landmarks.getLandmark(tipIndices[i])
            );

            if (angle < 30f) { // 手指较直
                straightCount++;
            }
        }

        // 检查手指是否张开
        float spread1 = distance(landmarks.getLandmark(8), landmarks.getLandmark(12)); // 食指-中指
        float spread2 = distance(landmarks.getLandmark(12), landmarks.getLandmark(16)); // 中指-无名指
        float spread3 = distance(landmarks.getLandmark(16), landmarks.getLandmark(20)); // 无名指-小指

        return straightCount >= 4 && spread1 > 0.05f && spread2 > 0.05f && spread3 > 0.05f;
    }

    private static boolean isLike(LandmarkProto.NormalizedLandmarkList landmarks) {
        // 拇指竖起，其他手指弯曲
        LandmarkProto.NormalizedLandmark thumbTip = landmarks.getLandmark(4);
        LandmarkProto.NormalizedLandmark thumbMCP = landmarks.getLandmark(2);
        LandmarkProto.NormalizedLandmark wrist = landmarks.getLandmark(0);

        // 拇指向上（Y坐标小于手腕）
        boolean thumbUp = thumbTip.getY() < wrist.getY();

        // 其他手指弯曲
        boolean otherFingersBent = true;
        int[] otherTipIndices = {8, 12, 16, 20};

        for (int index : otherTipIndices) {
            NormalizedLandmark tip = landmarks.getLandmark(index);
            NormalizedLandmark base = landmarks.getLandmark(index - 2);
            if (tip.getY() < base.getY()) { // 指尖高于指根，表示伸直
                otherFingersBent = false;
                break;
            }
        }

        return thumbUp && otherFingersBent;
    }

    private static boolean isDislike(NormalizedLandmarkList landmarks) {
        // 拇指向下，其他手指弯曲（与like相反）
        NormalizedLandmark thumbTip = landmarks.getLandmark(4);
        NormalizedLandmark thumbMCP = landmarks.getLandmark(2);
        NormalizedLandmark wrist = landmarks.getLandmark(0);

        // 拇指向下
        boolean thumbDown = thumbTip.getY() > wrist.getY();

        // 其他手指弯曲
        boolean otherFingersBent = true;
        int[] otherTipIndices = {8, 12, 16, 20};

        for (int index : otherTipIndices) {
            NormalizedLandmark tip = landmarks.getLandmark(index);
            NormalizedLandmark base = landmarks.getLandmark(index - 2);
            if (tip.getY() < base.getY()) {
                otherFingersBent = false;
                break;
            }
        }

        return thumbDown && otherFingersBent;
    }

    private static boolean isOK(NormalizedLandmarkList landmarks) {
        // 拇指和食指形成圆圈，其他手指伸直
        NormalizedLandmark thumbTip = landmarks.getLandmark(4);
        NormalizedLandmark indexTip = landmarks.getLandmark(8);

        // 拇指和食指尖距离很近
        float thumbIndexDistance = distance(thumbTip, indexTip);
        if (thumbIndexDistance > 0.08f) return false;

        // 其他手指伸直
        int[] otherTipIndices = {12, 16, 20};
        for (int index : otherTipIndices) {
            NormalizedLandmark tip = landmarks.getLandmark(index);
            NormalizedLandmark base = landmarks.getLandmark(index - 2);
            if (tip.getY() > base.getY()) { // 指尖低于指根，表示弯曲
                return false;
            }
        }

        return true;
    }

    private static boolean isPeace(NormalizedLandmarkList landmarks) {
        // 食指和中指伸直张开，其他手指弯曲
        NormalizedLandmark indexTip = landmarks.getLandmark(8);
        NormalizedLandmark middleTip = landmarks.getLandmark(12);
        NormalizedLandmark ringTip = landmarks.getLandmark(16);
        NormalizedLandmark pinkyTip = landmarks.getLandmark(20);

        // 食指和中指伸直
        boolean indexStraight = indexTip.getY() < landmarks.getLandmark(6).getY();
        boolean middleStraight = middleTip.getY() < landmarks.getLandmark(10).getY();

        // 其他手指弯曲
        boolean ringBent = ringTip.getY() > landmarks.getLandmark(14).getY();
        boolean pinkyBent = pinkyTip.getY() > landmarks.getLandmark(18).getY();
        boolean thumbBent = landmarks.getLandmark(4).getY() > landmarks.getLandmark(2).getY();

        // 食指和中指分开
        float vDistance = distance(indexTip, middleTip);

        return indexStraight && middleStraight &&
                ringBent && pinkyBent && thumbBent &&
                vDistance > 0.05f;
    }

    private static GestureResult recognizeNumber(NormalizedLandmarkList landmarks) {
        // 识别数字1-4
        int extendedFingers = countExtendedFingers(landmarks);

        switch (extendedFingers) {
            case 1:
                return new GestureResult(HaGRIDGesture.ONE, 0.9f, null);
            case 2:
                // 检查是普通two还是two_up
                if (isTwoUp(landmarks)) {
                    return new GestureResult(HaGRIDGesture.TWO_UP, 0.85f, null);
                }
                return new GestureResult(HaGRIDGesture.TWO_UP, 0.8f, null);
            case 3:
                return new GestureResult(HaGRIDGesture.THREE, 0.88f, null);
            case 4:
                return new GestureResult(HaGRIDGesture.FOUR, 0.87f, null);
            default:
                return new GestureResult(HaGRIDGesture.NO_GESTURE, 0f, null);
        }
    }

    private static int countExtendedFingers(NormalizedLandmarkList landmarks) {
        int count = 0;

        // 检查每个手指是否伸直
        for (int finger = 0; finger < 5; finger++) {
            if (isFingerExtended(landmarks, finger)) {
                count++;
            }
        }

        return count;
    }

    private static boolean isFingerExtended(LandmarkProto.NormalizedLandmarkList landmarks, int finger) {
        int baseIndex, midIndex, tipIndex;

        switch (finger) {
            case 0: // 拇指
                baseIndex = 2;
                midIndex = 3;
                tipIndex = 4;
                // 拇指判断逻辑不同
                return isThumbExtended(landmarks);
            case 1: // 食指
                baseIndex = 5;
                midIndex = 6;
                tipIndex = 8;
                break;
            case 2: // 中指
                baseIndex = 9;
                midIndex = 10;
                tipIndex = 12;
                break;
            case 3: // 无名指
                baseIndex = 13;
                midIndex = 14;
                tipIndex = 16;
                break;
            default: // 小指
                baseIndex = 17;
                midIndex = 18;
                tipIndex = 20;
                break;
        }

        // 指尖是否高于中间关节
        return landmarks.getLandmark(tipIndex).getY() < landmarks.getLandmark(midIndex).getY();
    }

    private static boolean isThumbExtended(LandmarkProto.NormalizedLandmarkList landmarks) {
        // 拇指特殊的判断逻辑
        LandmarkProto.NormalizedLandmark thumbTip = landmarks.getLandmark(4);
        LandmarkProto.NormalizedLandmark thumbIP = landmarks.getLandmark(3);
        LandmarkProto.NormalizedLandmark indexMCP = landmarks.getLandmark(5);

        // 拇指是否远离食指
        float distanceToIndex = distance(thumbTip, indexMCP);
        return distanceToIndex > 0.1f && thumbTip.getX() < thumbIP.getX();
    }

    private static boolean isTwoUp(LandmarkProto.NormalizedLandmarkList landmarks) {
        // 食指和中指向上，其他手指弯曲
        boolean indexExtended = isFingerExtended(landmarks, 1);
        boolean middleExtended = isFingerExtended(landmarks, 2);
        boolean ringBent = !isFingerExtended(landmarks, 3);
        boolean pinkyBent = !isFingerExtended(landmarks, 4);
        boolean thumbBent = !isThumbExtended(landmarks);

        return indexExtended && middleExtended && ringBent && pinkyBent && thumbBent;
    }

    private static boolean isCall(LandmarkProto.NormalizedLandmarkList landmarks) {
        // 拇指和小指伸直，其他手指弯曲
        boolean thumbExtended = isThumbExtended(landmarks);
        boolean pinkyExtended = isFingerExtended(landmarks, 4);
        boolean indexBent = !isFingerExtended(landmarks, 1);
        boolean middleBent = !isFingerExtended(landmarks, 2);
        boolean ringBent = !isFingerExtended(landmarks, 3);

        return thumbExtended && pinkyExtended &&
                indexBent && middleBent && ringBent;
    }

    private static boolean isRock(LandmarkProto.NormalizedLandmarkList landmarks) {
        // 食指和小指伸直，其他手指弯曲
        boolean indexExtended = isFingerExtended(landmarks, 1);
        boolean pinkyExtended = isFingerExtended(landmarks, 4);
        boolean middleBent = !isFingerExtended(landmarks, 2);
        boolean ringBent = !isFingerExtended(landmarks, 3);
        boolean thumbExtended = isThumbExtended(landmarks);

        return indexExtended && pinkyExtended &&
                middleBent && ringBent && thumbExtended;
    }

    private static boolean isStop(NormalizedLandmarkList landmarks) {
        // 手掌张开，手指并拢
        if (!isPalm(landmarks)) return false;

        // 检查手指是否并拢
        float spread1 = distance(landmarks.getLandmark(8), landmarks.getLandmark(12));
        float spread2 = distance(landmarks.getLandmark(12), landmarks.getLandmark(16));
        float spread3 = distance(landmarks.getLandmark(16), landmarks.getLandmark(20));

        return spread1 < 0.04f && spread2 < 0.04f && spread3 < 0.04f;
    }

    private static boolean isMute(NormalizedLandmarkList landmarks) {
        // 手掌姿势，但食指放在嘴唇位置（需要结合位置信息）
        // 这里简化为类似手掌但食指稍微弯曲
        if (!isPalm(landmarks)) return false;

        // 食指稍微弯曲
        float indexAngle = HandGestureFeatureExtractor.calculateFingerAngle(
                landmarks.getLandmark(5),
                landmarks.getLandmark(6),
                landmarks.getLandmark(8)
        );

        return indexAngle > 20f && indexAngle < 60f;
    }

    /**
     * 使用机器学习模型识别（需要集成TFLite模型）
     */
    private static GestureResult recognizeByMLModel(float[] features,
                                                    Classification handedness) {
        // 这里需要集成训练好的TFLite模型
        // 示例：使用简单的神经网络或SVM

        try {
            // 1. 加载TFLite模型
            // 2. 运行推理
            // 3. 解析结果

            // 临时返回无手势
            return new GestureResult(HaGRIDGesture.NO_GESTURE, 0.5f, handedness);
        } catch (Exception e) {
            e.printStackTrace();
            return new GestureResult(HaGRIDGesture.NO_GESTURE, 0f, handedness);
        }
    }

    // 手势结果类
    public static class GestureResult {
        public final HaGRIDGesture gesture;
        public final float confidence;
        public final Classification handedness;
        public final long timestamp;

        public GestureResult(HaGRIDGesture gesture, float confidence,
                             Classification handedness) {
            this.gesture = gesture;
            this.confidence = confidence;
            this.handedness = handedness;
            this.timestamp = System.currentTimeMillis();
        }
    }
}
