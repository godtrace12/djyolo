package com.example.djyolo.mediapipe.hands.detect;
import com.google.mediapipe.formats.proto.ClassificationProto;
import com.google.mediapipe.formats.proto.ClassificationProto.Classification;
import com.google.mediapipe.formats.proto.LandmarkProto;
import com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmark;
import com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmarkList;

import java.util.ArrayList;
import java.util.List;

/**
 * @Author：daijun
 * @Description:
 * @CreateDate：2026/1/12 13:00
 */
public class HandGestureFeatureExtractor {

    /**
     * 提取手势特征向量（基于21个关键点）
     */
    public static float[] extractFeatures(NormalizedLandmarkList landmarks) {
        List<Float> features = new ArrayList<>();

        // 1. 手指伸直角度特征
        features.addAll(getFingerStraightness(landmarks));

        // 2. 指尖到手掌距离
        features.addAll(getFingerToPalmDistances(landmarks));

        // 3. 手指间角度
        features.addAll(getFingerAngles(landmarks));

        // 4. 手掌张开程度
        features.addAll(getPalmOpenness(landmarks));

        // 5. 关键点相对位置
        features.addAll(getKeyPointPositions(landmarks));

        return convertToFloatArray(features);
    }

    /**
     * 判断手指是否伸直
     */
    private static List<Float> getFingerStraightness(NormalizedLandmarkList landmarks) {
        List<Float> straightness = new ArrayList<>();

        // 五指：拇指、食指、中指、无名指、小指
        for (int finger = 0; finger < 5; finger++) {
            int baseIndex, midIndex, tipIndex;

            switch (finger) {
                case 0: // 拇指
                    baseIndex = 2; // THUMB_MCP
                    midIndex = 3;  // THUMB_IP
                    tipIndex = 4;  // THUMB_TIP
                    break;
                case 1: // 食指
                    baseIndex = 5; // INDEX_MCP
                    midIndex = 6;  // INDEX_PIP
                    tipIndex = 8;  // INDEX_TIP
                    break;
                case 2: // 中指
                    baseIndex = 9;  // MIDDLE_MCP
                    midIndex = 10; // MIDDLE_PIP
                    tipIndex = 12; // MIDDLE_TIP
                    break;
                case 3: // 无名指
                    baseIndex = 13; // RING_MCP
                    midIndex = 14;  // RING_PIP
                    tipIndex = 16;  // RING_TIP
                    break;
                default: // 小指
                    baseIndex = 17; // PINKY_MCP
                    midIndex = 18;  // PINKY_PIP
                    tipIndex = 20;  // PINKY_TIP
                    break;
            }

            // 计算角度：指尖是否在指根连线的延长线上
            float angle = calculateFingerAngle(
                    landmarks.getLandmark(baseIndex),
                    landmarks.getLandmark(midIndex),
                    landmarks.getLandmark(tipIndex)
            );

            // 角度越小越直（0表示完全伸直）
            straightness.add(angle);
        }

        return straightness;
    }

    /**
     * 计算指尖到手掌中心的距离
     */
    private static List<Float> getFingerToPalmDistances(NormalizedLandmarkList landmarks) {
        List<Float> distances = new ArrayList<>();

        // 计算手掌中心（手腕和MCP点的中心）
        float palmCenterX = (landmarks.getLandmark(0).getX() +
                landmarks.getLandmark(9).getX()) / 2;
        float palmCenterY = (landmarks.getLandmark(0).getY() +
                landmarks.getLandmark(9).getY()) / 2;

        // 五指指尖
        int[] fingertipIndices = {4, 8, 12, 16, 20};

        for (int index : fingertipIndices) {
            NormalizedLandmark tip = landmarks.getLandmark(index);
            float distance = (float) Math.sqrt(
                    Math.pow(tip.getX() - palmCenterX, 2) +
                            Math.pow(tip.getY() - palmCenterY, 2)
            );
            distances.add(distance);
        }

        return distances;
    }

    /**
     * 计算手指间角度（用于区分不同数字手势）
     */
    private static List<Float> getFingerAngles(NormalizedLandmarkList landmarks) {
        List<Float> angles = new ArrayList<>();

        // 食指和中指角度（用于区分V字手势）
        angles.add(calculateAngleBetweenFingers(
                landmarks.getLandmark(5), landmarks.getLandmark(8),  // 食指
                landmarks.getLandmark(9), landmarks.getLandmark(12)  // 中指
        ));

        // 中指和无名指角度
        angles.add(calculateAngleBetweenFingers(
                landmarks.getLandmark(9), landmarks.getLandmark(12),  // 中指
                landmarks.getLandmark(13), landmarks.getLandmark(16)  // 无名指
        ));

        // 拇指和食指角度（用于OK手势）
        angles.add(calculateAngleBetweenFingers(
                landmarks.getLandmark(2), landmarks.getLandmark(4),  // 拇指
                landmarks.getLandmark(5), landmarks.getLandmark(8)   // 食指
        ));

        return angles;
    }

    /**
     * 计算手掌张开程度
     */
    private static List<Float> getPalmOpenness(NormalizedLandmarkList landmarks) {
        List<Float> openness = new ArrayList<>();

        // 手掌宽度（小指MCP到食指MCP）
        float palmWidth = distance(
                landmarks.getLandmark(5),  // INDEX_MCP
                landmarks.getLandmark(17)  // PINKY_MCP
        );

        // 手掌长度（中指MCP到手腕）
        float palmLength = distance(
                landmarks.getLandmark(0),  // WRIST
                landmarks.getLandmark(9)   // MIDDLE_MCP
        );

        // 手掌面积近似
        openness.add(palmWidth);
        openness.add(palmLength);
        openness.add(palmWidth * palmLength); // 近似面积

        return openness;
    }

    /**
     * 关键点相对位置特征
     */
    private static List<Float> getKeyPointPositions(NormalizedLandmarkList landmarks) {
        List<Float> positions = new ArrayList<>();

        // 手腕位置作为参考点
        float wristX = landmarks.getLandmark(0).getX();
        float wristY = landmarks.getLandmark(0).getY();

        // 所有关键点相对于手腕的位置
        for (int i = 1; i < 21; i++) {
            positions.add(landmarks.getLandmark(i).getX() - wristX);
            positions.add(landmarks.getLandmark(i).getY() - wristY);
        }

        return positions;
    }

    // ========== 辅助计算函数 ==========

    public static float calculateFingerAngle(NormalizedLandmark base,
                                              NormalizedLandmark mid,
                                              NormalizedLandmark tip) {
        // 向量1: base -> mid
        float v1x = mid.getX() - base.getX();
        float v1y = mid.getY() - base.getY();

        // 向量2: mid -> tip
        float v2x = tip.getX() - mid.getX();
        float v2y = tip.getY() - mid.getY();

        // 计算夹角
        float dot = v1x * v2x + v1y * v2y;
        float mag1 = (float) Math.sqrt(v1x * v1x + v1y * v1y);
        float mag2 = (float) Math.sqrt(v2x * v2x + v2y * v2y);

        if (mag1 == 0 || mag2 == 0) return 0;

        float cosAngle = Math.max(-1, Math.min(1, dot / (mag1 * mag2)));
        return (float) Math.toDegrees(Math.acos(cosAngle));
    }

    private static float calculateAngleBetweenFingers(NormalizedLandmark base1, NormalizedLandmark tip1,
                                                      NormalizedLandmark base2, NormalizedLandmark tip2) {
        // 计算两个指尖连线的角度
        float v1x = tip1.getX() - base1.getX();
        float v1y = tip1.getY() - base1.getY();

        float v2x = tip2.getX() - base2.getX();
        float v2y = tip2.getY() - base2.getY();

        float angle1 = (float) Math.toDegrees(Math.atan2(v1y, v1x));
        float angle2 = (float) Math.toDegrees(Math.atan2(v2y, v2x));

        float diff = Math.abs(angle1 - angle2) % 360;
        return diff > 180 ? 360 - diff : diff;
    }

    public static float distance(NormalizedLandmark p1, NormalizedLandmark p2) {
        return (float) Math.sqrt(
                Math.pow(p1.getX() - p2.getX(), 2) +
                        Math.pow(p1.getY() - p2.getY(), 2)
        );
    }

    private static float[] convertToFloatArray(List<Float> list) {
        float[] array = new float[list.size()];
        for (int i = 0; i < list.size(); i++) {
            array[i] = list.get(i);
        }
        return array;
    }
}
