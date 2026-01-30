package com.example.djyolo.util;

import android.graphics.RectF;


import com.example.djyolo.detector.tflite.Recognition;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.stream.Collectors;

/**
 * @Author：
 * @Description:
 * @CreateDate：
 */
public class PostProcessUtils {
    public static void qsortDescentInplace(ArrayList<Recognition> objects, int left, int right) {
        int i = left, j = right;
        float pivot = objects.get((left + right) / 2).getLabelScore();

        while (i <= j) {
            while (objects.get(i).getLabelScore() > pivot) i++;
            while (objects.get(j).getLabelScore() < pivot) j--;
            if (i <= j) {
                // Swap
                Collections.swap(objects, i, j);
                i++;
                j--;
            }
        }

        if (left < j) qsortDescentInplace(objects, left, j);
        if (i < right) qsortDescentInplace(objects, i, right);
    }

    public static void qsortDescentInplace(ArrayList<Recognition> objects) {
        if (!objects.isEmpty()) {
            qsortDescentInplace(objects, 0, objects.size() - 1);
        }
    }


    private static float intersectionArea(RectF a, RectF b) {
        if (!RectF.intersects(a, b)) return 0;

        float left = Math.max(a.left, b.left);
        float right = Math.min(a.right, b.right);
        float top = Math.max(a.top, b.top);
        float bottom = Math.min(a.bottom, b.bottom);

        return Math.max(0, right - left) * Math.max(0, bottom - top);
    }

    // Non-Maximum Suppression (NMS)
    public static void nmsSortedBboxes(ArrayList<Recognition> objects, ArrayList<Recognition> picked, float nmsThreshold) {
        picked.clear();

        int n = objects.size();
        ArrayList<Float> areas = new ArrayList<>(n);

        for (Recognition obj : objects) {
            areas.add(obj.getLocation().width() * obj.getLocation().height());
        }

        for (int i = 0; i < n; i++) {
            Recognition a = objects.get(i);
            boolean keep = true;

            for (Recognition b : picked) {
//                Recognition b = objects.get(j);
                Float areaB = b.getLocation().width() * b.getLocation().height();
                // Intersection over Union (IoU)
                float interArea = intersectionArea(a.getLocation(), b.getLocation());
                float unionArea = areas.get(i) + areaB - interArea;
                if (interArea / unionArea > nmsThreshold) {
                    keep = false;
                    break;
                }
            }

            if (keep) picked.add(a);
        }
    }



    private static float calculateIoU(Recognition box1, Recognition box2) {
        float intersectionArea = intersectionArea(box1.getLocation(), box2.getLocation());
        float box1Area = box1.getLocation().width() * box1.getLocation().height();
        float box2Area = box2.getLocation().width() * box2.getLocation().height();
        return intersectionArea / (box1Area + box2Area - intersectionArea);
    }

    public static ArrayList<Recognition> applyNMS(ArrayList<Recognition> boxes,float IOU_THRESHOLD) {
        List<Recognition> sortedBoxes = boxes.stream()
                .sorted((b1, b2) -> Float.compare(b2.getLabelScore(), b1.getLabelScore()))
                .collect(Collectors.toList());

        ArrayList<Recognition> selectedBoxes = new ArrayList<>();

        while (!sortedBoxes.isEmpty()) {
            Recognition first = sortedBoxes.get(0);
            selectedBoxes.add(first);
            sortedBoxes.remove(first);

            Iterator<Recognition> iterator = sortedBoxes.iterator();
            while (iterator.hasNext()) {
                Recognition nextBox = iterator.next();
                float iou = calculateIoU(first, nextBox);
                if (iou >= IOU_THRESHOLD) {
                    iterator.remove();
                }
            }
        }

        return selectedBoxes;
    }



}
