package com.example.djyolo.mediapipe.hands.detect;

/**
 * @Author：daijun
 * @Description:
 * @CreateDate：2026/1/12 13:01
 */
public enum HaGRIDGesture {
    NO_GESTURE(0, "no_gesture"),      // 无手势
    CALL(1, "call"),                  // 打电话手势
    DISLIKE(2, "dislike"),            // 不喜欢/拇指向下
    FIST(3, "fist"),                  // 握拳
    FOUR(4, "four"),                  // 数字4
    LIKE(5, "like"),                  // 喜欢/点赞
    OK(6, "ok"),                      // OK手势
    ONE(7, "one"),                    // 数字1
    PALM(8, "palm"),                   // 手掌张开
    PEACE(9, "peace"),                // 和平/胜利手势
    ROCK(10, "rock"),                 // 摇滚手势
    STOP(11, "stop"),                 // 停止手势
    STOP_INVERTED(12, "stop_inverted"), // 反手停止
    THREE(13, "three"),               // 数字3
    TWO_UP(14, "two_up"),             // 双指向上
    TWO_UP_INVERTED(15, "two_up_inverted"), // 反手双指向上
    THREE2(16, "three2"),             // 另一种数字3手势
    PEACE_INVERTED(17, "peace_inverted"),  // 反手和平手势
    MUTE(18, "mute");                 // 静音手势

    private final int id;
    private final String name;

    HaGRIDGesture(int id, String name) {
        this.id = id;
        this.name = name;
    }
}
