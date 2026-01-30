package com.example.djyolo.util;

import android.os.Handler;
import android.os.Looper;
import android.util.Log;

import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;


public class ThreadUtil {

    public static Handler getMainThreadHandler() {
        return mDefaultHandler;
    }

    private static final String TAG = "ThreadUtils";
    private static ExecutorService mExecutorService = Executors.newFixedThreadPool(
            Runtime.getRuntime().availableProcessors());
    private static ScheduledExecutorService mExecutorSingleService = Executors.newSingleThreadScheduledExecutor();
    private static Handler mDefaultHandler = new Handler(Looper.getMainLooper());
    private static ScheduledExecutorService mExecutorScheduledService = Executors.newScheduledThreadPool(10);

    public static ExecutorService getExecutorScheduledService() {
        return mExecutorScheduledService;
    }

    public static ExecutorService getmExecutorService() {
        return mExecutorService;
    }

    public static void runInThreadPool(final Runnable task) {
        try {
            mExecutorService.submit(task);
        } catch (Exception e) {
            Log.e(TAG, "runInThreadPool: "+e);
        }
    }

    public static void runInUIThread(Runnable task) {
        mDefaultHandler.post(task);
    }

    public static void runInUIThreadDelayed(Runnable task, long delayMillis) {
        mDefaultHandler.postDelayed(task, delayMillis);
    }

    public static void runInSingleThreadPool(Runnable task) {
        mExecutorSingleService.execute(task);//submit 与execute区别，see http://blog.csdn.net/hayre/article/details/53314599
    }

    public static void runInThreadPoolDelayed(Runnable task, long delay, TimeUnit delayUnits) {
        mExecutorScheduledService.schedule(task, delay, delayUnits);
    }
    public static List<byte[]> mMemory;

}