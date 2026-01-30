package com.example.djyolo.bean;

/**
 * @Author：daijun
 * @Description:
 * @CreateDate：2025/12/25 16:16
 */
public class NNDetObj {
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
