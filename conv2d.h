#ifndef CONV2D_H
#define CONV2D_H

void conv2d_forward(
    float* input,
    float* kernels,
    float* bias,
    float* output,
    int batch,
    int in_h,
    int in_w,
    int out_channels,
    int k_h,
    int k_w
);

#endif
