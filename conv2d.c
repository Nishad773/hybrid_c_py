#include "conv2d.h"

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
) {
    int out_h = in_h - k_h + 1;
    int out_w = in_w - k_w + 1;

    for (int n = 0; n < batch; n++) {
        for (int c = 0; c < out_channels; c++) {
            for (int y = 0; y < out_h; y++) {
                for (int x = 0; x < out_w; x++) {
                    float sum = bias[c];
                    for (int ky = 0; ky < k_h; ky++) {
                        for (int kx = 0; kx < k_w; kx++) {
                            int ii = n * in_h * in_w + (y + ky) * in_w + x + kx;
                            int ki = c * k_h * k_w + ky * k_w + kx;
                            sum += input[ii] * kernels[ki];
                        }
                    }
                    output[((n * out_channels + c) * out_h + y) * out_w + x] = sum;
                }
            }
        }
    }
}
