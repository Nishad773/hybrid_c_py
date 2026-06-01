#include <math.h>
#include <stdio.h>
#include "conv2d.h"

#define N 10
#define IN 28
#define C 8
#define CONV 26
#define POOL 13
#define FLAT 1352

static float relu(float x) { return x > 0.0f ? x : 0.0f; }

static int predict(float* img, float* kernels, float* cbias, float* w, float* b) {
    float conv[C * CONV * CONV], pool[FLAT], prob[10], mx = -1e30f, sum = 0.0f;
    conv2d_forward(img, kernels, cbias, conv, 1, IN, IN, C, 3, 3);

    for (int c = 0; c < C; c++) {
        for (int y = 0; y < POOL; y++) {
            for (int x = 0; x < POOL; x++) {
                int ci = c * CONV * CONV, pi = c * POOL * POOL;
                float m = relu(conv[ci + (2 * y) * CONV + 2 * x]);
                float a = relu(conv[ci + (2 * y) * CONV + 2 * x + 1]);
                float d = relu(conv[ci + (2 * y + 1) * CONV + 2 * x]);
                float e = relu(conv[ci + (2 * y + 1) * CONV + 2 * x + 1]);
                if (a > m) m = a; if (d > m) m = d; if (e > m) m = e;
                pool[pi + y * POOL + x] = m;
            }
        }
    }

    for (int o = 0; o < 10; o++) {
        prob[o] = b[o];
        for (int i = 0; i < FLAT; i++) prob[o] += pool[i] * w[o * FLAT + i];
        if (prob[o] > mx) mx = prob[o];
    }
    for (int o = 0; o < 10; o++) { prob[o] = expf(prob[o] - mx); sum += prob[o]; }
    int best = 0;
    for (int o = 1; o < 10; o++) if (prob[o] / sum > prob[best] / sum) best = o;
    return best;
}

int main() {
    float kernels[C * 9], cbias[C], w[10 * FLAT], b[10], img[IN * IN];
    FILE *wf = fopen("weights.bin", "rb"), *sf = fopen("samples.bin", "rb"), *log = fopen("inference_log.txt", "w");
    if (!wf || !sf || !log) return 1;
    if (fread(kernels, sizeof(float), C * 9, wf) != C * 9) return 1;
    if (fread(cbias, sizeof(float), C, wf) != C) return 1;
    if (fread(w, sizeof(float), 10 * FLAT, wf) != 10 * FLAT) return 1;
    if (fread(b, sizeof(float), 10, wf) != 10) return 1;
    fclose(wf);

    int ok = 1;
    for (int s = 0; s < N; s++) {
        int idx, py, label;
        if (fread(&idx, sizeof(int), 1, sf) != 1) return 1;
        if (fread(&py, sizeof(int), 1, sf) != 1) return 1;
        if (fread(&label, sizeof(int), 1, sf) != 1) return 1;
        if (fread(img, sizeof(float), IN * IN, sf) != IN * IN) return 1;
        int cp = predict(img, kernels, cbias, w, b);
        if (cp != py) ok = 0;
        printf("Sample %d\nPython Prediction: %d\nC Prediction: %d\nTrue Label: %d\n\n", idx, py, cp, label);
        fprintf(log, "Sample %d\nPython Prediction: %d\nC Prediction: %d\nTrue Label: %d\n\n", idx, py, cp, label);
    }
    fclose(sf); fclose(log);
    return ok ? 0 : 2;
}
