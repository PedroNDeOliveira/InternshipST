/**
  ******************************************************************************
  * @file    pd_pp_model.c
  * @author  MDG Application Team
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2024 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */

#include "pd_model_pp_if.h"
#include "vision_models_pp.h"
#include "pd_pp_loc.h"
#include "pd_pp_tracking.h"

// Initialize Kalman filter
void kalman_init(kalman_filter_t *kf, float32_t x, float32_t y) {
    memset(kf, 0, sizeof(kalman_filter_t));
    kf->state[0] = x;
    kf->state[1] = y;

    // Initialize P as identity
    for (int32_t i = 0; i < 4; i++) kf->P[i][i] = 1.0f;
}

// Predict state and covariance
void kalman_predict(kalman_filter_t *kf) {
    float32_t *x = kf->state;
    float32_t (*P)[4] = kf->P;

    // State prediction
    x[0] += x[2] * dt;
    x[1] += x[3] * dt;

    // Covariance prediction: P = FPF' + Q
    float32_t F[4][4] = {
        {1, 0, dt, 0},
        {0, 1, 0, dt},
        {0, 0, 1, 0},
        {0, 0, 0, 1}
    };
    float32_t Q[4][4] = {0};
    for (int32_t i = 0; i < 4; i++) Q[i][i] = process_noise;

    float32_t FP[4][4] = {0}, FPF[4][4] = {0};
    for (int32_t i = 0; i < 4; i++)
        for (int32_t j = 0; j < 4; j++)
            for (int32_t k = 0; k < 4; k++)
                FP[i][j] += F[i][k] * P[k][j];

    for (int32_t i = 0; i < 4; i++)
        for (int32_t j = 0; j < 4; j++)
            for (int32_t k = 0; k < 4; k++)
                FPF[i][j] += FP[i][k] * F[j][k];  // FPF' == FPF^T

    for (int32_t i = 0; i < 4; i++)
        for (int32_t j = 0; j < 4; j++)
            P[i][j] = FPF[i][j] + Q[i][j];
}

// Update step with new (x, y) measurement
void kalman_update(kalman_filter_t *kf, float32_t z_x, float32_t z_y) {
    float32_t *x = kf->state;
    float32_t (*P)[4] = kf->P;

    float32_t H[2][4] = {
        {1, 0, 0, 0},
        {0, 1, 0, 0}
    };
    float32_t R[2][2] = {
        {measurement_noise, 0},
        {0, measurement_noise}
    };

    // Innovation y = z - Hx
    float32_t y[2] = {
        z_x - x[0],
        z_y - x[1]
    };

    // S = HPH^T + R
    float32_t S[2][2] = {0};
    for (int32_t i = 0; i < 2; i++)
        for (int32_t j = 0; j < 2; j++)
            for (int32_t k = 0; k < 4; k++)
                S[i][j] += H[i][k] * P[k][j] * H[j][k];

    S[0][0] += R[0][0]; S[1][1] += R[1][1];

    // K = PH^T * S^-1
    float32_t K[4][2] = {0};
    float32_t invS0 = 1.0f / S[0][0];
    float32_t invS1 = 1.0f / S[1][1];

    for (int32_t i = 0; i < 4; i++) {
        K[i][0] = P[i][0] * invS0;
        K[i][1] = P[i][1] * invS1;
    }

    // Update state: x = x + Ky
    for (int32_t i = 0; i < 4; i++) {
        x[i] += K[i][0] * y[0] + K[i][1] * y[1];
    }

    // Update covariance: P = (I - KH)P
    float32_t KH[4][4] = {0};
    for (int32_t i = 0; i < 4; i++)
        for (int32_t j = 0; j < 4; j++)
            KH[i][j] = K[i][0] * H[0][j] + K[i][1] * H[1][j];

    float32_t I_KH[4][4];
    for (int32_t i = 0; i < 4; i++)
        for (int32_t j = 0; j < 4; j++)
            I_KH[i][j] = (i == j ? 1.0f : 0.0f) - KH[i][j];

    float32_t newP[4][4] = {0};
    for (int32_t i = 0; i < 4; i++)
        for (int32_t j = 0; j < 4; j++)
            for (int32_t k = 0; k < 4; k++)
                newP[i][j] += I_KH[i][k] * P[k][j];

    memcpy(P, newP, sizeof(newP));
}
