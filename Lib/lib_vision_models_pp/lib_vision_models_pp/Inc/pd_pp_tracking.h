/**
  ******************************************************************************
  * @file    pd_pp_loc.h
  * @author  MDG Application Team
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2024 STMicroelectronics.
 * All rights reserved.
 *
 * This software is licensed under terms that can be found in the LICENSE file in
 * the root directory of this software component.
 * If no LICENSE file comes with this software, it is provided AS-IS.
 *
******************************************************************************
*/

#ifndef _PD_PP_TRACK_
#define _PD_PP_TRACK_

#ifdef __cplusplus
 extern "C" {
#endif

#include "arm_math.h"

#define AI_PD_MODEL_PP_MAX_BOXES_LIMIT 20
#define MAX_FRAME_MISSES 15

 // Kalman filter
 typedef struct {
     float32_t state[4];   // [x, y, vx, vy]
     float32_t P[4][4];    // Covariance matrix
 } kalman_filter_t;

typedef struct {
	 pd_pp_box_t box;
	 int missed_frames;
	 kalman_filter_t kf;
}tracked_box_t;


//typedef struct {
//	pd_pp_box_t box;
//	int32_t missed_frames;
//}tracked_box_t

static const float32_t dt = 1.0f;  // time step
static const float32_t process_noise = 1e-2f;
static const float32_t measurement_noise = 1e-1f;

// Kalman filter motion prediction tracking.

// Initialize Kalman filter
void kalman_init(kalman_filter_t *kf, float32_t x, float32_t y);

// Predict state and covariance
void kalman_predict(kalman_filter_t *kf);

// Update step with new (x, y) measurement
void kalman_update(kalman_filter_t *kf, float32_t z_x, float32_t z_y);

#ifdef __cplusplus
 }
#endif

#endif // _PD_PP_OUTPUT_IF_
