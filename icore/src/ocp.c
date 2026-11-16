/*
 * Riccati primal-dual interior-point method for convex conic OCPs.
 *
 * Arrays are row-major.  The three numerical layers are deliberately separate:
 * cone algebra, an equality-constrained Riccati solve, and the IPM iteration.
 * All working storage is allocated by create/update, never by solve.
 */
#include "icore_ocp.h"
#include "driver.h"
#include "dense_la.h"
#include "ocp_la.h"
#include "mixed_factor.h"
#include "orthant.h"
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if defined(__AVX2__) && !defined(ICORE_OCP_SCALAR)
#include <immintrin.h>
#endif

#if defined(__GNUC__) || defined(__clang__)
#define ICORE_OCP_INLINE static inline __attribute__((always_inline))
#define ICORE_OCP_RESTRICT __restrict__
#else
#define ICORE_OCP_INLINE static inline
#define ICORE_OCP_RESTRICT
#endif

#ifndef ICORE_OCP_DYNAMICS_CACHE
#define ICORE_OCP_DYNAMICS_CACHE 1
#endif

/* Read the shared iteration in driver_loop.h first, then the OCP-specific layers below.
 * These private headers form one translation unit, preserving cross-function
 * optimization and keeping state ownership local to this solver. */
#include "ocp_workspace.h"
#include "ocp_vectors.h"
#include "ocp_cones.h"
#include "ocp_factor.h"
#include "ocp_dual_recovery.h"
#include "ocp_newton.h"
#include "ocp_initialize.h"
#include "ocp_model.h"
#include "ocp_driver.h"
