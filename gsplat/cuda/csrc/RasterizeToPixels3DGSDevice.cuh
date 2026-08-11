/*
 * SPDX-FileCopyrightText: Copyright 2025 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "Common.h"

// Per-(gaussian, pixel) alpha-blending math for 3DGS rasterization, shared by
// the dense and sparse rasterizers so the numerically sensitive forward and
// backward steps live in one place. These describe the contribution of a single
// gaussian to a single pixel; tile iteration, pixel addressing, shared-memory
// batching and warp reductions stay in the kernels.

namespace gsplat
{
constexpr float ANALYTIC_CDF_LINEAR     = 1.6f;
constexpr float ANALYTIC_CDF_CUBIC      = 0.07f;
constexpr float ANALYTIC_MIN_EIGENVALUE = 1e-12f;
constexpr float ANALYTIC_TWO_PI         = 6.283185307179586f;

template<RasterizeMode Mode>
struct GaussianRasterParams;

template<>
struct GaussianRasterParams<RasterizeMode::CLASSIC>
{
    vec3 conic;
};

template<>
struct GaussianRasterParams<RasterizeMode::ANALYTIC>
{
    vec3 conic;
    vec4 frame; // sin(theta), cos(theta), sigma_major, sigma_minor
};

template<RasterizeMode Mode>
__device__ __forceinline__ GaussianRasterParams<Mode> prepare_gaussian_raster_params(const vec3 &conic)
{
    if constexpr(Mode == RasterizeMode::CLASSIC)
    {
        return {conic};
    }
    else
    {
        const float center           = 0.5f * (conic.x + conic.z);
        const float half_diff        = 0.5f * (conic.x - conic.z);
        const float radius           = hypotf(half_diff, conic.y);
        const float eigenvalue_large = center + radius;
        const float determinant      = fmaf(conic.x, conic.z, -conic.y * conic.y);
        const float eigenvalue_small = fmaxf(determinant / eigenvalue_large, ANALYTIC_MIN_EIGENVALUE);
        const float theta            = 0.5f * atan2f(conic.y, half_diff);
        float sin_theta;
        float cos_theta;
        sincosf(theta, &sin_theta, &cos_theta);
        const float sigma_major = rsqrtf(eigenvalue_small);
        const float sigma_minor = rsqrtf(eigenvalue_large);
        return {
            conic, {sin_theta, cos_theta, sigma_major, sigma_minor}
        };
    }
}

__device__ __forceinline__ float analytic_normal_cdf(const float x)
{
    const float polynomial = ANALYTIC_CDF_LINEAR * x + ANALYTIC_CDF_CUBIC * x * x * x;
    return 0.5f * tanhf(0.5f * polynomial) + 0.5f;
}

__device__ __forceinline__ float eval_axis_integral_value(const float position, const float sigma)
{
    const float upper = (position + 0.5f) / sigma;
    const float lower = (position - 0.5f) / sigma;
    return sigma * (analytic_normal_cdf(upper) - analytic_normal_cdf(lower));
}

struct AxisIntegral
{
    float value;
    float v_position;
    float v_sigma;
};

__device__ __forceinline__ AxisIntegral eval_axis_integral(const float position, const float sigma)
{
    const float upper     = (position + 0.5f) / sigma;
    const float lower     = (position - 0.5f) / sigma;
    const float cdf_upper = analytic_normal_cdf(upper);
    const float cdf_lower = analytic_normal_cdf(lower);
    const float dcdf_upper
        = (ANALYTIC_CDF_LINEAR + 3.f * ANALYTIC_CDF_CUBIC * upper * upper) * cdf_upper * (1.0f - cdf_upper);
    const float dcdf_lower
        = (ANALYTIC_CDF_LINEAR + 3.f * ANALYTIC_CDF_CUBIC * lower * lower) * cdf_lower * (1.0f - cdf_lower);
    const float cdf_delta = cdf_upper - cdf_lower;
    return {
        sigma * cdf_delta,
        dcdf_upper - dcdf_lower,
        cdf_delta - upper * dcdf_upper + lower * dcdf_lower,
    };
}

template<RasterizeMode Mode>
__device__ __forceinline__ float eval_gaussian_response(
    const GaussianRasterParams<Mode> &params, const float dx, const float dy
)
{
    if constexpr(Mode == RasterizeMode::CLASSIC)
    {
        const vec3 conic  = params.conic;
        const float sigma = 0.5f * (conic.x * dx * dx + conic.z * dy * dy) + conic.y * dx * dy;
        return sigma < 0.f ? -1.f : __expf(-sigma);
    }
    else
    {
        const float sin_theta  = params.frame.x;
        const float cos_theta  = params.frame.y;
        const float u          = -sin_theta * dx + cos_theta * dy;
        const float v          = cos_theta * dx + sin_theta * dy;
        const float integral_u = eval_axis_integral_value(u, params.frame.z);
        const float integral_v = eval_axis_integral_value(v, params.frame.w);
        return ANALYTIC_TWO_PI * integral_u * integral_v;
    }
}

// Per-(gaussian, pixel) response shared by the forward and backward passes.
// `response` is a center sample in classic mode and a pixel-area integral in
// analytic mode; `valid` also applies the compositing alpha threshold.
struct GaussianWeight
{
    float response;
    float alpha;
    bool valid;
};

template<RasterizeMode Mode>
__device__ __forceinline__ GaussianWeight
    eval_gaussian_weight(const GaussianRasterParams<Mode> &params, const float dx, const float dy, const float opac)
{
    const float response = eval_gaussian_response(params, dx, dy);
    const float alpha    = min(MAX_ALPHA, opac * response);
    GaussianWeight out;
    out.response = response;
    out.alpha    = alpha;
    out.valid    = response >= 0.f && alpha >= ALPHA_THRESHOLD;
    return out;
}

__device__ __forceinline__ GaussianWeight
    eval_gaussian_weight(const vec3 &conic, const float dx, const float dy, const float opac)
{
    const auto params = prepare_gaussian_raster_params<RasterizeMode::CLASSIC>(conic);
    return eval_gaussian_weight<RasterizeMode::CLASSIC>(params, dx, dy, opac);
}

template<RasterizeMode Mode>
__device__ __forceinline__ void gaussian_response_vjp(
    const GaussianRasterParams<Mode> &params,
    const vec2 &delta,
    const float response,
    const float v_response,
    vec3 &v_conic,
    vec2 &v_delta
)
{
    if constexpr(Mode == RasterizeMode::CLASSIC)
    {
        const vec3 conic    = params.conic;
        const float v_sigma = -response * v_response;
        v_conic             = {
            0.5f * v_sigma * delta.x * delta.x,
            v_sigma * delta.x * delta.y,
            0.5f * v_sigma * delta.y * delta.y,
        };
        v_delta = {
            v_sigma * (conic.x * delta.x + conic.y * delta.y),
            v_sigma * (conic.y * delta.x + conic.z * delta.y),
        };
    }
    else
    {
        const vec3 conic          = params.conic;
        const float sin_theta     = params.frame.x;
        const float cos_theta     = params.frame.y;
        const float sigma_major   = params.frame.z;
        const float sigma_minor   = params.frame.w;
        const float u             = -sin_theta * delta.x + cos_theta * delta.y;
        const float v             = cos_theta * delta.x + sin_theta * delta.y;
        const AxisIntegral iu     = eval_axis_integral(u, sigma_major);
        const AxisIntegral iv     = eval_axis_integral(v, sigma_minor);
        const float v_u           = v_response * ANALYTIC_TWO_PI * iv.value * iu.v_position;
        const float v_v           = v_response * ANALYTIC_TWO_PI * iu.value * iv.v_position;
        const float v_sigma_major = v_response * ANALYTIC_TWO_PI * iv.value * iu.v_sigma;
        const float v_sigma_minor = v_response * ANALYTIC_TWO_PI * iu.value * iv.v_sigma;

        v_delta = {
            -sin_theta * v_u + cos_theta * v_v,
            cos_theta * v_u + sin_theta * v_v,
        };

        const float center             = 0.5f * (conic.x + conic.z);
        const float half_diff          = 0.5f * (conic.x - conic.z);
        const float radius_sq          = half_diff * half_diff + conic.y * conic.y;
        const float radius             = sqrtf(radius_sq);
        const float eigenvalue_large   = center + radius;
        const float determinant        = fmaf(conic.x, conic.z, -conic.y * conic.y);
        const float eigenvalue_small   = determinant / eigenvalue_large;
        const float v_eigenvalue_small = eigenvalue_small > ANALYTIC_MIN_EIGENVALUE
                                           ? -0.5f * sigma_major * sigma_major * sigma_major * v_sigma_major
                                           : 0.f;
        const float v_eigenvalue_large = -0.5f * sigma_minor * sigma_minor * sigma_minor * v_sigma_minor
                                       - v_eigenvalue_small * eigenvalue_small / eigenvalue_large;
        const float v_determinant      = v_eigenvalue_small / eigenvalue_large;

        float v_half_diff = 0.f;
        float v_offdiag   = 0.f;
        if(radius_sq > 1e-12f * center * center)
        {
            const float v_theta = -v * v_u + u * v_v;
            v_half_diff         = half_diff / radius * v_eigenvalue_large - 0.5f * conic.y / radius_sq * v_theta;
            v_offdiag           = conic.y / radius * v_eigenvalue_large + 0.5f * half_diff / radius_sq * v_theta;
        }

        v_conic.x = 0.5f * (v_eigenvalue_large + v_half_diff) + conic.z * v_determinant;
        v_conic.z = 0.5f * (v_eigenvalue_large - v_half_diff) + conic.x * v_determinant;
        v_conic.y = v_offdiag - 2.f * conic.y * v_determinant;
    }
}

// Forward: blend one gaussian into one pixel's running color/transmittance.
// Returns true when the pixel has saturated (transmittance fell to the
// threshold) and should stop accumulating -- the caller marks it done and the
// current gaussian is excluded, matching the dense kernel's behavior.
template<uint32_t CDIM>
__device__ __forceinline__ bool rasterize_to_pixels_3dgs_blend_fwd(
    const vec3 &conic,
    const vec3 &xy_opacity, // (mean.x, mean.y, opacity)
    const float px,
    const float py,
    const float *__restrict__ color_ptr, // colors + g * CDIM
    const uint32_t gaussian_idx,         // running index, stored as cur_idx on hit
    float &T,                            // transmittance, updated on accumulate
    float (&pix_out)[CDIM],              // per-channel accumulator, updated
    uint32_t &cur_idx                    // last contributing gaussian, updated
)
{
    const float opac        = xy_opacity.z;
    const float dx          = xy_opacity.x - px;
    const float dy          = xy_opacity.y - py;
    const GaussianWeight gw = eval_gaussian_weight(conic, dx, dy, opac);
    if(!gw.valid)
    {
        return false; // negligible contribution; pixel not done
    }
    const float alpha  = gw.alpha;
    const float next_T = T * (1.0f - alpha);
    if(next_T <= TRANSMITTANCE_THRESHOLD)
    {
        return true; // pixel saturated; exclusive of this gaussian
    }
    const float vis = alpha * T;
#pragma unroll
    for(uint32_t k = 0; k < CDIM; ++k)
    {
        pix_out[k] += color_ptr[k] * vis;
    }
    cur_idx = gaussian_idx;
    T       = next_T;
    return false;
}

// Backward: gradient contribution of one gaussian to one pixel. Walks the
// blend in reverse (the caller iterates gaussians back-to-front), updating the
// running transmittance `T` and color `buffer`, and producing this lane's local
// gradients. The caller zero-initializes the `*_local` outputs and reduces them
// across the warp before the atomic scatter. `compute_abs` mirrors absgrad.
template<uint32_t CDIM>
__device__ __forceinline__ void rasterize_to_pixels_3dgs_blend_bwd(
    const vec3 &conic,
    const vec2 &delta, // (mean.x - px, mean.y - py)
    const float opac,
    const float response,
    const float alpha,
    const float *__restrict__ rgbs,       // rgbs_batch + t * CDIM
    const float *__restrict__ v_render_c, // [CDIM]
    const float v_render_a,
    const float T_final,
    const float *__restrict__ backgrounds, // [CDIM] or nullptr
    const bool compute_abs,
    float &T,              // running transmittance, updated
    float (&buffer)[CDIM], // running color buffer, updated
    float (&v_rgb_local)[CDIM],
    vec3 &v_conic_local,
    vec2 &v_xy_local,
    vec2 &v_xy_abs_local,
    float &v_opacity_local
)
{
    // compute the current T for this gaussian
    const float ra   = 1.0f / fmaxf(MIN_ONE_MINUS_ALPHA, 1.0f - alpha);
    T               *= ra;
    // update v_rgb for this gaussian
    const float fac  = alpha * T;
#pragma unroll
    for(uint32_t k = 0; k < CDIM; ++k)
    {
        v_rgb_local[k] = fac * v_render_c[k];
    }
    // contribution from this pixel
    float v_alpha = 0.f;
#pragma unroll
    for(uint32_t k = 0; k < CDIM; ++k)
    {
        v_alpha += (rgbs[k] * T - buffer[k] * ra) * v_render_c[k];
    }
    v_alpha += T_final * ra * v_render_a;
    // contribution from background pixel
    if(backgrounds != nullptr)
    {
        float accum = 0.f;
#pragma unroll
        for(uint32_t k = 0; k < CDIM; ++k)
        {
            accum += backgrounds[k] * v_render_c[k];
        }
        v_alpha += -T_final * ra * accum;
    }
    if(opac * response <= MAX_ALPHA)
    {
        const auto params = prepare_gaussian_raster_params<RasterizeMode::CLASSIC>(conic);
        gaussian_response_vjp<RasterizeMode::CLASSIC>(
            params, delta, response, opac * v_alpha, v_conic_local, v_xy_local
        );
        if(compute_abs)
        {
            v_xy_abs_local = {abs(v_xy_local.x), abs(v_xy_local.y)};
        }
        v_opacity_local = response * v_alpha;
    }
#pragma unroll
    for(uint32_t k = 0; k < CDIM; ++k)
    {
        buffer[k] += rgbs[k] * fac;
    }
}
} // namespace gsplat
