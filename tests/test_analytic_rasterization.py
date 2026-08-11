# SPDX-FileCopyrightText: Copyright 2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

import gsplat
from gsplat.cuda._backend import _C

if _C is None:
    pytest.skip("gsplat CUDA extension not available", allow_module_level=True)


def _analytic_reference(means2d, conics, colors, opacities, width, height):
    mean = means2d.reshape(-1, 2)[0]
    conic = conics.reshape(-1, 3)[0]
    color = colors.reshape(-1, colors.shape[-1])[0]
    opacity = opacities.reshape(-1)[0]

    center = 0.5 * (conic[0] + conic[2])
    half_diff = 0.5 * (conic[0] - conic[2])
    radius = torch.hypot(half_diff, conic[1])
    theta = 0.5 * torch.atan2(conic[1], half_diff)
    sigma_major = torch.rsqrt(center - radius)
    sigma_minor = torch.rsqrt(center + radius)

    ys, xs = torch.meshgrid(
        torch.arange(height, device=mean.device, dtype=mean.dtype) + 0.5,
        torch.arange(width, device=mean.device, dtype=mean.dtype) + 0.5,
        indexing="ij",
    )
    dx = mean[0] - xs
    dy = mean[1] - ys
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    u = -sin_theta * dx + cos_theta * dy
    v = cos_theta * dx + sin_theta * dy

    def integrate_axis(position, sigma):
        upper = (position + 0.5) / sigma
        lower = (position - 0.5) / sigma

        def cdf(x):
            return 0.5 * torch.tanh(0.5 * (1.6 * x + 0.07 * x**3)) + 0.5

        return sigma * (cdf(upper) - cdf(lower))

    response = (
        2.0 * math.pi * integrate_axis(u, sigma_major) * integrate_axis(v, sigma_minor)
    )
    alpha = torch.clamp_max(opacity * response, 0.99)
    alpha = torch.where(alpha >= 1.0 / 255.0, alpha, 0.0)
    return (alpha[..., None] * color)[None], alpha[None, ..., None]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgs(), reason="3DGS support isn't built in")
@pytest.mark.parametrize("packed", [False, True])
def test_analytic_rasterization_matches_reference(packed):
    from gsplat.cuda._wrapper import rasterize_to_pixels

    width = height = 8
    means2d = torch.tensor([[[3.7, 4.2]]], device="cuda")
    conics = torch.tensor([[[4.0, 0.6, 1.5]]], device="cuda")
    colors = torch.tensor([[[0.2, 0.5, 0.8]]], device="cuda")
    opacities = torch.tensor([[0.55]], device="cuda")
    if packed:
        means2d = means2d.flatten(0, 1)
        conics = conics.flatten(0, 1)
        colors = colors.flatten(0, 1)
        opacities = opacities.flatten(0, 1)

    inputs = [
        tensor.requires_grad_() for tensor in (means2d, conics, colors, opacities)
    ]
    reference_inputs = [tensor.detach().clone().requires_grad_() for tensor in inputs]
    isect_offsets = torch.zeros((1, 1, 1), dtype=torch.int32, device="cuda")
    flatten_ids = torch.zeros((1,), dtype=torch.int32, device="cuda")

    rendered = rasterize_to_pixels(
        *inputs,
        width,
        height,
        16,
        isect_offsets,
        flatten_ids,
        packed=packed,
        rasterize_mode="analytic",
    )
    reference = _analytic_reference(*reference_inputs, width, height)
    for actual, expected in zip(rendered, reference):
        torch.testing.assert_close(actual, expected, rtol=5e-4, atol=5e-5)

    torch.manual_seed(42)
    cotangents = [torch.randn_like(output) for output in rendered]
    gradients = torch.autograd.grad(
        sum(
            (output * cotangent).sum()
            for output, cotangent in zip(rendered, cotangents)
        ),
        inputs,
    )
    reference_gradients = torch.autograd.grad(
        sum(
            (output * cotangent).sum()
            for output, cotangent in zip(reference, cotangents)
        ),
        reference_inputs,
    )
    for actual, expected in zip(gradients, reference_gradients):
        torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgs(), reason="3DGS support isn't built in")
def test_high_level_analytic_mode_uses_analytic_pixel_rasterizer():
    from gsplat.cuda._wrapper import rasterize_to_pixels
    from gsplat.rendering import rasterization

    width = height = 8
    colors = torch.tensor([[0.2, 0.5, 0.8]], device="cuda")
    render_colors, render_alphas, meta = rasterization(
        means=torch.tensor([[0.0, 0.0, 3.0]], device="cuda"),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda"),
        scales=torch.tensor([[0.02, 0.01, 0.01]], device="cuda"),
        opacities=torch.tensor([0.55], device="cuda"),
        colors=colors,
        viewmats=torch.eye(4, device="cuda")[None],
        Ks=torch.tensor(
            [[[100.0, 0.0, 4.0], [0.0, 100.0, 4.0], [0.0, 0.0, 1.0]]],
            device="cuda",
        ),
        width=width,
        height=height,
        packed=False,
        rasterize_mode="analytic",
    )
    direct_colors, direct_alphas = rasterize_to_pixels(
        meta["means2d"],
        meta["conics"],
        colors[None],
        meta["opacities"],
        width,
        height,
        meta["tile_size"],
        meta["isect_offsets"],
        meta["flatten_ids"],
        rasterize_mode="analytic",
    )
    torch.testing.assert_close(render_colors, direct_colors)
    torch.testing.assert_close(render_alphas, direct_alphas)
