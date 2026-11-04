from __future__ import annotations

from dataclasses import MISSING

import numpy as np
import scipy.interpolate as interpolate
from isaaclab.terrains.height_field.hf_terrains_cfg import HfRandomUniformTerrainCfg
from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.utils.configclass import configclass


@height_field_to_mesh
def curriculum_random_uniform_terrain(difficulty: float, cfg: "HfCurriculumRandomUniformTerrainCfg") -> np.ndarray:
    """Generate random-uniform rough terrain whose height range scales with difficulty."""
    if cfg.downsampled_scale is None:
        downsampled_scale = cfg.horizontal_scale
    else:
        downsampled_scale = cfg.downsampled_scale
    if downsampled_scale < cfg.horizontal_scale:
        raise ValueError(
            "Downsampled scale must be larger than or equal to the horizontal scale:"
            f" {downsampled_scale} < {cfg.horizontal_scale}."
        )

    difficulty = float(np.clip(difficulty, 0.0, 1.0))
    noise_min = cfg.noise_range_start[0] + difficulty * (cfg.noise_range[0] - cfg.noise_range_start[0])
    noise_max = cfg.noise_range_start[1] + difficulty * (cfg.noise_range[1] - cfg.noise_range_start[1])
    if noise_min > noise_max:
        raise ValueError(
            f"Interpolated noise range is invalid: ({noise_min}, {noise_max}). Check noise_range_start and noise_range."
        )

    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    width_downsampled = int(cfg.size[0] / downsampled_scale)
    length_downsampled = int(cfg.size[1] / downsampled_scale)

    height_min = int(noise_min / cfg.vertical_scale)
    height_max = int(noise_max / cfg.vertical_scale)
    height_step = int(cfg.noise_step / cfg.vertical_scale)
    if height_step <= 0:
        raise ValueError(
            "Noise step must be at least one vertical discretization step:"
            f" noise_step={cfg.noise_step}, vertical_scale={cfg.vertical_scale}."
        )

    height_range = np.arange(height_min, height_max + height_step, height_step)
    height_field_downsampled = np.random.choice(height_range, size=(width_downsampled, length_downsampled))

    x = np.linspace(0, cfg.size[0] * cfg.horizontal_scale, width_downsampled)
    y = np.linspace(0, cfg.size[1] * cfg.horizontal_scale, length_downsampled)
    func = interpolate.RectBivariateSpline(x, y, height_field_downsampled)

    x_upsampled = np.linspace(0, cfg.size[0] * cfg.horizontal_scale, width_pixels)
    y_upsampled = np.linspace(0, cfg.size[1] * cfg.horizontal_scale, length_pixels)
    z_upsampled = func(x_upsampled, y_upsampled)
    return np.rint(z_upsampled).astype(np.int16)


@configclass
class HfCurriculumRandomUniformTerrainCfg(HfRandomUniformTerrainCfg):
    """Random uniform terrain that becomes rougher as terrain difficulty increases."""

    function = curriculum_random_uniform_terrain

    noise_range_start: tuple[float, float] = MISSING
    """The height noise range used at curriculum difficulty 0.0."""
