"""Configuration for custom terrains."""

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg

from .curriculum_random_uniform import HfCurriculumRandomUniformTerrainCfg

RANDOM_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "random_rough": HfCurriculumRandomUniformTerrainCfg(
            proportion=1.0,
            noise_range_start=(0.0, 0.0),
            noise_range=(0.02, 0.10),
            noise_step=0.02,
            border_width=0.25,
        ),
    },
)
"""Rough terrains configuration."""
