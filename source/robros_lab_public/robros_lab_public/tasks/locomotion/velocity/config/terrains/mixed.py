"""Configuration for custom terrains."""

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg

from .curriculum_random_uniform import HfCurriculumRandomUniformTerrainCfg

MIXED_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        # "random_rough": HfCurriculumRandomUniformTerrainCfg(
        #     proportion=0.1, noise_range_start=(0., 0.), noise_range=(-0.1, 0.1), noise_step=0.01, border_width=0.25
        # ),
        # "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
        #     proportion=0.2,
        #     step_height_range=(0.0, 0.23),
        #     step_width=0.35,
        #     platform_width=3.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
        # "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
        #     proportion=0.2,
        #     step_height_range=(0.0, 0.23),
        #     step_width=0.35,
        #     platform_width=3.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
        "random_rough": HfCurriculumRandomUniformTerrainCfg(
            proportion=0.2,
            noise_range_start=(0.0, 0.0),
            noise_range=(-0.05, 0.05),
            noise_step=0.01,
            border_width=0.25,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2, grid_width=1.5, grid_height_range=(0.0, 0.115), platform_width=2.0
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.3, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.3, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
    },
)
"""Rough terrains configuration."""
