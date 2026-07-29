# robros_lab_public

Sim-to-real validated Isaac Lab training framework for customers training IGRIS-C locomotion policies.

This repository contains the public IGRIS-C robot asset, velocity-tracking locomotion tasks, RSL-RL PPO configuration, and runner scripts needed to train and play policies for IGRIS-C in Isaac Lab.

## Setup

Isaac Sim 5.x / Isaac Lab require a Python 3.11 environment. Create and activate a conda environment before installing the training stack:

```bash
conda create -n igris_c_isaaclab python=3.11 -y
conda activate igris_c_isaaclab
python -m pip install --upgrade pip
```

Install Isaac Sim and Isaac Lab by following NVIDIA's public documentation:

- Isaac Sim installation: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/index.html
- Isaac Lab installation: https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html
- Isaac Lab pip workflow: https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html

After the Isaac Sim / Isaac Lab environment is working, install this repository's submodules and Python packages from the repository root:

```bash
git submodule update --init --recursive
python -m pip install -e third_party/rsl_rl
python -m pip install -e source/robros_lab_public
```

## Train

Train the base rough-terrain velocity policy:

```bash
python scripts/rsl_rl/base/train.py --task Isaac-Velocity-Rough-IGRISC --headless
```

Train the phase-based rough-terrain velocity policy:

```bash
python scripts/rsl_rl/base/train.py --task Isaac-Velocity-Phase-Rough-IGRISC --headless
```

Use `Flat` instead of `Rough` for the flat-terrain variants:

```bash
python scripts/rsl_rl/base/train.py --task Isaac-Velocity-Flat-IGRISC --headless
python scripts/rsl_rl/base/train.py --task Isaac-Velocity-Phase-Flat-IGRISC --headless
```

## Play

Play the latest checkpoint for the base rough-terrain task:

```bash
python scripts/rsl_rl/base/play.py --task Isaac-Velocity-Rough-IGRISC_Play
```

Play the latest checkpoint for the phase-based rough-terrain task:

```bash
python scripts/rsl_rl/base/play.py --task Isaac-Velocity-Phase-Rough-IGRISC_Play
```

Flat-terrain play variants:

```bash
python scripts/rsl_rl/base/play.py --task Isaac-Velocity-Flat-IGRISC_Play
python scripts/rsl_rl/base/play.py --task Isaac-Velocity-Phase-Flat-IGRISC_Play
```

## Model Note

The IGRIS-C model sets the torso/root as `base_link`. If you implement pelvis-centric observations, rewards, commands, or deployment logic, customize those functions to use the pelvis link instead. In the IGRIS-C model, the pelvis link is named `Link_Waist_Pitch`.
