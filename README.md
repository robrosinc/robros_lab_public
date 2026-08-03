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

- Isaac Sim/Lab installation: https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/source_installation.html
- Isaac Lab pip workflow: https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html

If Isaac Sim source build fails with a GCC version error, install GCC/G++ 11 and expose them inside the active conda environment. This avoids changing the system default compiler:

```bash
sudo apt-get update
sudo apt-get install -y gcc-11 g++-11

ln -sf /usr/bin/gcc-11 "$CONDA_PREFIX/bin/gcc"
ln -sf /usr/bin/g++-11 "$CONDA_PREFIX/bin/g++"
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export CUDAHOSTCXX=/usr/bin/g++-11
hash -r
gcc --version
g++ --version

cd third_party/IsaacSim
./build.sh
cd ../..
```

After the Isaac Sim / Isaac Lab environment is working, install this repository's submodules and Python packages from the repository root:

```bash
git submodule update --init --recursive
python -m pip install -e third_party/rsl_rl
python -m pip install -e source/robros_lab_public
```

## Train

Train the phase-based rough-terrain velocity policy:

```bash
python scripts/rsl_rl/base/train.py --task Isaac-Velocity-Phase-Rough-IGRISC --headless
```

Use `Flat` instead of `Rough` for the flat-terrain variants:

```bash
python scripts/rsl_rl/base/train.py --task Isaac-Velocity-Phase-Flat-IGRISC --headless
```

## Play

```

Play the latest checkpoint for the phase-based rough-terrain task:

```bash
python scripts/rsl_rl/base/play.py --task Isaac-Velocity-Phase-Rough-IGRISC_Play
```

Flat-terrain play variants:

```bash
python scripts/rsl_rl/base/play.py --task Isaac-Velocity-Phase-Flat-IGRISC_Play
```

## Note

- The IGRIS-C model sets the torso/root as `base_link`. If you implement pelvis-centric observations, rewards, commands, or deployment logic, customize those functions to use the pelvis link instead. In the IGRIS-C model, the pelvis link is named `Link_Waist_Pitch`.
- The reward settings are minimal examples. Users may have to tune the rewards further to see desired policy behaviors.

## Common Issues

If training fails with `ModuleNotFoundError: No module named 'isaacsim'`, expose the built Isaac Sim Python environment inside the active conda environment:

```bash
conda activate igris_c_isaaclab
cd ~/isaac_ws/robros_lab_public
source third_party/IsaacSim/_build/linux-x86_64/release/setup_conda_env.sh
python -c "from isaacsim import SimulationApp; print('isaacsim ok')"
```

## Support

If you have problems using this repository, contact `woohyun321@robros.co.kr`.
