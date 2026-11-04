# robros_lab_public

Sim-to-real validated Isaac Lab training framework for customers training IGRIS-C locomotion policies.

This repository contains the public IGRIS-C robot asset, velocity-tracking locomotion tasks, RSL-RL PPO + AMP configuration, and runner scripts needed to train and play policies for IGRIS-C in Isaac Lab with the Newton / MuJoCo Warp (MJWarp) physics backend.

## Demo

<table>
  <tr>
    <th align="center">Isaac Lab</th>
    <th align="center">Sim-to-Sim (MuJoCo)</th>
    <th align="center">Sim-to-Real (IGRIS-C)</th>
  </tr>
  <tr>
    <td align="center" width="260">TBD</td>
    <td align="center" width="260">TBD</td>
    <td align="center" width="260">TBD</td>
  </tr>
</table>

## Setup

Isaac Sim 6.0.1 / Isaac Lab require a Python 3.12 environment. Isaac Sim 6.0.1, a fixed Isaac Lab commit, and RSL-RL 5.4.2 are pinned as submodules under `third_party/`. Use the pinned revisions; do not check out `main` inside a submodule.

Clone the repository and fetch the Git LFS assets and submodules:

```bash
git clone https://github.com/robrosinc/robros_lab_public.git
cd robros_lab_public
git lfs install
git lfs pull
git submodule update --init --recursive
```

Install Isaac Sim and Isaac Lab from the pinned submodules. For details, see NVIDIA's public documentation:

- Isaac Sim/Lab installation: https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/source_installation.html
- Isaac Lab pip workflow: https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html

The Isaac Sim source build requires GCC/G++ 11. If the build fails with a GCC version error, install GCC/G++ 11 and expose them only for the build. This avoids changing the system default compiler:

```bash
sudo apt-get update
sudo apt-get install -y gcc-11 g++-11
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export CUDAHOSTCXX=/usr/bin/g++-11
```

Build Isaac Sim (this requires accepting NVIDIA's license terms), link it into Isaac Lab, create the Python 3.12 conda environment, and install the training stack from the repository root:

```bash
# Build the pinned Isaac Sim checkout.
cd third_party/IsaacSim
git lfs pull
./build.sh -r -j "$(nproc)"
cd ../..

# Point the pinned Isaac Lab checkout at the local Isaac Sim build (x86-64 path).
ln -sfn "$(pwd)/third_party/IsaacSim/_build/linux-x86_64/release" \
  third_party/IsaacLab/_isaac_sim

# Create and activate the Python 3.12 conda environment.
cd third_party/IsaacLab
./isaaclab.sh --conda igris_c_isaaclab
conda activate igris_c_isaaclab

# Install the core Isaac Lab packages, including the Newton/MJWarp dependencies.
./isaaclab.sh --install core
cd ../..

# Install the pinned RSL-RL and this repository's package.
python -m pip install -e third_party/rsl_rl
python -m pip install -e source/robros_lab_public
```

On aarch64, replace `linux-x86_64` in the symlink command with `linux-aarch64`.

If the manual installation does not work on your machine, run the setup script from the repository root instead. It performs the same steps, uses GCC/G++ 11 only for the Isaac Sim build, and registers the Isaac Sim environment in the conda activation hook:

```bash
ENV_NAME=igris_c_isaaclab bash setup.sh
conda activate igris_c_isaaclab
```

Verify the installation:

```bash
python -c "import isaaclab, isaaclab_newton, robros_lab_public, rsl_rl; print('OK')"
```

## Train

Train the phase-based rough-terrain velocity policy:

```bash
python scripts/rsl_rl/locomotion/train.py --task Isaac-Velocity-Phase-Rough-IGRISC --visualizer none physics=newton_mjwarp
```

Use `Flat` instead of `Rough` for the flat-terrain variants:

```bash
python scripts/rsl_rl/locomotion/train.py --task Isaac-Velocity-Phase-Flat-IGRISC --visualizer none physics=newton_mjwarp
```

## Play

Play the latest checkpoint for the phase-based rough-terrain task:

```bash
python scripts/rsl_rl/locomotion/play.py --task Isaac-Velocity-Phase-Rough-IGRISC_Play --num_envs 1 --visualizer newton physics=newton_mjwarp
```

Flat-terrain play variants:

```bash
python scripts/rsl_rl/locomotion/play.py --task Isaac-Velocity-Phase-Flat-IGRISC_Play --num_envs 1 --visualizer newton physics=newton_mjwarp
```

## Note

- The IGRIS-C model sets the torso/root as `base_link`. If you implement pelvis-centric observations, rewards, commands, or deployment logic, customize those functions to use the pelvis link instead. In the IGRIS-C model, the pelvis link is named `Link_Waist_Pitch`.
- The reward settings are minimal examples. Users may have to tune the rewards further to see desired policy behaviors.
- AMP reference motions are in `source/robros_lab_public/robros_lab_public/dataset/igris_c/amp_locomotion` (tracked with Git LFS).

## Common Issues

If training fails with `ModuleNotFoundError: No module named 'isaacsim'`, expose the built Isaac Sim Python environment inside the active conda environment:

```bash
conda activate igris_c_isaaclab
cd <path-to>/robros_lab_public  # repository root
source third_party/IsaacSim/_build/linux-x86_64/release/setup_conda_env.sh
python -c "from isaacsim import SimulationApp; print('isaacsim ok')"
```

On aarch64, use `_build/linux-aarch64/release` instead.

The Isaac Sim 6.0.1 source build does not generate `setup_conda_env.sh` by itself. If the file is missing after a manual installation, run `setup.sh` as described in [Setup](#setup); it creates the file and registers it in the conda activation hook.

## Support

If you have problems using this repository, please open an issue on GitHub.

## License

This project is licensed under the Apache License 2.0 (see `LICENSE`). Third-party code keeps its original license; see the file headers and `licenses/`.

## Acknowledgements

- [fan-ziqi/robot_lab](https://github.com/fan-ziqi/robot_lab)
- [Alescontrela/AMP_for_hardware](https://github.com/Alescontrela/AMP_for_hardware)
- [NVlabs/GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl) (GEAR-SONIC motion utilities in `utils/motions`, Apache-2.0)
- [ZhengyiLuo/PHC](https://github.com/ZhengyiLuo/PHC) (humanoid forward kinematics in `utils/motions/torch_humanoid_batch.py`, BSD-3-Clause-Clear)
