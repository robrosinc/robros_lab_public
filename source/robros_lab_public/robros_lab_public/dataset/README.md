This folder contains canonical DOF-based PKL motions for supported robots.

For IGRIS-C, motion files are grouped in flat, usage-prefixed directories
under `igris_c/`:

- `amp_<category>` contains expert-reference motions used by AMP. For example,
  `amp_locomotion`.

Each category directory contains PKL files directly; there are no nested
dataset or discriminator directories. The PKLs use the canonical dictionary
schema (`fps`, root pose/velocities, and `dof_pos`/`dof_vel`) and are tracked
with Git LFS.
