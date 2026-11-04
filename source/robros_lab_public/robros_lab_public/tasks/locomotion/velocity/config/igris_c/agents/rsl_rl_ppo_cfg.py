import os

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlSymmetryCfg,
)

from robros_lab_public.config.rsl_rl.rl_cfg import (
    RslRlAmpCfg,
    RslRlAmpDiscriminatorCfg,
    RslRlAmpMotionDatasetCfg,
    RslRlAmpPpoAlgorithmCfg,
    RslRlCustomMLPModelCfg,
)
from robros_lab_public.dataset import igris_c_motion_dir

from ..symmetry import compute_symmetric_states

AMP_MOTION_DIR = str(igris_c_motion_dir("amp_locomotion"))
AMP_MJCF_FILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../../../../assets/robots/robros/igris_c/mujoco/igris_c_v2.xml")
)


@configclass
class IGRISCPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 12000
    save_interval = 100
    experiment_name = "igris_c_locomotion"
    logger = "tensorboard"
    wandb_project = "igris_c_locomotion"

    """Configuration of the runner for on-policy algorithms."""

    class_name: str = "OnPolicyRunner"
    """The runner class name. Defaults to OnPolicyRunner."""

    obs_groups = {
        "actor": ["policy"],
        "critic": ["critic"],
        "amp": ["amp"],
        "amp_label": ["amp_label"],
    }

    actor: RslRlCustomMLPModelCfg = RslRlCustomMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        normalized_obs_groups=["policy"],
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    """The actor configuration."""

    critic: RslRlMLPModelCfg = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
    )
    """The critic configuration."""

    algorithm: RslRlAmpPpoAlgorithmCfg = RslRlAmpPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        # upstream rsl_rl symmetry augmentation
        symmetry_cfg=RslRlSymmetryCfg(use_data_augmentation=True, data_augmentation_func=compute_symmetric_states),
        amp_cfg=RslRlAmpCfg(
            task_reward_lerp=0.3,
            motion_datasets={
                "motion": RslRlAmpMotionDatasetCfg(
                    motion_dir=AMP_MOTION_DIR,
                    mjcf_file=AMP_MJCF_FILE,
                    preload_transitions=True,
                    num_preload_transitions=200000,
                ),
            },
            discriminator=RslRlAmpDiscriminatorCfg(
                hidden_dims=[1024, 512],
                reward_coef=2.0 / 50,
                activation="relu",
                grad_penalty_coef=10.0,
                trunk_weight_decay=1.0e-3,
                head_weight_decay=1.0e-1,
                learning_rate=2.0e-4,
            ),
            num_transition_steps=5,
        ),
    )
