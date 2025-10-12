# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # === General training parameters ===
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 200
    experiment_name = "hexapod_tripod"
    device = "cuda:0"

    # === Observation group mapping ===
    # Critical fix: maps env observation dict -> actor/critic inputs
    obs_groups = {
        "policy": ["policy"],
        "critic": ["policy"],
    }

    # === PPO Policy network ===
    policy = RslRlPpoActorCriticCfg(
    init_noise_std=0.25,
    actor_obs_normalization=True,
    critic_obs_normalization=True,
    actor_hidden_dims=[512, 256],
    critic_hidden_dims=[512, 256],
    activation="elu", 
    )

    # === PPO Algorithm hyperparameters ===
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=2.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=30,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

    # algorithm = RslRlPpoAlgorithmCfg(
    # value_loss_coef=1.0,
    # use_clipped_value_loss=True,
    # clip_param=0.2,
    # entropy_coef=0.02,          # more exploration early
    # num_learning_epochs=20,     # not 40, better balance
    # num_mini_batches=8,         # smoother gradient
    # learning_rate=2.5e-4,       # safe for bigger network
    # schedule="adaptive",
    # gamma=0.99,
    # lam=0.95,
    # desired_kl=0.008,           # tighter control
    # max_grad_norm=1.0,
    # )


    # === Optional: smoother rollouts ===
    # clip_actions = True
    resume = False
    logging_interval = 10
