# Hexapod Walk RL Env Config

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from my_hexapod_walk_rl.robots.my_hexapod_walk_rl import MY_HEXAPOD_WALK_RL_CFG



@configclass
class MyHexapodWalkRlEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 4
    episode_length_s = 20.0

    # observation/action spaces
    action_space = 12
    observation_space = 12 * 2 + 6  # joints + base vel
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot
    robot_cfg: ArticulationCfg = MY_HEXAPOD_WALK_RL_CFG

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024, env_spacing=3.0, replicate_physics=True
    )

    # reward scales tuned for hybrid tripod+RL
    rew_scale_alive = 1.0
    rew_scale_forward = 3.0
    rew_scale_upright = 1.0
    rew_scale_energy = -0.005
    rew_scale_tripod = 1.0
