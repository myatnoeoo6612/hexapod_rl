from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from my_hexapod_walk_rl.robots.my_hexapod_walk_rl import MY_HEXAPOD_WALK_RL_CFG


@configclass
class MyHexapodWalkRlEnvCfg(DirectRLEnvCfg):
    decimation = 4
    episode_length_s = 20.0

    # observation/action dimensions
    action_space = 12
    observation_space = 12 * 2 + 6
    state_space = 0

    # simulation config (Isaac Lab 5.0 style)
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        gravity=(0, 0, -9.81),
        render_interval=decimation,
    )

    robot_cfg: ArticulationCfg = MY_HEXAPOD_WALK_RL_CFG

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024,
        env_spacing=3.0,
        replicate_physics=True,
    )

    # reward scaling
    rew_scale_alive = 1.0
    rew_scale_forward = 3.0
    rew_scale_upright = 1.0
    rew_scale_energy = -0.005
    rew_scale_tripod = 0.5       
    rew_scale_yaw = 0.3          