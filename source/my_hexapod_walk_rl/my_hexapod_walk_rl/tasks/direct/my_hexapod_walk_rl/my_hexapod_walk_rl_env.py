import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply
from .my_hexapod_walk_rl_env_cfg import MyHexapodWalkRlEnvCfg


class MyHexapodWalkRlEnv(DirectRLEnv):
    """Hybrid tripod + RL environment for the hexapod."""

    cfg: MyHexapodWalkRlEnvCfg

    def __init__(self, cfg: MyHexapodWalkRlEnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._joint_indices = None
        self.joint_pos = None
        self.joint_vel = None
        # forward direction mapping (matches your ROS 2 tripod gait)
        self.forward_sign = {1: -1, 2: -1, 3: -1, 4: +1, 5: -1, 6: +1}
        self.sim_frame = 0 
    # ------------------------------------------------------------------ #
    # Scene setup
    # ------------------------------------------------------------------ #
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------ #
    # Core RL loop
    # ------------------------------------------------------------------ #
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if self._joint_indices is None:
            n = self.robot.num_joints
            self._joint_indices = list(range(n))
            self.joint_pos = self.robot.data.joint_pos
            self.joint_vel = self.robot.data.joint_vel
        self.actions = actions.clone()

    def _tripod_reference(self, t: float) -> torch.Tensor:
        """Generate reference tripod gait (hips + knees)."""
        hip_swing, knee_lift = 0.45, 0.25
        hip_min, hip_max = -0.523, 0.523
        knee_down, knee_up = 0.5, 0.7
        hip_init = 0.0
        freq = 1.5  # Hz

        # Which group swings this half-cycle?
        phase_val = math.sin(2 * math.pi * freq * t)
        swingA, swingB = [1, 3, 5], [2, 4, 6]
        swing_legs = swingA if phase_val > 0 else swingB
        stance_legs = swingB if phase_val > 0 else swingA

        pos = torch.zeros((self.num_envs, 12), device=self.device)

        for leg in range(1, 7):
            hip_i = (leg - 1) * 2
            knee_i = hip_i + 1
            sign = self.forward_sign[leg]
            if leg in swing_legs:
                hip = hip_init + sign * hip_swing * math.sin(2 * math.pi * freq * t)
                knee = knee_down + (knee_up - knee_down) * abs(math.sin(math.pi * freq * t))
            else:
                hip = hip_init - sign * hip_swing * math.sin(2 * math.pi * freq * t)
                knee = knee_down
            pos[:, hip_i] = torch.clamp(torch.tensor(hip, device=self.device), hip_min, hip_max)
            pos[:, knee_i] = torch.clamp(torch.tensor(knee, device=self.device), knee_down, knee_up)
        return pos

    def _apply_action(self) -> None:
        """Combine reference tripod motion + RL offset using simulation time."""
        t = self.sim_frame * self.cfg.sim.dt
        ref = self._tripod_reference(t)
        if self.sim_frame % 500 == 0:           # print every few frames
            print(f"time {t:.2f}s | ref[0,:6]={ref[0,:6].cpu().numpy()}")

        offset_scale = 0.5
        target = ref + offset_scale * self.actions
        self.robot.set_joint_position_target(target, joint_ids=self._joint_indices)
        self.ref_pos = ref  # store for reward calculation
        self.sim_frame += 1

    # ------------------------------------------------------------------ #
    # Observations / Rewards / Terminations
    # ------------------------------------------------------------------ #
    def _get_observations(self) -> dict:
        jp = self.joint_pos[:, self._joint_indices]
        jv = self.joint_vel[:, self._joint_indices]
        lin = self.robot.data.root_lin_vel_w
        ang = self.robot.data.root_ang_vel_w

        # --- Flatten possible [N,1,3] tensors to [N,3]
        def ensure_flat(x):
            if x.ndim == 3:
                return x.squeeze(1)
            if x.ndim == 1:
                return x.unsqueeze(0)
            return x

        jp = ensure_flat(jp)
        jv = ensure_flat(jv)
        lin = ensure_flat(lin)
        ang = ensure_flat(ang)

        obs = torch.cat((jp, jv, lin, ang), dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        return compute_rewards(
            self.robot.data.root_lin_vel_w,
            self.robot.data.root_quat_w,
            self.joint_pos,
            self.joint_vel,
            self.actions,
            self.ref_pos,
            self.reset_terminated,
            self.cfg,
        )

    def _get_dones(self):
        base_z = self.robot.data.root_pos_w[:, 2]
        base_q = self.robot.data.root_quat_w
        N = base_q.shape[0]
        up = quat_apply(base_q, torch.tensor([0.0, 0.0, 1.0], device=base_q.device).repeat(N, 1))
        timeout = self.episode_length_buf >= self.max_episode_length - 1
        fallen = (base_z < 0.08) | (up[:, 2] < 0.5)
        return fallen, timeout

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        if self.joint_pos is None or self.joint_vel is None:
            self.joint_pos = self.robot.data.joint_pos
            self.joint_vel = self.robot.data.joint_vel

        jp = self.robot.data.default_joint_pos[env_ids]
        jv = self.robot.data.default_joint_vel[env_ids]
        root = self.robot.data.default_root_state[env_ids].clone()
        root[:, :3] += self.scene.env_origins[env_ids]
        root[:, 2] += 0.05  # start just above ground
        self.joint_pos[env_ids] = jp
        self.joint_vel[env_ids] = jv
        self.robot.write_root_pose_to_sim(root[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(jp, jv, None, env_ids)


# ---------------------------------------------------------------------- #
# Reward computation
# ---------------------------------------------------------------------- #
def compute_rewards(base_lin_vel, base_quat, joint_pos, joint_vel, actions, ref_pos,
                    reset_terminated, cfg: MyHexapodWalkRlEnvCfg):
    N = base_quat.shape[0]
    up = quat_apply(base_quat, torch.tensor([0.0, 0.0, 1.0], device=base_quat.device).repeat(N, 1))

    rew_alive = cfg.rew_scale_alive * (1.0 - reset_terminated.float())
    rew_forward = cfg.rew_scale_forward * base_lin_vel[:, 0]
    rew_upright = cfg.rew_scale_upright * up[:, 2]
    rew_energy = cfg.rew_scale_energy * torch.sum(actions**2, dim=1)

    # imitation term
    rew_ref = -0.5 * torch.mean(torch.square(joint_pos - ref_pos), dim=1)

    # tripod alternation encouragement
    hip_vel_a = torch.mean(joint_vel[:, [3, 1, 5]], dim=1)
    hip_vel_b = torch.mean(joint_vel[:, [0, 4, 2]], dim=1)
    rew_tripod = cfg.rew_scale_tripod * (-torch.abs(hip_vel_a + hip_vel_b))

    # smoothness + lateral stability
    rew_smooth = -0.001 * torch.sum(torch.square(joint_vel), dim=1)
    rew_lat = -0.2 * torch.abs(base_lin_vel[:, 1])

    reward = (
        rew_alive
        + rew_forward
        + 0.5 * rew_upright
        + rew_tripod
        + rew_ref
        + rew_smooth
        + rew_lat
        + rew_energy
    )
    return reward

