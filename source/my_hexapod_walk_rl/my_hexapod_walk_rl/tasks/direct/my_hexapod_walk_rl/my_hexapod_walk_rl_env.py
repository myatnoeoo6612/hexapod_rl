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
    cfg: MyHexapodWalkRlEnvCfg

    def __init__(self, cfg: MyHexapodWalkRlEnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._joint_indices = None
        self.joint_pos = None
        self.joint_vel = None
        self.sim_frame = 0          # step counter for timing

    # -------------------------------------------------------------
    # Scene setup
    # -------------------------------------------------------------
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        sim_utils.spawn_light("/World/Light",
                              sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)))

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        self.scene.articulations["robot"] = self.robot

 
    # -------------------------------------------------------------
    # Tripod reference gait
    # -------------------------------------------------------------
    def _tripod_reference(self, t: float) -> torch.Tensor:
        """Generate ideal tripod gait reference for all 12 joints with joint limits."""
        # === Base gait parameters ===
        freq = 3.0          # Hz
        amp_hip = 0.20      # rad amplitude
        amp_knee = 0.08     # rad amplitude
        phase_a, phase_b = 0.0, math.pi

        # === Real joint limits (from Dynamixel hardware) ===
        hip_min, hip_max = -0.5236, 0.5236     # ±30°
        knee_min, knee_max = 0.50, 0.70        # 28.7°–40.1°

        # === Time tensor ===
        t_tensor = torch.full((self.num_envs,), t, device=self.device)
        omega_t = 2 * math.pi * freq * t_tensor

        ref = torch.zeros((self.num_envs, 12), device=self.device)

        # === Tripod grouping ===
        tripod_a = [0, 2, 4]   # legs 1,3,5
        tripod_b = [1, 3, 5]   # legs 2,4,6

        # === Hip direction (mirror left/right) ===
        hip_signs = [-1, -1, -1, -1, +1, +1]   # right(-), left(+)
        hip_amp_scale = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # reduce hip_4 a bit

        for leg in range(6):
            hip_i, knee_i = 2 * leg, 2 * leg + 1
            phase = phase_a if leg in tripod_a else phase_b
            sign = hip_signs[leg]
            scale = hip_amp_scale[leg]

            # Hip motion (sinusoidal swing)
            hip_angle = sign * amp_hip * scale * torch.sin(omega_t + phase)
            hip_angle = torch.clamp(hip_angle, hip_min, hip_max)

            # Knee motion (sinusoidal lift)
            knee_angle = 0.6 + amp_knee * torch.sin(omega_t + phase + math.pi / 2)
            knee_angle = torch.clamp(knee_angle, knee_min, knee_max)

            ref[:, hip_i] = hip_angle
            ref[:, knee_i] = knee_angle

        return ref



    # -------------------------------------------------------------
    # RL hooks
    # -------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # Lazy initialize after physics is ready
        if self._joint_indices is None:
            num_joints = self.robot.data.joint_pos.shape[1]
            self._joint_indices = list(range(num_joints))
            self.joint_pos = self.robot.data.joint_pos
            self.joint_vel = self.robot.data.joint_vel
            print(f"[Init] Joint data ready: {num_joints} DOF")


        self.actions = actions.clone()


    def _apply_action(self) -> None:
        t = self.sim_frame * self.cfg.sim.dt
        ref = self._tripod_reference(t)
        target = ref + 0.3 * self.actions  # 30% RL offset
        self.robot.set_joint_position_target(target, joint_ids=self._joint_indices)

        if self.sim_frame % 200 == 0:
            jp = self.joint_pos[0, self._joint_indices].detach().cpu().numpy()
            print(f"[{self.sim_frame}] mean|pos|={jp.mean():+.3f}")
        self.sim_frame += 1

    # -------------------------------------------------------------
    def _get_observations(self) -> dict:
        """Collect observations (safe during first reset)."""
        # --- ensure data buffers exist ---
        if self._joint_indices is None:
            # return a safe dummy observation until physics starts
            dummy = torch.zeros((self.num_envs, 30), device=self.device)
            return {"policy": dummy}

        # make sure joint data exist
        if self.joint_pos is None or self.joint_vel is None:
            dummy = torch.zeros((self.num_envs, 30), device=self.device)
            return {"policy": dummy}

        # --- real observations ---
        jp = self.joint_pos[:, self._joint_indices]
        jv = self.joint_vel[:, self._joint_indices]
        lin = self.robot.data.root_lin_vel_w
        ang = self.robot.data.root_ang_vel_w

        # flatten shapes like [N,1,3] -> [N,3]
        def ensure_flat(x):
            if x.ndim == 3:
                return x.squeeze(1)
            if x.ndim == 1:
                return x.unsqueeze(0)
            return x

        jp, jv, lin, ang = map(ensure_flat, (jp, jv, lin, ang))
        obs = torch.cat((jp, jv, lin, ang), dim=-1)

        # ✅ always return dict with key 'policy'
        return {"policy": obs}



    # -------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        return compute_rewards(
            self.robot.data.root_lin_vel_w,
            self.robot.data.root_quat_w,
            self.joint_vel,
            self.actions,
            self.reset_terminated,
            self.cfg,
        )

    # -------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        base_z = self.robot.data.root_pos_w[:, 2]
        base_q = self.robot.data.root_quat_w
        N = base_q.shape[0]
        up = quat_apply(base_q, torch.tensor([0.0, 0.0, 1.0],
                                             device=base_q.device, dtype=base_q.dtype).repeat(N, 1))
        timeout = self.episode_length_buf >= self.max_episode_length - 1
        fallen = (base_z < 0.05) | (up[:, 2] < 0.5)
        return fallen, timeout

    # -------------------------------------------------------------
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        self.sim_frame = 0
        jp = self.robot.data.default_joint_pos[env_ids]
        jv = self.robot.data.default_joint_vel[env_ids]
        root = self.robot.data.default_root_state[env_ids].clone()
        root[:, :3] += self.scene.env_origins[env_ids]
        root[:, 2] += 0.25
        self.robot.write_root_pose_to_sim(root[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(jp, jv, None, env_ids)


# ================================================================
# Reward function
# ================================================================
def compute_rewards(base_lin_vel, base_quat, joint_vel, actions, reset_terminated, cfg):
    rew_alive = cfg.rew_scale_alive * (1.0 - reset_terminated.float())
    rew_forward = cfg.rew_scale_forward * base_lin_vel[:, 0]
    N = base_quat.shape[0]
    up = quat_apply(base_quat, torch.tensor([0.0, 0.0, 1.0],
                                            device=base_quat.device, dtype=base_quat.dtype).repeat(N, 1))
    rew_upright = cfg.rew_scale_upright * up[:, 2]
    rew_energy = cfg.rew_scale_energy * torch.sum(actions ** 2, dim=1)
    return rew_alive + rew_forward + rew_upright + rew_energy
