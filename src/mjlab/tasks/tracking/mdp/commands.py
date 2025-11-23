from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import mujoco
import numpy as np
import torch

from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_error_magnitude,
  quat_from_euler_xyz,
  quat_inv,
  quat_mul,
  sample_uniform,
  yaw_quat,
)
from mjlab.viewer.debug_visualizer import DebugVisualizer

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_DESIRED_FRAME_COLORS = ((1.0, 0.5, 0.5), (0.5, 1.0, 0.5), (0.5, 0.5, 1.0))


class MotionLoader:
  def __init__(
    self, motion_file: str, body_indexes: torch.Tensor, device: str = "cpu"
  ) -> None:
    data = np.load(motion_file)
    self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
    self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
    self._body_pos_w = torch.tensor(
      data["body_pos_w"], dtype=torch.float32, device=device
    )
    self._body_quat_w = torch.tensor(
      data["body_quat_w"], dtype=torch.float32, device=device
    )
    self._body_lin_vel_w = torch.tensor(
      data["body_lin_vel_w"], dtype=torch.float32, device=device
    )
    self._body_ang_vel_w = torch.tensor(
      data["body_ang_vel_w"], dtype=torch.float32, device=device
    )
    self._body_indexes = body_indexes
    self.time_step_total = self.joint_pos.shape[0]

    # Optional object arrays
    if "object_pos_w" in data and "object_quat_w" in data:
      self._object_pos_w = torch.tensor(
        data["object_pos_w"], dtype=torch.float32, device=device
      )
      self._object_quat_w = torch.tensor(
        data["object_quat_w"], dtype=torch.float32, device=device
      )
      self._object_lin_vel_w = torch.tensor(
        data["object_lin_vel_w"], dtype=torch.float32, device=device
      )
      self._object_ang_vel_w = torch.tensor(
        data["object_ang_vel_w"], dtype=torch.float32, device=device
      )

    if "contact_indicators" in data:
      self._object_contact = torch.tensor(
        data["contact_indicators"], dtype=torch.bool, device=device
      )
    else:
      self._object_contact = None

    # Load contact_positions if available
    if "contact_positions" in data:
      self._contact_positions = torch.tensor(
        data["contact_positions"], dtype=torch.float32, device=device
      )
    else:
      self._contact_positions = None

  @property
  def body_pos_w(self) -> torch.Tensor:
    return self._body_pos_w[:, self._body_indexes]

  @property
  def body_quat_w(self) -> torch.Tensor:
    return self._body_quat_w[:, self._body_indexes]

  @property
  def body_lin_vel_w(self) -> torch.Tensor:
    return self._body_lin_vel_w[:, self._body_indexes]

  @property
  def body_ang_vel_w(self) -> torch.Tensor:
    return self._body_ang_vel_w[:, self._body_indexes]

    # Object pose (global/world) at each time step

  @property
  def object_pos_w(self) -> torch.Tensor:
    if not hasattr(self, "_object_pos_w"):
      return torch.zeros(self.time_step_total, 3, device=self._body_pos_w.device)
    return self._object_pos_w

  @property
  def object_quat_w(self) -> torch.Tensor:
    if not hasattr(self, "_object_quat_w"):
      q = torch.zeros(self.time_step_total, 4, device=self._body_pos_w.device)
      q[:, 0] = 1.0
      return q
    return self._object_quat_w

  @property
  def object_lin_vel_w(self) -> torch.Tensor | None:
    if not hasattr(self, "_object_lin_vel_w"):
      return None
    return self._object_lin_vel_w

  @property
  def object_ang_vel_w(self) -> torch.Tensor | None:
    if not hasattr(self, "_object_ang_vel_w"):
      return None
    return self._object_ang_vel_w

  @property
  def object_contact(self) -> torch.Tensor | None:
    """Contact indicator from motion data. Shape: (time_step_total, num_contacts)"""
    if not hasattr(self, "_object_contact") or self._object_contact is None:
      return None
    return self._object_contact

  @property
  def contact_positions(self) -> torch.Tensor | None:
    """Contact positions from motion data. Shape: (time_step_total, num_contacts, 3)"""
    if not hasattr(self, "_contact_positions") or self._contact_positions is None:
      return None
    return self._contact_positions


class MotionCommand(CommandTerm):
  cfg: MotionCommandCfg
  _env: ManagerBasedRlEnv

  def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.asset_name]
    self.robot_anchor_body_index = self.robot.body_names.index(
      self.cfg.anchor_body_name
    )
    self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
    self.body_indexes = torch.tensor(
      self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0],
      dtype=torch.long,
      device=self.device,
    )
    self.eef_body_indexes = torch.tensor(
      self.robot.find_bodies(self.cfg.eef_body_names, preserve_order=True)[0],
      dtype=torch.long,
      device=self.device,
    )
    self.motion = MotionLoader(
      self.cfg.motion_file, self.body_indexes, device=self.device
    )
    self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.body_pos_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 3, device=self.device
    )
    self.body_quat_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 4, device=self.device
    )
    self.body_quat_relative_w[:, :, 0] = 1.0

    self.bin_count = int(self.motion.time_step_total // (1 / env.step_dt)) + 1
    self.bin_failed_count = torch.zeros(
      self.bin_count, dtype=torch.float, device=self.device
    )
    self._current_bin_failed = torch.zeros(
      self.bin_count, dtype=torch.float, device=self.device
    )
    self.kernel = torch.tensor(
      [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)],
      device=self.device,
    )
    self.kernel = self.kernel / self.kernel.sum()

    self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_anchor_lin_vel"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["error_anchor_ang_vel"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)

    # Ghost model created lazily on first visualization
    self._ghost_model: mujoco.MjModel | None = None
    self._ghost_color = np.array(cfg.viz.ghost_color, dtype=np.float32)

    # Object tracking metrics - only if object exists in motion data
    self._has_object = hasattr(self.motion, "_object_pos_w")
    if self._has_object:
      self.metrics["error_object_pos"] = torch.zeros(self.num_envs, device=self.device)
      self.metrics["error_object_rot"] = torch.zeros(self.num_envs, device=self.device)

    # Contact indicator from motion data
    self._object_contact = (
      self.motion.object_contact
    )  # (time_step_total, num_contacts) or None
    self._contact_positions = (
      self.motion.contact_positions
    )  # (time_step_total, num_contacts, 3) or None
    if self._object_contact is not None:
      # Initialize contact reference arrays
      num_contacts = (
        self._object_contact.shape[1] if self._object_contact.ndim > 1 else 1
      )
      self.ref_object_contact_future = torch.zeros(
        self.num_envs, 1, num_contacts, dtype=torch.bool, device=self.device
      )
      self.ref_object_contact = torch.zeros(
        self.num_envs, num_contacts, dtype=torch.bool, device=self.device
      )
    else:
      self.ref_object_contact_future = None
      self.ref_object_contact = None

    # # Contact target positions (reconstructed from offsets in YAML/config)
    # # These will be set in _update_command based on object pose and offsets
    # contact_target_pos_offset = getattr(cfg, "contact_target_pos_offset", None)
    # contact_eef_pos_offset = getattr(cfg, "contact_eef_pos_offset", None)

    # if contact_target_pos_offset is not None:
    #   # Convert list of lists to tensor
    #   if isinstance(contact_target_pos_offset, list):
    #     self.contact_target_pos_offset = torch.tensor(
    #       contact_target_pos_offset, dtype=torch.float32, device=self.device
    #     )  # (num_contacts, 3)
    #   else:
    #     self.contact_target_pos_offset = contact_target_pos_offset
    #   num_contacts = self.contact_target_pos_offset.shape[0]
    #   self.contact_target_pos_w = torch.zeros(
    #     self.num_envs, num_contacts, 3, dtype=torch.float32, device=self.device
    #   )
    # else:
    #   self.contact_target_pos_offset = None
    #   self.contact_target_pos_w = None

    # if contact_eef_pos_offset is not None:
    #   if isinstance(contact_eef_pos_offset, list):
    #     self.contact_eef_pos_offset = torch.tensor(
    #       contact_eef_pos_offset, dtype=torch.float32, device=self.device
    #     )  # (num_contacts, 3)
    #   else:
    #     self.contact_eef_pos_offset = contact_eef_pos_offset
    # else:
    #   self.contact_eef_pos_offset = None

  @property
  def command(self) -> torch.Tensor:
    return torch.cat([self.joint_pos, self.joint_vel], dim=1)

  @property
  def joint_pos(self) -> torch.Tensor:
    return self.motion.joint_pos[self.time_steps]

  @property
  def joint_vel(self) -> torch.Tensor:
    return self.motion.joint_vel[self.time_steps]

  @property
  def body_pos_w(self) -> torch.Tensor:
    return (
      self.motion.body_pos_w[self.time_steps] + self._env.scene.env_origins[:, None, :]
    )

  @property
  def body_quat_w(self) -> torch.Tensor:
    return self.motion.body_quat_w[self.time_steps]

  @property
  def body_lin_vel_w(self) -> torch.Tensor:
    return self.motion.body_lin_vel_w[self.time_steps]

  @property
  def body_ang_vel_w(self) -> torch.Tensor:
    return self.motion.body_ang_vel_w[self.time_steps]

  @property
  def anchor_pos_w(self) -> torch.Tensor:
    return (
      self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index]
      + self._env.scene.env_origins
    )

  @property
  def anchor_quat_w(self) -> torch.Tensor:
    return self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]

  @property
  def anchor_lin_vel_w(self) -> torch.Tensor:
    return self.motion.body_lin_vel_w[self.time_steps, self.motion_anchor_body_index]

  @property
  def anchor_ang_vel_w(self) -> torch.Tensor:
    return self.motion.body_ang_vel_w[self.time_steps, self.motion_anchor_body_index]

  @property
  def robot_joint_pos(self) -> torch.Tensor:
    return self.robot.data.joint_pos

  @property
  def robot_joint_vel(self) -> torch.Tensor:
    return self.robot.data.joint_vel

  @property
  def robot_body_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.body_indexes]

  @property
  def robot_body_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.body_indexes]

  @property
  def robot_body_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.body_indexes]

  @property
  def robot_body_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.body_indexes]

  @property
  def robot_anchor_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.robot_anchor_body_index]

  @property
  def robot_eef_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.eef_body_indexes]

  @property
  def robot_eef_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.eef_body_indexes]

  @property
  def object_pos_w(self) -> torch.Tensor:
    pos = self.motion.object_pos_w[self.time_steps]
    # Offset by env origins in x,y to align with robot placement
    return pos + self._env.scene.env_origins

  @property
  def object_quat_w(self) -> torch.Tensor:
    return self.motion.object_quat_w[self.time_steps]

  @property
  def object_lin_vel_w(self) -> torch.Tensor:
    return (
      self.motion.object_lin_vel_w[self.time_steps]
      if self.motion.object_lin_vel_w is not None
      else None
    )

  @property
  def object_ang_vel_w(self) -> torch.Tensor:
    return (
      self.motion.object_ang_vel_w[self.time_steps]
      if self.motion.object_ang_vel_w is not None
      else None
    )

  @property
  def contact_positions(self) -> torch.Tensor | None:
    """Contact positions for current timestep (world frame). Shape: (num_envs, num_contacts, 3)"""
    if self._contact_positions is None:
      return None
    pos = self._contact_positions[self.time_steps]  # (num_envs, num_contacts, 3)
    # Offset by env origins in x,y to align with robot placement
    # env_origins: (num_envs, 3), need to expand to (num_envs, 1, 3) for broadcasting
    return pos + self._env.scene.env_origins[:, None, :]

  def _update_metrics(self):
    self.metrics["error_anchor_pos"] = torch.norm(
      self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1
    )
    self.metrics["error_anchor_rot"] = quat_error_magnitude(
      self.anchor_quat_w, self.robot_anchor_quat_w
    )
    self.metrics["error_anchor_lin_vel"] = torch.norm(
      self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1
    )
    self.metrics["error_anchor_ang_vel"] = torch.norm(
      self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1
    )

    self.metrics["error_body_pos"] = torch.norm(
      self.body_pos_relative_w - self.robot_body_pos_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_body_rot"] = quat_error_magnitude(
      self.body_quat_relative_w, self.robot_body_quat_w
    ).mean(dim=-1)

    self.metrics["error_body_lin_vel"] = torch.norm(
      self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_body_ang_vel"] = torch.norm(
      self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1
    ).mean(dim=-1)

    self.metrics["error_joint_pos"] = torch.norm(
      self.joint_pos - self.robot_joint_pos, dim=-1
    )
    self.metrics["error_joint_vel"] = torch.norm(
      self.joint_vel - self.robot_joint_vel, dim=-1
    )

    if self._has_object:
      # Get actual object pose from simulation
      try:
        box = self._env.scene.entities.get("box")  # type: ignore[attr-defined]
        if box is not None:
          # Desired object pose from motion data
          desired_pos = self.object_pos_w  # (N, 3)
          desired_quat = self.object_quat_w  # (N, 4)

          # Actual object pose from simulation
          actual_pos = box.data.body_link_pos_w[:, 0]  # (N, 3) root body
          actual_quat = box.data.body_link_quat_w[:, 0]  # (N, 4) root body

          # Compute tracking errors
          self.metrics["error_object_pos"] = torch.norm(
            desired_pos - actual_pos, dim=-1
          )
          self.metrics["error_object_rot"] = quat_error_magnitude(
            desired_quat, actual_quat
          )
        else:
          # Box entity doesn't exist, set metrics to zero
          self.metrics["error_object_pos"] = torch.zeros(
            self.num_envs, device=self.device
          )
          self.metrics["error_object_rot"] = torch.zeros(
            self.num_envs, device=self.device
          )
      except Exception:
        # If anything goes wrong, set metrics to zero
        self.metrics["error_object_pos"] = torch.zeros(
          self.num_envs, device=self.device
        )
        self.metrics["error_object_rot"] = torch.zeros(
          self.num_envs, device=self.device
        )

  def _adaptive_sampling(self, env_ids: torch.Tensor):
    episode_failed = self._env.termination_manager.terminated[env_ids]
    if torch.any(episode_failed):
      current_bin_index = torch.clamp(
        (self.time_steps * self.bin_count) // max(self.motion.time_step_total, 1),
        0,
        self.bin_count - 1,
      )
      fail_bins = current_bin_index[env_ids][episode_failed]
      self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count)

    # Sample.
    sampling_probabilities = (
      self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
    )
    sampling_probabilities = torch.nn.functional.pad(
      sampling_probabilities.unsqueeze(0).unsqueeze(0),
      (0, self.cfg.adaptive_kernel_size - 1),  # Non-causal kernel
      mode="replicate",
    )
    sampling_probabilities = torch.nn.functional.conv1d(
      sampling_probabilities, self.kernel.view(1, 1, -1)
    ).view(-1)

    sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

    sampled_bins = torch.multinomial(
      sampling_probabilities, len(env_ids), replacement=True
    )
    self.time_steps[env_ids] = (
      (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
      / self.bin_count
      * (self.motion.time_step_total - 1)
    ).long()

    # Update metrics.
    H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
    H_norm = H / math.log(self.bin_count)
    pmax, imax = sampling_probabilities.max(dim=0)
    self.metrics["sampling_entropy"][:] = H_norm
    self.metrics["sampling_top1_prob"][:] = pmax
    self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

  def _uniform_sampling(self, env_ids: torch.Tensor):
    self.time_steps[env_ids] = torch.randint(
      0, self.motion.time_step_total, (len(env_ids),), device=self.device
    )
    self.metrics["sampling_entropy"][:] = 1.0  # Maximum entropy for uniform.
    self.metrics["sampling_top1_prob"][:] = 1.0 / self.bin_count
    self.metrics["sampling_top1_bin"][:] = 0.5  # No specific bin preference.

  def _resample_command(self, env_ids: torch.Tensor):
    if self.cfg.sampling_mode == "start":
      self.time_steps[env_ids] = 0
    elif self.cfg.sampling_mode == "uniform":
      self._uniform_sampling(env_ids)
    else:
      assert self.cfg.sampling_mode == "adaptive"
      self._adaptive_sampling(env_ids)

    root_pos = self.body_pos_w[:, 0].clone()
    root_ori = self.body_quat_w[:, 0].clone()
    root_lin_vel = self.body_lin_vel_w[:, 0].clone()
    root_ang_vel = self.body_ang_vel_w[:, 0].clone()

    range_list = [
      self.cfg.pose_range.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=self.device)
    rand_samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    )
    root_pos[env_ids] += rand_samples[:, 0:3]
    orientations_delta = quat_from_euler_xyz(
      rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
    )
    root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
    range_list = [
      self.cfg.velocity_range.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=self.device)
    rand_samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    )
    root_lin_vel[env_ids] += rand_samples[:, :3]
    root_ang_vel[env_ids] += rand_samples[:, 3:]

    joint_pos = self.joint_pos.clone()
    joint_vel = self.joint_vel.clone()

    joint_pos += sample_uniform(
      lower=self.cfg.joint_position_range[0],
      upper=self.cfg.joint_position_range[1],
      size=joint_pos.shape,
      device=joint_pos.device,  # type: ignore
    )
    soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
    joint_pos[env_ids] = torch.clip(
      joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
    )
    self.robot.write_joint_state_to_sim(
      joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids
    )

    root_state = torch.cat(
      [
        root_pos[env_ids],
        root_ori[env_ids],
        root_lin_vel[env_ids],
        root_ang_vel[env_ids],
      ],
      dim=-1,
    )

    try:
      box = self._env.scene.entities.get("box")  # type: ignore[attr-defined]
    except Exception:
      box = None

    if box is not None and not box.data.is_fixed_base:
      # Pose from motion object state (already includes per-env origin offset via object_pos_w)
      box_pos = self.object_pos_w[env_ids]
      box_quat = self.object_quat_w[env_ids]
      box_lin_vel = self.object_lin_vel_w[env_ids]
      box_ang_vel = self.object_ang_vel_w[env_ids]

      box_state = torch.cat(
        [
          box_pos,
          box_quat,
          box_lin_vel,
          box_ang_vel,
        ],
        dim=-1,
      )

      box.write_root_state_to_sim(box_state, env_ids=env_ids)

    self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)

    self.robot.clear_state(env_ids=env_ids)
    if box is not None:
      box.clear_state(env_ids=env_ids)

  def _update_command(self):
    self.time_steps += 1
    env_ids = torch.where(self.time_steps >= self.motion.time_step_total)[0]
    if env_ids.numel() > 0:
      self._resample_command(env_ids)

    anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )

    delta_pos_w = robot_anchor_pos_w_repeat
    delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
    delta_ori_w = yaw_quat(
      quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat))
    )

    self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
    self.body_pos_relative_w = delta_pos_w + quat_apply(
      delta_ori_w, self.body_pos_w - anchor_pos_w_repeat
    )

    if self.cfg.sampling_mode == "adaptive":
      self.bin_failed_count = (
        self.cfg.adaptive_alpha * self._current_bin_failed
        + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
      )
      self._current_bin_failed.zero_()

    # Update contact indicator from motion data
    if self._object_contact is not None:
      # Get contact indicator for current time steps
      # time_steps: (num_envs,), _object_contact: (time_step_total, num_contacts)
      contact_at_timesteps = self._object_contact[
        self.time_steps
      ]  # (num_envs, num_contacts)
      if contact_at_timesteps.ndim == 1:
        contact_at_timesteps = contact_at_timesteps.unsqueeze(1)  # (num_envs, 1)
      # Update future contact (for horizon=1, just current step)
      self.ref_object_contact_future = contact_at_timesteps.unsqueeze(
        1
      )  # (num_envs, 1, num_contacts)
      self.ref_object_contact = contact_at_timesteps  # (num_envs, num_contacts)

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    """Draw ghost robot or frames based on visualization mode."""
    if self.cfg.viz.mode == "ghost":
      if self._ghost_model is None:
        self._ghost_model = copy.deepcopy(self._env.sim.mj_model)
        self._ghost_model.geom_rgba[:] = self._ghost_color

      entity: Entity = self._env.scene[self.cfg.asset_name]
      indexing = entity.indexing
      free_joint_q_adr = indexing.free_joint_q_adr.cpu().numpy()
      joint_q_adr = indexing.joint_q_adr.cpu().numpy()

      qpos = np.zeros(self._env.sim.mj_model.nq)
      qpos[free_joint_q_adr[0:3]] = self.body_pos_w[visualizer.env_idx, 0].cpu().numpy()
      qpos[free_joint_q_adr[3:7]] = (
        self.body_quat_w[visualizer.env_idx, 0].cpu().numpy()
      )
      qpos[joint_q_adr] = self.joint_pos[visualizer.env_idx].cpu().numpy()

      visualizer.add_ghost_mesh(qpos, model=self._ghost_model)

    elif self.cfg.viz.mode == "frames":
      desired_body_pos = self.body_pos_w[visualizer.env_idx].cpu().numpy()
      desired_body_quat = self.body_quat_w[visualizer.env_idx]
      desired_body_rotm = matrix_from_quat(desired_body_quat).cpu().numpy()

      current_body_pos = self.robot_body_pos_w[visualizer.env_idx].cpu().numpy()
      current_body_quat = self.robot_body_quat_w[visualizer.env_idx]
      current_body_rotm = matrix_from_quat(current_body_quat).cpu().numpy()

      for i, body_name in enumerate(self.cfg.body_names):
        visualizer.add_frame(
          position=desired_body_pos[i],
          rotation_matrix=desired_body_rotm[i],
          scale=0.08,
          label=f"desired_{body_name}",
          axis_colors=_DESIRED_FRAME_COLORS,
        )
        visualizer.add_frame(
          position=current_body_pos[i],
          rotation_matrix=current_body_rotm[i],
          scale=0.12,
          label=f"current_{body_name}",
        )

      desired_anchor_pos = self.anchor_pos_w[visualizer.env_idx].cpu().numpy()
      desired_anchor_quat = self.anchor_quat_w[visualizer.env_idx]
      desired_rotation_matrix = matrix_from_quat(desired_anchor_quat).cpu().numpy()
      visualizer.add_frame(
        position=desired_anchor_pos,
        rotation_matrix=desired_rotation_matrix,
        scale=0.1,
        label="desired_anchor",
        axis_colors=_DESIRED_FRAME_COLORS,
      )

      current_anchor_pos = self.robot_anchor_pos_w[visualizer.env_idx].cpu().numpy()
      current_anchor_quat = self.robot_anchor_quat_w[visualizer.env_idx]
      current_rotation_matrix = matrix_from_quat(current_anchor_quat).cpu().numpy()
      visualizer.add_frame(
        position=current_anchor_pos,
        rotation_matrix=current_rotation_matrix,
        scale=0.15,
        label="current_anchor",
      )


@dataclass(kw_only=True)
class MotionCommandCfg(CommandTermCfg):
  motion_file: str
  anchor_body_name: str
  body_names: tuple[str, ...]
  eef_body_names: tuple[str, ...]
  asset_name: str
  class_type: type[CommandTerm] = MotionCommand
  pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  joint_position_range: tuple[float, float] = (-0.52, 0.52)
  adaptive_kernel_size: int = 1
  adaptive_lambda: float = 0.8
  adaptive_uniform_ratio: float = 0.1
  adaptive_alpha: float = 0.001
  sampling_mode: Literal["adaptive", "uniform", "start"] = "adaptive"

  @dataclass
  class VizCfg:
    mode: Literal["ghost", "frames"] = "ghost"
    ghost_color: tuple[float, float, float, float] = (0.5, 0.7, 0.5, 0.5)

  viz: VizCfg = field(default_factory=VizCfg)
