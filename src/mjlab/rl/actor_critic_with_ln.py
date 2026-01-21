# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any, NoReturn

import torch
import torch.nn as nn
from rsl_rl.networks import MLP, EmpiricalNormalization
from torch.distributions import Normal


class ActorCriticLayerNorm(nn.Module):
  is_recurrent: bool = False

  def __init__(
    self,
    obs,
    obs_groups: dict[str, list[str]],
    num_actions: int,
    actor_obs_normalization: bool = False,
    critic_obs_normalization: bool = False,
    actor_hidden_dims: tuple[int] | list[int] = [256, 256, 256],  # noqa: B006
    critic_hidden_dims: tuple[int] | list[int] = [256, 256, 256],  # noqa: B006
    activation: str = "elu",
    init_noise_std: float = 1.0,
    noise_std_type: str = "scalar",
    state_dependent_std: bool = False,
    layernorm_eps: float = 1e-5,  # noqa: B008
    **kwargs: dict[str, Any],
  ) -> None:
    if kwargs:
      print(
        "ActorCriticLayerNorm.__init__ got unexpected arguments, which will be ignored: "
        + str(kwargs.keys()),
      )
    super().__init__()

    # Get the observation dimensions
    self.obs_groups = obs_groups
    num_actor_obs = 0
    for obs_group in obs_groups["policy"]:
      assert len(obs[obs_group].shape) == 2, (
        "The ActorCriticLayerNorm module only supports 1D observations."
      )
      num_actor_obs += obs[obs_group].shape[-1]
    num_critic_obs = 0
    for obs_group in obs_groups["critic"]:
      assert len(obs[obs_group].shape) == 2, (
        "The ActorCriticLayerNorm module only supports 1D observations."
      )
      num_critic_obs += obs[obs_group].shape[-1]

    # Actor (MLP-based)
    self.state_dependent_std = state_dependent_std
    # Input to actor MLP: concatenated actor observation vector
    if self.state_dependent_std:
      self.actor = MLP(num_actor_obs, [2, num_actions], actor_hidden_dims, activation)
    else:
      self.actor = MLP(num_actor_obs, num_actions, actor_hidden_dims, activation)
    print(f"Actor MLP: {self.actor}")

    # Actor observation normalization
    self.actor_obs_normalization = actor_obs_normalization
    if actor_obs_normalization:
      self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
    else:
      self.actor_obs_normalizer = torch.nn.Identity()

    # Actor LayerNorm applied to actor observation vector before MLP
    self.actor_layernorm = nn.LayerNorm(num_actor_obs, eps=layernorm_eps)

    # Critic (MLP-based)
    self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)
    print(f"Critic MLP: {self.critic}")

    # Critic observation normalization
    self.critic_obs_normalization = critic_obs_normalization
    if critic_obs_normalization:
      self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
    else:
      self.critic_obs_normalizer = torch.nn.Identity()

    # Critic LayerNorm applied to critic observation vector before MLP
    self.critic_layernorm = nn.LayerNorm(num_critic_obs, eps=layernorm_eps)

    # Action noise
    self.noise_std_type = noise_std_type
    if self.state_dependent_std:
      torch.nn.init.zeros_(self.actor[-2].weight[num_actions:])  # type: ignore[index]
      if self.noise_std_type == "scalar":
        torch.nn.init.constant_(self.actor[-2].bias[num_actions:], init_noise_std)  # type: ignore[index]
      elif self.noise_std_type == "log":
        torch.nn.init.constant_(
          self.actor[-2].bias[num_actions:],  # type: ignore[index]
          torch.log(torch.tensor(init_noise_std + 1e-7)),  # type: ignore[arg-type]
        )
      else:
        raise ValueError(
          f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'"
        )
    else:
      if self.noise_std_type == "scalar":
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
      elif self.noise_std_type == "log":
        self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
      else:
        raise ValueError(
          f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'"
        )

    # Action distribution
    # Note: Populated in update_distribution
    self.distribution = None

    # Disable args validation for speedup
    Normal.set_default_validate_args(False)

  @property
  def action_mean(self) -> torch.Tensor:
    return self.distribution.mean  # type: ignore[return-value]

  @property
  def action_std(self) -> torch.Tensor:
    return self.distribution.stddev  # type: ignore[return-value]

  @property
  def entropy(self) -> torch.Tensor:
    return self.distribution.entropy().sum(dim=-1)  # type: ignore[return-value]

  def reset(self, dones: torch.Tensor | None = None) -> None:
    # No recurrent state to reset for MLP-only actor-critic
    return None

  def forward(self) -> NoReturn:
    raise NotImplementedError

  def _update_distribution(self, obs: torch.Tensor) -> None:
    if self.state_dependent_std:
      # Compute mean and standard deviation
      mean_and_std = self.actor(self.actor_layernorm(obs))
      if self.noise_std_type == "scalar":
        mean, std = torch.unbind(mean_and_std, dim=-2)
      elif self.noise_std_type == "log":
        mean, log_std = torch.unbind(mean_and_std, dim=-2)
        std = torch.exp(log_std)
      else:
        raise ValueError(
          f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'"
        )
    else:
      # Compute mean
      mean = self.actor(self.actor_layernorm(obs))
      # Compute standard deviation
      if self.noise_std_type == "scalar":
        std = self.std.expand_as(mean)
      elif self.noise_std_type == "log":
        std = torch.exp(self.log_std).expand_as(mean)
      else:
        raise ValueError(
          f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'"
        )
    # Create distribution
    self.distribution = Normal(mean, std)

  def act(self, obs) -> torch.Tensor:
    obs = self.get_actor_obs(obs)
    obs = self.actor_obs_normalizer(obs)
    # No RNN: pass normalized observation vector into actor
    self._update_distribution(obs)
    return self.distribution.sample()  # type: ignore[return-value]

  def act_inference(self, obs) -> torch.Tensor:
    obs = self.get_actor_obs(obs)
    obs = self.actor_obs_normalizer(obs)
    # MLP-only inference
    if self.state_dependent_std:
      return self.actor(self.actor_layernorm(obs))[..., 0, :]
    else:
      return self.actor(self.actor_layernorm(obs))

  def evaluate(self, obs) -> torch.Tensor:
    obs = self.get_critic_obs(obs)
    obs = self.critic_obs_normalizer(obs)
    # MLP-only evaluation (apply critic LayerNorm)
    return self.critic(self.critic_layernorm(obs))

  def get_actor_obs(self, obs) -> torch.Tensor:
    obs_list = [obs[obs_group] for obs_group in self.obs_groups["policy"]]
    return torch.cat(obs_list, dim=-1)

  def get_critic_obs(self, obs) -> torch.Tensor:
    obs_list = [obs[obs_group] for obs_group in self.obs_groups["critic"]]
    return torch.cat(obs_list, dim=-1)

  def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
    return self.distribution.log_prob(actions).sum(dim=-1)  # type: ignore[return-value]

  def update_normalization(self, obs) -> None:
    if self.actor_obs_normalization:
      actor_obs = self.get_actor_obs(obs)
      self.actor_obs_normalizer.update(actor_obs)  # type: ignore[union-attr]
    if self.critic_obs_normalization:
      critic_obs = self.get_critic_obs(obs)
      self.critic_obs_normalizer.update(critic_obs)  # type: ignore[union-attr]

  def load_state_dict(self, state_dict, strict=True):  # type: ignore
    """Load the parameters of the actor-critic model.

    Args:
        state_dict (dict): State dictionary of the model.
        strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                        module's state_dict() function.

    Returns:
        bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
    """

    super().load_state_dict(state_dict, strict=strict)
    return True  # training resumes
