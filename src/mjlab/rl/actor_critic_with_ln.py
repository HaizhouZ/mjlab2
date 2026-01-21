# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any, NoReturn

import torch
import torch.nn as nn
from rsl_rl.networks import EmpiricalNormalization
from rsl_rl.utils import resolve_nn_activation
from tensordict import TensorDict
from torch.distributions import Normal


def make_layernorm_block(
  in_dim: int,
  out_dim: int,
  ln_pos: str = "pre",
  activation: nn.Module | None = None,
) -> nn.Module:
  """Create a block with LayerNorm, Linear, and activation (no activation if None)."""
  if activation is None:
    activation = nn.Identity()
  if ln_pos == "pre":
    block = nn.Sequential(
      nn.LayerNorm(in_dim),
      nn.Linear(in_dim, out_dim),
      activation,
    )
  elif ln_pos == "post":
    block = nn.Sequential(
      nn.Linear(in_dim, out_dim),
      nn.LayerNorm(out_dim),
      activation,
    )
  else:
    raise ValueError(
      f"Invalid ln_pos '{ln_pos}' or in_dim '{in_dim}' or out_dim '{out_dim}'"
    )
  return block


class residual_block(nn.Module):
  def __init__(
    self,
    input_dim: int,
    out_dim: int,
    ln_pos: str = "pre",
    activation: nn.Module | None = None,
  ) -> None:
    super().__init__()
    if activation is None:
      activation = nn.Mish()
    self.block = nn.Sequential(
      make_layernorm_block(input_dim, out_dim, ln_pos, activation),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return x + self.block(x)


def make_network(
  input_dim: int,
  output_dim: int,
  res_dim: int,
  num_res_blocks: int,
  input_activation: nn.Module | None = None,
  res_activation: nn.Module | None = None,
  output_activation: nn.Module | None = None,
) -> nn.Module:
  # make a BRO-like network with layernorm
  layers: list[nn.Module] = []
  # input layer
  layers.append(make_layernorm_block(input_dim, res_dim, "pre", input_activation))
  # residual blocks
  for _ in range(num_res_blocks):
    layers.append(residual_block(res_dim, res_dim, "pre", res_activation))
  # output layer
  layers.append(make_layernorm_block(res_dim, output_dim, "pre", output_activation))
  return nn.Sequential(*layers)


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
    activation_fn = resolve_nn_activation(activation)
    self.actor = make_network(
      num_actor_obs,
      2 * num_actions if self.state_dependent_std else num_actions,
      res_dim=actor_hidden_dims[0],
      num_res_blocks=len(actor_hidden_dims),
      input_activation=activation_fn,
      res_activation=activation_fn,
      output_activation=None,
    )
    print(f"Actor Network: {self.actor}")

    # Actor observation normalization
    self.actor_obs_normalization = actor_obs_normalization
    if actor_obs_normalization:
      self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
    else:
      self.actor_obs_normalizer = torch.nn.Identity()

    # Critic (MLP-based)
    self.critic = make_network(
      num_critic_obs,
      1,
      res_dim=critic_hidden_dims[0],
      num_res_blocks=len(critic_hidden_dims),
      input_activation=activation_fn,
      res_activation=activation_fn,
      output_activation=None,
    )
    print(f"Critic Network: {self.critic}")

    # Critic observation normalization
    self.critic_obs_normalization = critic_obs_normalization
    if critic_obs_normalization:
      self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
    else:
      self.critic_obs_normalizer = torch.nn.Identity()

    # Action noise
    self.noise_std_type = noise_std_type
    actor_output_layer = self.actor[-1][1]  # type: ignore[index]
    if self.state_dependent_std:
      torch.nn.init.zeros_(actor_output_layer.weight[num_actions:])  # type: ignore[index]
      if self.noise_std_type == "scalar":
        torch.nn.init.constant_(actor_output_layer.bias[num_actions:], init_noise_std)  # type: ignore[index]
      elif self.noise_std_type == "log":
        torch.nn.init.constant_(
          actor_output_layer.bias[num_actions:],  # type: ignore[index]
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
      mean_and_std = self.actor(obs)
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
      mean = self.actor(obs)
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

  def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
    obs = self.get_actor_obs(obs)  # type: ignore
    obs = self.actor_obs_normalizer(obs)
    self._update_distribution(obs)
    assert isinstance(self.distribution, Normal), (
      "Action distribution has not been initialized."
    )
    return self.distribution.sample()

  def act_inference(self, obs: TensorDict) -> torch.Tensor:
    obs = self.get_actor_obs(obs)  # type: ignore
    obs = self.actor_obs_normalizer(obs)
    if self.state_dependent_std:
      return self.actor(obs)[..., 0, :]
    else:
      return self.actor(obs)

  def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
    obs = self.get_critic_obs(obs)  # type: ignore
    obs = self.critic_obs_normalizer(obs)
    return self.critic(obs)

  def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
    obs_list = [obs[obs_group] for obs_group in self.obs_groups["policy"]]
    return torch.cat(obs_list, dim=-1)

  def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
    obs_list = [obs[obs_group] for obs_group in self.obs_groups["critic"]]
    return torch.cat(obs_list, dim=-1)

  def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
    assert isinstance(self.distribution, Normal), (
      "Action distribution has not been initialized."
    )
    return self.distribution.log_prob(actions).sum(dim=-1)

  def update_normalization(self, obs: TensorDict) -> None:
    if self.actor_obs_normalization:
      actor_obs = self.get_actor_obs(obs)
      self.actor_obs_normalizer.update(actor_obs)  # type: ignore
    if self.critic_obs_normalization:
      critic_obs = self.get_critic_obs(obs)
      self.critic_obs_normalizer.update(critic_obs)  # type: ignore

  def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:  # type: ignore
    """Load the parameters of the actor-critic model.

    Args:
        state_dict: State dictionary of the model.
        strict: Whether to strictly enforce that the keys in `state_dict` match the keys returned by this module's
            :meth:`state_dict` function.

    Returns:
        Whether this training resumes a previous training. This flag is used by the :func:`load` function of
            :class:`OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
    """
    super().load_state_dict(state_dict, strict=strict)
    return True
