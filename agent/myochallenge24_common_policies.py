# import copy
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union, no_type_check

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.preprocessing import is_image_space, maybe_transpose
from stable_baselines3.common.utils import is_vectorized_observation


class Flatten(nn.Module):
    """
    Equivalent to PyTorch nn.Flatten() layer.
    """

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x.reshape((x.shape[0], -1))


class BaseJaxPolicy(BasePolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
        )

    @staticmethod
    @jax.jit
    # def sample_action(actor_state, obervations, key):
    # dist = actor_state.apply_fn(actor_state.params, obervations)
    # action = dist.sample(seed=key)
    # return action
    def sample_action(
        actor_params,
        actor_state,
        observations,
        key,
        synergies_state,
        dynamics_state,
        phase,
    ):
        # key, subkey = jax.random.split(key)
        # squash_controls_fn = lambda u: jnp.concatenate([nn.sigmoid(u[...,:63]*5.),nn.tanh(u[...,63:]*3.)],axis=-1)
        # _, _, mu_u_source, A, explore_var, A_complement, _, _, _, _, _ = synergies_state.apply_fn(synergies_state.params, observations[...,:216], dynamics_state, key, False, jnp.zeros((observations.shape[0],1)), squash_controls_fn)
        # key, subkey = jax.random.split(key)
        # base_dist, explore_dist, A, mu_u_source, A_complement, squash_controls_fn = actor_state.apply_fn(actor_params, observations, synergies_state, dynamics_state, subkey,  mu_u_source, A, explore_var, A_complement, deterministic=False)
        key, subkey = jax.random.split(key)
        base_dist, explore_dist, A, mu_u_source, A_complement, squash_controls_fn = (
            actor_state.apply_fn(
                actor_params,
                observations,
                synergies_state,
                dynamics_state,
                subkey,
                deterministic=False,
            )
        )
        key, subkey = jax.random.split(key)
        # actor_z_actions = base_dist.sample(seed=subkey)
        sampled_actions = base_dist.sample(seed=subkey)
        # key, subkey = jax.random.split(key)
        # MLP_actions = MPL_dist.sample(seed=subkey)
        actor_z_actions, MLP_actions = jnp.split(
            sampled_actions, [A.shape[-1]], axis=-1
        )
        MLP_all_actions = jnp.zeros((actor_z_actions.shape[0], 17))

        # MLP_all_actions = MLP_all_actions.at[...,4:8].set(MLP_actions[...,:4])
        # MLP_all_actions = MLP_all_actions.at[...,8:11].set(MLP_actions[...,4][:,None].repeat(3,axis=-1)) # thumb flexion synergy
        # MLP_all_actions = MLP_all_actions.at[...,11:13].set(MLP_actions[...,5:7])
        # MLP_all_actions = MLP_all_actions.at[...,13:].set(MLP_actions[...,7][:,None].repeat(4,axis=-1)) # fingers flexion synergy
        # actor_actions = actor_actions.at[...,63:67].set(jnp.array([0.325, -0.907, -0.259, 1.17]))

        # arm
        # MLP_all_actions = MLP_all_actions.at[...,:4].set(MLP_actions[...,:4])

        # # 3 wrist and thumb opposition
        # MLP_all_actions = MLP_all_actions.at[...,4:8].set(MLP_actions[...,4:8])

        # deviation synergy
        # MLP_all_actions = MLP_all_actions.at[...,11].set(0.) # 0.17
        # MLP_all_actions = MLP_all_actions.at[...,15].set(0.) # 0.17

        # # 5 digit flexion-extension synergy
        # MLP_all_actions = MLP_all_actions.at[...,8:11].set(MLP_actions[...,8][:,None].repeat(3,axis=-1))
        # MLP_all_actions = MLP_all_actions.at[...,12:15].set(MLP_actions[...,8][:,None].repeat(3,axis=-1))
        # MLP_all_actions = MLP_all_actions.at[...,16].set(MLP_actions[...,8])

        exploration_noise = explore_dist.sample(seed=key)
        full_actions = jnp.concatenate(
            (
                mu_u_source[..., :63]
                + jnp.einsum("...ij,...j->...i", A, actor_z_actions)
                + jnp.einsum("...ij,...j->...i", A_complement, exploration_noise),
                MLP_all_actions,
            ),
            axis=-1,
        )
        actor_actions = squash_controls_fn((full_actions))

        def false_fun(actor_actions):
            actor_actions = actor_actions.at[63:67].set(
                jnp.array([0.988, 0.0, 1.07, 1.24])
            )
            actor_actions = actor_actions.at[67:71].set(
                jnp.array([0.848, -0.0343, 0.41, 1.47])
            )
            actor_actions = actor_actions.at[71:74].set(jnp.array([0.0, 0.0, 0.0]))
            actor_actions = actor_actions.at[74].set(0.0)
            actor_actions = actor_actions.at[75:78].set(
                jnp.array([0.112, 0.712, 0.928])
            )
            actor_actions = actor_actions.at[78].set(0.0)
            actor_actions = actor_actions.at[79].set(1.06)

            return actor_actions

        def true_fun(actor_actions):
            # actor_actions = actor_actions.at[:63].set(jnp.zeros(63))
            actor_actions = actor_actions.at[63:67].set(
                jnp.array([0.988, 0.0, 1.07, 1.24])
            )  # second element -1.15 is pillar; -0.15 is moved a little
            actor_actions = actor_actions.at[67:71].set(
                jnp.array([0.848, -0.0343, 0.41, 1.47])
            )  # first element to -.25 (rotate wrist)
            actor_actions = actor_actions.at[73:75].set(jnp.array([0.0, 0.0]))
            actor_actions = actor_actions.at[77].set(0.928)
            actor_actions = actor_actions.at[78].set(0.0)
            actor_actions = actor_actions.at[79].set(1.06)

            return actor_actions

        def conditional_actions(phase, actor_actions):
            # actor_actions = jax.lax.cond(False, true_fun, false_fun, actor_actions)
            actor_actions = jax.lax.cond(
                jnp.isclose(phase, 2), true_fun, false_fun, actor_actions
            )

            return actor_actions

        batch_conditional_actions = jax.vmap(conditional_actions)

        actor_actions = batch_conditional_actions(phase, actor_actions)

        negative_entropy = -base_dist.entropy()
        return actor_actions, negative_entropy, explore_dist

    @staticmethod
    @jax.jit
    # def select_action(actor_state, obervations):
    #     return actor_state.apply_fn(actor_state.params, obervations).mode()
    def select_action(
        actor_params,
        actor_state,
        observations,
        key,
        synergies_state,
        dynamics_state,
        phase,
    ):
        key, subkey = jax.random.split(key)
        base_dist, explore_dist, A, mu_u_source, A_complement, squash_controls_fn = (
            actor_state.apply_fn(
                actor_params,
                observations,
                synergies_state,
                dynamics_state,
                subkey,
                deterministic=True,
            )
        )
        key, subkey = jax.random.split(key)
        sampled_actions = base_dist.mode()
        actor_z_actions, MLP_actions = jnp.split(
            sampled_actions, [A.shape[-1]], axis=-1
        )
        MLP_all_actions = jnp.zeros((actor_z_actions.shape[0], 17))

        exploration_noise = explore_dist.sample(seed=key)
        full_actions = jnp.concatenate(
            (
                mu_u_source[..., :63]
                + jnp.einsum("...ij,...j->...i", A, actor_z_actions)
                + jnp.einsum("...ij,...j->...i", A_complement, exploration_noise),
                MLP_all_actions,
            ),
            axis=-1,
        )
        actor_actions = squash_controls_fn((full_actions))

        # def false_fun(actor_actions):

        #     actor_actions = actor_actions.at[63:67].set(jnp.array([0.988, 0., 1.07, 1.24]))
        #     actor_actions = actor_actions.at[67:71].set(jnp.array([0.848, -.0343, 0.41, 1.47])) # first element to -.25 (rotate wrist)
        #     actor_actions = actor_actions.at[71:74].set(jnp.array([0., 0., 0.]))
        #     actor_actions = actor_actions.at[74].set(0.)
        #     actor_actions = actor_actions.at[75:78].set(jnp.array([0.112, 0.712, 0.928]))
        #     actor_actions = actor_actions.at[78].set(0.)
        #     actor_actions = actor_actions.at[79].set(1.06)

        #     return actor_actions

        # def true_fun(actor_actions):

        #     # actor_actions = actor_actions.at[:63].set(jnp.zeros(63))
        #     actor_actions = actor_actions.at[63:67].set(jnp.array([0.988, 0., 1.07, 1.24]))
        #     # actor_actions = actor_actions.at[63:67].set(jnp.array([0.988, -0.15, 1.07, 1.24])) # second element -1.15 is pillar
        #     actor_actions = actor_actions.at[67:71].set(jnp.array([0.848, -.0343, 0.41, 1.47])) # first element to -.25 (rotate wrist)
        #     actor_actions = actor_actions.at[73:75].set(jnp.array([0., 0.]))
        #     actor_actions = actor_actions.at[77].set(0.928)
        #     actor_actions = actor_actions.at[78].set(0.)
        #     actor_actions = actor_actions.at[79].set(1.06)

        #     return actor_actions

        # def conditional_actions(phase, actor_actions):

        #     actor_actions = jax.lax.cond(jnp.isclose(phase,2), true_fun, false_fun, actor_actions)

        #     return actor_actions

        # batch_conditional_actions = jax.vmap(conditional_actions)

        # actor_actions = batch_conditional_actions(phase[None], actor_actions)

        return actor_actions

    @no_type_check
    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        # self.set_training_mode(False)

        observation, vectorized_env = self.prepare_obs(observation)

        touching_myo, touching_mpl, phase = self.get_phase(observation)

        # if observation[...,-1]==0.: # at reset (timepoint 1)
        # self.get_target_qpos(observation, np.array([0., 0., 0.25]))

        # normalize obs and throw away time observation
        normalized_observation = self.normalize_obs(observation)
        # ec_stats.normalize_obs(obs)[...,:-1]

        actions = self._predict(
            normalized_observation, phase, deterministic=deterministic
        )

        # Convert to numpy, and reshape to the original action shape
        actions = np.array(actions).reshape((-1, *self.action_space.shape))

        if isinstance(self.action_space, spaces.Box):
            if self.squash_output:
                # Clip due to numerical instability
                actions = np.clip(actions, -1, 1)
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(
                    actions, self.action_space.low, self.action_space.high
                )

        actions = self.new_actions(
            actions, observation, touching_myo, touching_mpl, phase
        )

        # Remove batch dimension if needed
        if not vectorized_env:
            actions = actions.squeeze(axis=0)  # type: ignore[call-overload]

        return actions, state

    def new_actions(self, actions, observation, touching_myo, touching_mpl, phase):
        if phase == 1:
            actions[:, 63:67] = np.array([0.988, 0.0, 1.07, 1.24])
            actions[:, 67:71] = np.array([0.848, -0.0343, 0.41, 1.47])
            actions[:, 71:74] = np.array([0.0, 0.0, 0.0])
            actions[:, 74] = 0
            actions[:, 75:78] = np.array([0.112, 0.712, 0.928])
            actions[:, 78] = 0
            actions[:, 79] = 1.06

        elif phase == 2:
            actions[:, 63:67] = np.array([0.988, 0.0, 1.07, 1.24])
            actions[:, 67:71] = np.array([0.848, -0.0343, 0.41, 1.47])
            actions[:, 73:75] = np.array([0.0, 0.0])
            actions[:, 77] = 0.928
            actions[:, 78] = 0.0
            actions[:, 79] = 1.06

        if self.touching_mpl_count > 0:
            actions[:, :63] = np.zeros(63)

            # # thumb synergy, 0-1 (1 flexed)
            # actions[:,71:73] = np.ones(2)*1.

            # # fingers synergy, 0-1.6 (1.6 flexed)
            # actions[:,75:78] = np.ones(3)*1.6
            # actions[:,79] = np.ones(1)*1.6

        if (
            self.touching_mpl_count > 10
            and self.touching_myo_count > 10
            and self.phase_three == False
        ):
            self.phase_three = True
            self.get_target_qpos(observation, np.array([0, 0.05, 0.1]))

        if (
            np.linalg.norm(observation[0, 139:143] - self.target_qpos) < 0.15
            and self.phase_three
            and self.phase_four == False
        ):
            self.phase_four = True
            self.get_target_qpos(observation, np.array([0.0, 0.05, 0.1]))

        if (
            np.linalg.norm(observation[0, 139:143] - self.target_qpos) < 0.15
            and self.phase_three
            and self.phase_four
            and self.phase_five == False
        ):
            self.phase_five = True

        if self.phase_three:
            actions[:, 63:67] = (
                self.target_qpos
            )  # self.get_target_qpos(observation, np.array([0., 0., 0.25]))

        if self.phase_four:
            pass
            # (rotate wrist)
            # actions[:,67] = np.ones(1)*-.25

        if self.phase_five:
            # thumb synergy, 0-1 (1 flexed)
            actions[:, 71:73] = np.ones(2) * 0.0

            # fingers synergy, 0-1.6 (1.6 flexed)
            actions[:, 75:78] = np.ones(3) * 0.0
            actions[:, 79] = np.ones(1) * 0.0

        return actions

    def get_target_qpos(self, observation, delta_goal):
        goal_pos = observation[0, 194:197].copy()
        init_qpos = observation[0, 139:143].copy()
        self.target_qpos = self.ik.calculate(
            goal_pos.copy() + delta_goal,
            init_qpos.copy(),
            self.body_id,
            self.prosthesis_ids.copy(),
        )

    def normalize_obs(self, observation):
        # add dummy observations
        enlarged_obs = np.concatenate(
            (observation, np.zeros(observation.shape[:-1] + (2,))), axis=-1
        )

        # ignore time and dummy observations
        normalized_observation = self.vec_stats.normalize_obs(enlarged_obs)[..., :-3]

        return normalized_observation

    def get_phase(self, observation):
        observation = observation.reshape((-1, *self.observation_space.shape))

        # how does batch obs work, not in jax here
        # if observation[...,-1]==0.: # at reset (timepoint 1)
        if observation[..., -1] == 0.0:  # at reset (timepoint 1)
            self.phase_two = np.array(False)
            self.phase_three = np.array(False)
            self.phase_four = np.array(False)
            self.phase_five = np.array(False)
            self.touching_myo_count = 0
            self.touching_mpl_count = 0

        touching_myo = observation[..., -6]
        touching_mpl = observation[..., -5]

        if touching_myo:
            self.touching_myo_count = 0
        else:
            self.touching_myo_count += 1

        if touching_mpl:
            self.touching_mpl_count += 1
        else:
            self.touching_mpl_count = 0

        phase_two_criterion = touching_mpl
        phase_three_criterion = touching_mpl and not touching_myo

        # if phase_three_criterion:
        #     self.phase_three=np.array(True)
        # elif not phase_three_criterion and self.max_num_mode_switches == 'unbounded':
        #     self.phase_three=np.array(False)

        if phase_two_criterion:
            self.phase_two = np.array(True)
        elif not phase_two_criterion and self.max_num_mode_switches == "unbounded":
            self.phase_two = np.array(False)

        # if self.phase_three:
        #     phase=np.array(3)
        # elif self.phase_two:
        #     phase=np.array(2)
        if self.phase_two:
            phase = np.array(2)
        else:
            phase = np.array(1)

        return touching_myo, touching_mpl, phase

    def prepare_obs(
        self, observation: Union[np.ndarray, Dict[str, np.ndarray]]
    ) -> Tuple[np.ndarray, bool]:
        vectorized_env = False
        if isinstance(observation, dict):
            assert isinstance(self.observation_space, spaces.Dict)
            # Minimal dict support: flatten
            keys = list(self.observation_space.keys())
            vectorized_env = is_vectorized_observation(
                observation[keys[0]], self.observation_space[keys[0]]
            )

            # Add batch dim and concatenate
            observation = np.concatenate(
                [
                    observation[key].reshape(-1, *self.observation_space[key].shape)
                    for key in keys
                ],
                axis=1,
            )
            # need to copy the dict as the dict in VecFrameStack will become a torch tensor
            # observation = copy.deepcopy(observation)
            # for key, obs in observation.items():
            #     obs_space = self.observation_space.spaces[key]
            #     if is_image_space(obs_space):
            #         obs_ = maybe_transpose(obs, obs_space)
            #     else:
            #         obs_ = np.array(obs)
            #     vectorized_env = vectorized_env or is_vectorized_observation(obs_, obs_space)
            #     # Add batch dimension if needed
            #     observation[key] = obs_.reshape((-1, *self.observation_space[key].shape))

        elif is_image_space(self.observation_space):
            # Handle the different cases for images
            # as PyTorch use channel first format
            observation = maybe_transpose(observation, self.observation_space)

        else:
            observation = np.array(observation)

        if not isinstance(self.observation_space, spaces.Dict):
            assert isinstance(observation, np.ndarray)
            vectorized_env = is_vectorized_observation(
                observation, self.observation_space
            )
            # Add batch dimension if needed
            observation = observation.reshape((-1, *self.observation_space.shape))  # type: ignore[misc]

        assert isinstance(observation, np.ndarray)
        return observation, vectorized_env

    def set_training_mode(self, mode: bool) -> None:
        # self.actor.set_training_mode(mode)
        # self.critic.set_training_mode(mode)
        self.training = mode


class ContinuousCritic(nn.Module):
    net_arch: Sequence[int]
    use_layer_norm: bool = False
    dropout_rate: Optional[float] = None
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, x: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        x = Flatten()(x)
        x = jnp.concatenate([x, action], -1)
        for n_units in self.net_arch:
            x = nn.Dense(n_units)(x)
            if self.dropout_rate is not None and self.dropout_rate > 0:
                x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=False)
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)
            x = self.activation_fn(x)
        x = nn.Dense(1)(x)
        return x


class VectorCritic(nn.Module):
    net_arch: Sequence[int]
    obs_variables: List
    use_layer_norm: bool = False
    dropout_rate: Optional[float] = None
    n_critics: int = 2
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, obs: jnp.ndarray, action: jnp.ndarray):
        # Idea taken from https://github.com/perrin-isir/xpag
        # Similar to https://github.com/tinkoff-ai/CORL for PyTorch
        vmap_critic = nn.vmap(
            ContinuousCritic,
            variable_axes={"params": 0},  # parameters not shared between the critics
            split_rngs={"params": True, "dropout": True},  # different initializations
            in_axes=None,
            out_axes=0,
            axis_size=self.n_critics,
        )
        q_values = vmap_critic(
            use_layer_norm=self.use_layer_norm,
            dropout_rate=self.dropout_rate,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
        )(obs[..., self.obs_variables], action)
        return q_values
