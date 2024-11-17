from functools import partial
from typing import Any, ClassVar, Dict, Literal, Optional, Tuple, Type, Union, List, Callable

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from gymnasium import spaces
from jax.typing import ArrayLike
from stable_baselines3.common.buffers import ReplayBuffer
# from my_buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule

# from sbx.common.off_policy_algorithm import OffPolicyAlgorithmJax
# from my_sbx_off_policy_algorithm import OffPolicyAlgorithmJax
# from my_sbx_off_policy_algorithm_24 import OffPolicyAlgorithmJax
from myochallenge24_my_sbx_off_policy_algorithm import OffPolicyAlgorithmJax
from sbx.common.type_aliases import ReplayBufferSamplesNp, RLTrainState
# from sbx.sac.policies import SACPolicy
from myochallenge24_policy import SACPolicy
# import distrax
# from common_policies_24 import BaseJaxPolicy
from myochallenge24_common_policies import BaseJaxPolicy

class EntropyCoef(nn.Module):
    ent_coef_init: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_ent_coef = self.param("log_ent_coef", init_fn=lambda key: jnp.full((), jnp.log(self.ent_coef_init)))
        return jnp.exp(log_ent_coef)
        # log_ent_coef = self.param("log_ent_coef", init_fn=lambda key: jnp.full((), jnp.log(jnp.exp(self.ent_coef_init)-1.)))
        # return nn.softplus(log_ent_coef)

class ConstantEntropyCoef(nn.Module):
    ent_coef_init: float = 1.0

    @nn.compact
    def __call__(self) -> float:
        # Hack to not optimize the entropy coefficient while not having to use if/else for the jit
        # TODO: add parameter in train to remove that hack
        self.param("dummy_param", init_fn=lambda key: jnp.full((), self.ent_coef_init))
        return self.ent_coef_init


class SAC(OffPolicyAlgorithmJax):
    policy_aliases: ClassVar[Dict[str, Type[SACPolicy]]] = {  # type: ignore[assignment]
        "MlpPolicy": SACPolicy,
        # Minimal dict support using flatten()
        "MultiInputPolicy": SACPolicy,
    }

    policy: SACPolicy
    action_space: spaces.Box  # type: ignore[assignment]

    def __init__(
        self,
        policy,
        MjModel,
        max_num_mode_switches,
        vec_stats,
        env: Union[GymEnv, str],
        control_variables: List = [],
        task_variables: List = [],
        obs_variables: List = [],
        init_axes_scales: float = -3,
        power_coefficient: float = 0.05,
        init_power_coeff: float = 0.01,
        std_max: float = 1.,
        dynamics_dropout: float = 1e-1,
        cocontraction: str = None,
        num_MC_samples: int = 1,
        explore_coeff: float = 1.,
        learning_rate: Union[float, Schedule] = 3e-4,
        qf_learning_rate: Optional[float] = None,
        dynamics_learning_rate: float = 3e-4,
        source_mu: List = [],
        buffer_size: int = 1_000_000, # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        policy_delay: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        ent_coef: Union[str, float] = "auto",
        target_entropy: Union[Literal["auto"], float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: str = "auto",
        _init_setup_model: bool = True,
    ) -> None:
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            qf_learning_rate=qf_learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            seed=seed,
            supported_action_spaces=(spaces.Box,),
            support_multi_env=True,
        )

        self.MjModel = MjModel
        self.max_num_mode_switches = max_num_mode_switches
        self.vec_stats = vec_stats
        self.control_variables = control_variables
        self.task_variables = task_variables
        self.obs_variables = obs_variables
        self.policy_delay = policy_delay
        self.ent_coef_init = ent_coef
        self.target_entropy = target_entropy
        self.init_axes_scales = init_axes_scales
        self.power_coefficient = power_coefficient
        self.init_power_coeff = init_power_coeff
        self.std_max = std_max
        self.dynamics_dropout = dynamics_dropout
        self.cocontraction = cocontraction
        self.num_MC_samples = num_MC_samples
        self.explore_coeff = explore_coeff
        self.dynamics_learning_rate = dynamics_learning_rate
        self.source_mu = source_mu

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()

        if not hasattr(self, "policy") or self.policy is None:
            self.policy = self.policy_class(  # type: ignore[assignment]
                self.MjModel,
                self.max_num_mode_switches,
                self.vec_stats,
                self.control_variables,
                self.task_variables,
                self.obs_variables,
                self.observation_space,
                self.action_space,
                self.lr_schedule,
                self.init_axes_scales,
                self.power_coefficient,
                self.init_power_coeff,
                self.std_max,
                self.dynamics_dropout,
                self.cocontraction,
                self.num_MC_samples,
                self.explore_coeff,
                self.dynamics_learning_rate,
                self.source_mu,
                **self.policy_kwargs,
            )

            assert isinstance(self.qf_learning_rate, float)

            self.key = self.policy.build(self.key, self.lr_schedule, self.qf_learning_rate)

            self.key, ent_key = jax.random.split(self.key, 2)

            self.dynamics = self.policy.dynamics  # type: ignore[assignment]
            # self.source = self.policy.source  # type: ignore[assignment]
            self.synergies = self.policy.synergies  # type: ignore[assignment]
            self.actor = self.policy.actor  # type: ignore[assignment]
            self.qf = self.policy.qf  # type: ignore[assignment]

            # The entropy coefficient or entropy can be learned automatically
            # see Automating Entropy Adjustment for Maximum Entropy RL section
            # of https://arxiv.org/abs/1812.05905
            if isinstance(self.ent_coef_init, str) and self.ent_coef_init.startswith("auto"):
                # Default initial value of ent_coef when learned
                
                # ratio of entropies
                # init_var = 1.
                # ent_coef_init = (1+jnp.log(2*jnp.pi*init_var*self.power_target))/(1+jnp.log(2*jnp.pi*init_var))
                # can be negative, but alpha is only allowed to be positive alpha=exp(x), so next best thing make alpha small
                
                # difference of entropies
                # no way to use this with alpha
                # ent_diff = D * (-jnp.log(power))/2 # = D * (1+log(2*pi*var))/2 - D * (1+log(2*pi*var*power))/2
                
                ent_coef_init = 1.0
                if "_" in self.ent_coef_init:
                    ent_coef_init = float(self.ent_coef_init.split("_")[1])
                    assert ent_coef_init > 0.0, "The initial value of ent_coef must be greater than 0"

                # Note: we optimize the log of the entropy coeff which is slightly different from the paper
                # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
                self.ent_coef = EntropyCoef(ent_coef_init)
            else:
                # This will throw an error if a malformed string (different from 'auto') is passed
                assert isinstance(
                    self.ent_coef_init, float
                ), f"Entropy coef must be float when not equal to 'auto', actual: {self.ent_coef_init}"
                self.ent_coef = ConstantEntropyCoef(self.ent_coef_init)  # type: ignore[assignment]

            self.ent_coef_state = TrainState.create(
                apply_fn=self.ent_coef.apply,
                params=self.ent_coef.init(ent_key)["params"],
                tx=optax.adam(
                    learning_rate=self.learning_rate,
                ),
            )

        # Target entropy is used when learning the entropy coefficient
        if self.target_entropy == "auto":
            # automatically set target entropy if needed
            self.target_entropy = -np.prod(self.env.action_space.shape).astype(np.float32)  # type: ignore
        else:
            # Force conversion
            # this will also throw an error for unexpected string
            # self.target_entropy = float(self.target_entropy)
            self.target_entropy = np.float32(self.target_entropy)
        self.target_entropy_schedule = optax.schedules.linear_schedule(self.target_entropy*0.75, self.target_entropy, 250_000)
        self.num_timesteps_start = self.num_timesteps

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "SAC",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def train(self, gradient_steps: int, batch_size: int) -> None:
        assert self.replay_buffer is not None
        # Sample all at once for efficiency (so we can jit the for loop)
        data = self.replay_buffer.sample(batch_size * gradient_steps, env=self._vec_normalize_env)

        if isinstance(data.observations, dict):
            keys = list(self.observation_space.keys())  # type: ignore[attr-defined]
            obs = np.concatenate([data.observations[key].numpy() for key in keys], axis=1)
            next_obs = np.concatenate([data.next_observations[key].numpy() for key in keys], axis=1)
        else:
            obs = data.observations.numpy()
            next_obs = data.next_observations.numpy()

        # Convert to numpy
        data = ReplayBufferSamplesNp(  # type: ignore[assignment]
            obs,
            data.actions.numpy(),
            next_obs,
            data.dones.numpy().flatten(),
            data.rewards.numpy().flatten(),
        )

        (
            self.policy.qf_state,
            self.policy.actor_state,
            self.ent_coef_state,
            self.policy.dynamics_state,
            self.policy.source_state,
            self.policy.power_coef_state,
            self.policy.synergies_state,
            self.key,
            (actor_loss_value, qf_loss_value, ent_coef_loss, dynamics_loss, entropy, ent_coef_value, p_ratio_source, entropy_marginal_transition, epistemic_uncertainty, mu_power_share),
        ) = self._train(
            self.gamma,
            self.tau,
            self.target_entropy,
            # self.target_entropy_schedule(self.num_timesteps-self.num_timesteps_start),
            0.05,
            self.control_variables,
            gradient_steps,
            data,
            self.policy_delay,
            (self._n_updates + 1) % self.policy_delay,
            self.policy.qf_state,
            self.policy.actor_state,
            self.ent_coef_state,
            self.policy.dynamics_state,
            self.policy.source_state,
            self.policy.power_coef_state,
            self.policy.synergies_state,
            self.policy.posterior,
            self.key,
            self.policy.unscale_action,
            self._vec_normalize_env.unnormalize_obs,
        )
        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/actor_loss", actor_loss_value.item())
        self.logger.record("train/critic_loss", qf_loss_value.item())
        self.logger.record("train/dynamics_loss", dynamics_loss.item())
        self.logger.record("train/ent_coef_loss", ent_coef_loss.item())
        self.logger.record("train/ent_coef_value", ent_coef_value.item())
        self.logger.record("train/entropy", entropy.item())
        self.logger.record("train/entropy_target", self.target_entropy.item())
        self.logger.record("train/p_ratio_source", p_ratio_source.item())
        self.logger.record("train/entropy_marginal_transition", entropy_marginal_transition.item())
        self.logger.record("train/mu_power_share", mu_power_share.item())
        self.logger.record("train/orthogonal_complement_std", epistemic_uncertainty.item())

    @staticmethod
    @jax.jit
    def update_dynamics(
        dynamics_state: TrainState,
        observations: jax.Array,
        actions: jax.Array,
        next_observations: jax.Array,
        control_variables: List,
        key: jax.Array,
    ):

        def dynamics_loss_fn(params, key):
            # dynamics_loss = dynamics_state.apply_fn(params, observations, actions, next_observations, key)
            def log_likelihood_diagonal_Gaussian(x, mu, log_var):
                """
                Calculate the log likelihood of x under a diagonal Gaussian distribution
                var_min is added to the variances for numerical stability
                """
                log_likelihood = -0.5 * (log_var + jnp.log(2 * jnp.pi) + (x - mu) ** 2 / jnp.exp(log_var))
                return log_likelihood
                # return -(x - mu) ** 2
            s_prime_mean, s_prime_log_var = dynamics_state.apply_fn(params, observations, actions, key, deterministic=False)
            dynamics_loss = -log_likelihood_diagonal_Gaussian(next_observations[:,control_variables]-observations[:,control_variables], s_prime_mean, s_prime_log_var).sum(axis=-1)
            dynamics_loss /= len(control_variables)
            return dynamics_loss.mean()

        key, subkey = jax.random.split(key)

        dynamics_loss, grads = jax.value_and_grad(dynamics_loss_fn, has_aux=False)(dynamics_state.params, subkey)
        dynamics_state = dynamics_state.apply_gradients(grads=grads)

        return dynamics_state, dynamics_loss, key

    staticmethod
    @jax.jit
    def update_synergies(
        synergies_state: TrainState,
        dynamics_state: TrainState,
        observations: jax.Array,
        key: jax.Array,
    ):

        def synergies_loss_fn(params, observations, dynamics_state, key):
            entropy_marginal_transition, p_ratio_source, _, _, explore_var, _, mu_power_share, _, _, _, _ = synergies_state.apply_fn(params, observations, dynamics_state, key, deterministic=False)
            return -entropy_marginal_transition.mean(), (explore_var.mean(), mu_power_share.mean())

        key, subkey = jax.random.split(key)

        (synergies_loss, (explore_var, mu_power_share)), grads = jax.value_and_grad(synergies_loss_fn, has_aux=True)(synergies_state.params, observations, dynamics_state, subkey)
        synergies_state = synergies_state.apply_gradients(grads=grads)

        return synergies_state, -synergies_loss, explore_var, mu_power_share, key

    @staticmethod
    @jax.jit
    def update_source(
        source_state: TrainState,
        dynamics_state: TrainState,
        power_coef_state: TrainState,
        observations: jax.Array,
        key: jax.Array,
    ):

        def source_loss(params, key):
            power_coef_value = power_coef_state.apply_fn({"params": power_coef_state.params})
            loss, ll_a_inverse, ll_a, effective_dimensionality, avg_power, axes_scales = source_state.apply_fn(params, observations, key, dynamics_state.params, power_coef_value)
            return loss.mean(), (ll_a_inverse.mean(), ll_a.mean(), effective_dimensionality.mean(), avg_power.mean(), axes_scales)

        key, subkey = jax.random.split(key)

        (source_loss_value, (ll_a_inverse, ll_a, effective_dimensionality, avg_power, axes_scales)), grads = jax.value_and_grad(source_loss, has_aux=True)(source_state.params, subkey)
        source_state = source_state.apply_gradients(grads=grads)

        return source_state, (ll_a_inverse, ll_a, effective_dimensionality, avg_power, axes_scales, key)

    @staticmethod
    @jax.jit
    def update_power_coef(power_coef_state: TrainState, avg_power: float, power_target: float):
        def power_coef_loss(temp_params: flax.core.FrozenDict) -> jax.Array:
            power_coef_value = power_coef_state.apply_fn({"params": temp_params})
            power_coef_loss = -power_coef_value * (avg_power-power_target)
            return power_coef_loss, (power_coef_value)

        (power_coef_loss, (power_coef_value)), grads = jax.value_and_grad(power_coef_loss, has_aux=True)(power_coef_state.params)
        power_coef_state = power_coef_state.apply_gradients(grads=grads)

        return power_coef_state, power_coef_loss, power_coef_value

    @classmethod
    def update_source_and_power_coeff(
        cls,
        source_state: TrainState,
        dynamics_state: TrainState,
        power_coef_state: TrainState,
        power_target: float,
        observations: jax.Array,
        key: jax.Array,
    ):

        source_state, (ll_a_inverse, ll_a, effective_dimensionality, avg_power, axes_scales, key) = cls.update_source(
                source_state,
                dynamics_state,
                power_coef_state,
                observations,
                key,
            )

        power_coef_state, power_coef_loss, power_coef_value = cls.update_power_coef(power_coef_state, avg_power, power_target)

        return source_state, power_coef_state, power_coef_loss, power_coef_value, ll_a_inverse, ll_a, effective_dimensionality, avg_power, axes_scales, key

    @staticmethod
    @jax.jit
    def update_critic(
        gamma: float,
        actor_state: TrainState,
        qf_state: RLTrainState,
        ent_coef_state: TrainState,
        dynamics_state: TrainState,
        synergies_state: TrainState,
        posterior: Callable,
        observations: jax.Array,
        actions: jax.Array,
        next_observations: jax.Array,
        rewards: jax.Array,
        dones: jax.Array,
        key: jax.Array,
        next_touch_MPL: jax.Array,
    ):
        key, noise_key1, noise_key2, dropout_key_target, dropout_key_current, actor_key, posterior_key, z_key = jax.random.split(key, 8)
        # sample action from the actor
        # base_dist, explore_dist, A, mu_u_source = actor_state.apply_fn(actor_state.params, next_observations, synergies_state, dynamics_state, actor_key, deterministic=False)
        # next_state_z_actions, _ = base_dist.sample_and_log_prob(seed=noise_key1)
        # next_log_prob = -base_dist.entropy()

        next_state_actions, next_log_prob, _ = BaseJaxPolicy.sample_action(actor_state.params, actor_state, next_observations, actor_key, synergies_state, dynamics_state)
        
        ################### z space Q ###################

        # qf_next_values = qf_state.apply_fn(
        #     qf_state.target_params,
        #     next_observations,
        #     next_state_z_actions,
        #     rngs={"dropout": dropout_key_target},
        # )

        # z_posterior = posterior(observations, next_observations, synergies_state, dynamics_state, posterior_key)
        # z_actions = z_posterior.sample(seed=z_key)
        # # z_posterior.covariance().mean()

        # def mse_loss(params: flax.core.FrozenDict, dropout_key: jax.Array) -> jax.Array:
        #     # shape is (n_critics, batch_size, 1)
        #     current_q_values = qf_state.apply_fn(params, observations, z_actions, rngs={"dropout": dropout_key})
        #     return 0.5 * ((target_q_values - current_q_values) ** 2).mean(axis=1).sum()

        ################### z space Q ###################

        ################### a space Q ###################

        # next_state_actions, next_log_prob = pi_dist.sample_and_log_prob(seed=noise_key1)
        # next_state_actions = dist.sample(seed=noise_key)
        # next_state_actions = nn.sigmoid((mu_u_source + jnp.einsum("...ij,...j->...i", A, next_state_z_actions) + explore_dist.sample(seed=noise_key2))*5.)
        qf_next_values = qf_state.apply_fn(
            qf_state.target_params,
            next_observations,
            next_state_actions,
            rngs={"dropout": dropout_key_target},
        )

        def mse_loss(params: flax.core.FrozenDict, dropout_key: jax.Array) -> jax.Array:
            # shape is (n_critics, batch_size, 1)
            current_q_values = qf_state.apply_fn(params, observations, actions, rngs={"dropout": dropout_key})
            return 0.5 * ((target_q_values - current_q_values) ** 2).mean(axis=1).sum()

        ################### a space Q ###################

        ent_coef_value = ent_coef_state.apply_fn({"params": ent_coef_state.params})

        next_q_values = jnp.min(qf_next_values, axis=0)
        # td error + entropy term
        next_q_values = next_q_values - ent_coef_value * next_log_prob.reshape(-1, 1)
        # shape is (batch_size, 1)
        target_q_values = rewards.reshape(-1, 1) + (1 - dones.reshape(-1, 1)) * gamma * next_q_values

        qf_loss_value, grads = jax.value_and_grad(mse_loss, has_aux=False)(qf_state.params, dropout_key_current)
        qf_state = qf_state.apply_gradients(grads=grads)

        return (
            qf_state,
            (qf_loss_value, ent_coef_value),
            key,
        )

    @staticmethod
    @jax.jit
    def update_actor(
        actor_state: RLTrainState,
        qf_state: RLTrainState,
        ent_coef_state: TrainState,
        dynamics_state: TrainState,
        synergies_state: TrainState,
        observations: jax.Array,
        key: jax.Array,
        touch_MPL: jax.Array,
    ):
        key, dropout_key, noise_key1, noise_key2, dynamics_key = jax.random.split(key, 5)

        def actor_loss(params: flax.core.FrozenDict, synergies_state, dynamics_state, key) -> Tuple[jax.Array, jax.Array]:
            # base_dist, explore_dist, A, mu_u_source = actor_state.apply_fn(params, observations, synergies_state, dynamics_state, key, deterministic=False)
            # actor_z_actions, log_prob = base_dist.sample_and_log_prob(seed=noise_key1)
            # log_prob = -base_dist.entropy()

            actor_actions, log_prob, explore_dist = BaseJaxPolicy.sample_action(params, actor_state, observations, key, synergies_state, dynamics_state)
            
            ################### z space Q ###################

            # qf_pi = qf_state.apply_fn(
            #     qf_state.params,
            #     observations,
            #     actor_z_actions,
            #     rngs={"dropout": dropout_key},
            # )

            ################### z space Q ###################

            ################## a space Q ###################
            
            # actor_actions, log_prob = pi_dist.sample_and_log_prob(seed=noise_key1)
            # actor_actions = dist.sample(seed=noise_key)
            # actor_actions = nn.sigmoid((mu_u_source + jnp.einsum("...ij,...j->...i", A, actor_z_actions) + explore_dist.sample(seed=noise_key2))*5.)
            qf_pi = qf_state.apply_fn(
                qf_state.params,
                observations,
                actor_actions,
                rngs={"dropout": dropout_key},
            )

            ################## a space Q ###################

            log_prob = log_prob.reshape(-1, 1)

            # Take min among all critics (mean for droq)
            min_qf_pi = jnp.min(qf_pi, axis=0)
            ent_coef_value = ent_coef_state.apply_fn({"params": ent_coef_state.params})
            actor_loss = (ent_coef_value * log_prob - min_qf_pi).mean() #- log_prob_source.mean() * .2
            return actor_loss, (-log_prob.mean(), explore_dist.stddev().mean())
            # return actor_loss, (base_dist.entropy().mean(), -log_prob_source.mean(), effective_dimensionality)

        (actor_loss_value, (entropy, epistemic_uncertainty)), grads = jax.value_and_grad(actor_loss, has_aux=True)(actor_state.params, synergies_state, dynamics_state, dynamics_key)
        actor_state = actor_state.apply_gradients(grads=grads)

        return actor_state, qf_state, actor_loss_value, key, entropy, epistemic_uncertainty

    @staticmethod
    @jax.jit
    def soft_update(tau: float, qf_state: RLTrainState) -> RLTrainState:
        qf_state = qf_state.replace(target_params=optax.incremental_update(qf_state.params, qf_state.target_params, tau))
        return qf_state

    @staticmethod
    @jax.jit
    def update_temperature(target_entropy: ArrayLike, ent_coef_state: TrainState, entropy: float):
        def temperature_loss(temp_params: flax.core.FrozenDict) -> jax.Array:
            ent_coef_value = ent_coef_state.apply_fn({"params": temp_params})
            ent_coef_loss = jnp.log(ent_coef_value) * (entropy - target_entropy).mean()  # type: ignore[union-attr]
            return ent_coef_loss, (entropy, ent_coef_value)

        (ent_coef_loss, (entropy, ent_coef_value)), grads = jax.value_and_grad(temperature_loss, has_aux=True)(ent_coef_state.params)
        ent_coef_state = ent_coef_state.apply_gradients(grads=grads)

        return ent_coef_state, ent_coef_loss, entropy, ent_coef_value

    @classmethod
    @partial(jax.jit, static_argnames=["cls"])
    def update_actor_and_temperature(
        cls,
        actor_state: RLTrainState,
        qf_state: RLTrainState,
        ent_coef_state: TrainState,
        dynamics_state: TrainState,
        synergies_state: TrainState,
        observations: jax.Array,
        target_entropy: ArrayLike,
        key: jax.Array,
        batch_touch_MPL: jax.Array,
    ):
        (actor_state, qf_state, actor_loss_value, key, entropy, epistemic_uncertainty) = cls.update_actor(
            actor_state,
            qf_state,
            ent_coef_state,
            dynamics_state,
            synergies_state,
            observations,
            key,
            batch_touch_MPL,
        )
        ent_coef_state, ent_coef_loss_value, entropy, ent_coef_value = cls.update_temperature(target_entropy, ent_coef_state, entropy)
        return actor_state, qf_state, ent_coef_state, actor_loss_value, ent_coef_loss_value, entropy, ent_coef_value, key, epistemic_uncertainty

    @classmethod
    @partial(jax.jit, static_argnames=["cls", "gradient_steps", "policy_delay", "policy_delay_offset", "unscale_action", "unnormalize_obs"])
    def _train(
        cls,
        gamma: float,
        tau: float,
        target_entropy: ArrayLike,
        power_target: float,
        control_variables: List,
        gradient_steps: int,
        data: ReplayBufferSamplesNp,
        policy_delay: int,
        policy_delay_offset: int,
        qf_state: RLTrainState,
        actor_state: TrainState,
        ent_coef_state: TrainState,
        dynamics_state: TrainState,
        source_state: TrainState,
        power_coef_state: TrainState,
        synergies_state: TrainState,
        posterior: Callable,
        key: jax.Array,
        unscale_action: Callable,
        unnormalize_obs: Callable,
    ):
        assert data.observations.shape[0] % gradient_steps == 0
        batch_size = data.observations.shape[0] // gradient_steps

        carry = {
            "actor_state": actor_state,
            "qf_state": qf_state,
            "ent_coef_state": ent_coef_state,
            "dynamics_state": dynamics_state,
            "source_state": source_state,
            "power_coef_state": power_coef_state,
            "synergies_state": synergies_state,
            "key": key,
            "info": {
                "actor_loss": jnp.array(0.0),
                "qf_loss": jnp.array(0.0),
                "ent_coef_loss": jnp.array(0.0),
                "dynamics_loss": jnp.array(0.0),
                "entropy": jnp.array(0.0),
                "ent_coef_value": jnp.array(0.0),
                "p_ratio_source": jnp.array(0.0),
                "entropy_marginal_transition": jnp.array(0.0),
                "epistemic_uncertainty": jnp.array(0.0),
                "mu_power_share": jnp.array(0.0),
            },
        }

        def one_update(i: int, carry: Dict[str, Any]) -> Dict[str, Any]:
            # Note: this method must be defined inline because
            # `fori_loop` expect a signature fn(index, carry) -> carry
            actor_state = carry["actor_state"]
            qf_state = carry["qf_state"]
            ent_coef_state = carry["ent_coef_state"]
            dynamics_state = carry["dynamics_state"]
            source_state = carry["source_state"]
            power_coef_state = carry["power_coef_state"]
            synergies_state = carry["synergies_state"]
            key = carry["key"]
            info = carry["info"]
            batch_obs_all = jax.lax.dynamic_slice_in_dim(data.observations, i * batch_size, batch_size)
            batch_scaled_act = jax.lax.dynamic_slice_in_dim(data.actions, i * batch_size, batch_size)
            batch_act = unscale_action(batch_scaled_act) # sbx SAC scales the actions between (-1,1) before storing them
            batch_next_obs_all = jax.lax.dynamic_slice_in_dim(data.next_observations, i * batch_size, batch_size)
            batch_rew = jax.lax.dynamic_slice_in_dim(data.rewards, i * batch_size, batch_size)
            batch_done = jax.lax.dynamic_slice_in_dim(data.dones, i * batch_size, batch_size)
            batch_touch_MPL = unnormalize_obs(batch_obs_all)[...,-1]
            batch_next_touch_MPL = unnormalize_obs(batch_next_obs_all)[...,-1]
            batch_obs = batch_obs_all
            batch_next_obs = batch_next_obs_all

            dynamics_state, dynamics_loss, key = cls.update_dynamics(dynamics_state,
                batch_obs,
                batch_act,
                batch_next_obs,
                control_variables,
                key
            )

            # synergies_state, entropy_marginal_transition, p_ratio_source, mu_power_share, key = cls.update_synergies(synergies_state,
            #     dynamics_state,
            #     batch_obs,
            #     key
            # )

            entropy_marginal_transition = 0.
            p_ratio_source = 0.
            mu_power_share = 0.

            # source_state, power_coef_state, power_coef_loss, power_coef_value, ll_a_inverse, ll_a, eff_dim_source, avg_power, axes_scales, key = cls.update_source_and_power_coeff(
            #     source_state,
            #     dynamics_state,
            #     power_coef_state,
            #     power_target,
            #     batch_obs,
            #     key,
            # )
            
            (
                qf_state,
                (qf_loss_value, ent_coef_value),
                key,
            ) = cls.update_critic(
                gamma,
                actor_state,
                qf_state,
                ent_coef_state,
                dynamics_state,
                synergies_state,
                posterior,
                batch_obs,
                batch_act,
                batch_next_obs,
                batch_rew,
                batch_done,
                key,
                batch_next_touch_MPL,
            )
            qf_state = cls.soft_update(tau, qf_state)

            (actor_state, qf_state, ent_coef_state, actor_loss_value, ent_coef_loss_value, entropy, ent_coef_value, key, epistemic_uncertainty) = jax.lax.cond(
                (policy_delay_offset + i) % policy_delay == 0,
                # If True:
                cls.update_actor_and_temperature,
                # If False:
                lambda *_: (actor_state, qf_state, ent_coef_state, info["actor_loss"], info["ent_coef_loss"], info["entropy"], info["ent_coef_value"], key, info["epistemic_uncertainty"]),
                actor_state,
                qf_state,
                ent_coef_state,
                dynamics_state,
                synergies_state,
                batch_obs,
                target_entropy,
                key,
                batch_touch_MPL,
            )
            info = {"actor_loss": actor_loss_value, "qf_loss": qf_loss_value, "ent_coef_loss": ent_coef_loss_value,
                    "dynamics_loss": dynamics_loss, "entropy": entropy, "ent_coef_value": ent_coef_value, "p_ratio_source": p_ratio_source,
                    "entropy_marginal_transition": entropy_marginal_transition, "epistemic_uncertainty": epistemic_uncertainty,
                    "mu_power_share": mu_power_share}

            return {
                "actor_state": actor_state,
                "qf_state": qf_state,
                "ent_coef_state": ent_coef_state,
                "dynamics_state": dynamics_state,
                "source_state": source_state,
                "power_coef_state": power_coef_state,
                "synergies_state": synergies_state,
                "key": key,
                "info": info,
            }

        update_carry = jax.lax.fori_loop(0, gradient_steps, one_update, carry)

        return (
            update_carry["qf_state"],
            update_carry["actor_state"],
            update_carry["ent_coef_state"],
            update_carry["dynamics_state"],
            update_carry["source_state"],
            update_carry["power_coef_state"],
            update_carry["synergies_state"],
            update_carry["key"],
            (update_carry["info"]["actor_loss"], update_carry["info"]["qf_loss"], update_carry["info"]["ent_coef_loss"],
             update_carry["info"]["dynamics_loss"], update_carry["info"]["entropy"], update_carry["info"]["ent_coef_value"],
             update_carry["info"]["p_ratio_source"], update_carry["info"]["entropy_marginal_transition"],
             update_carry["info"]["epistemic_uncertainty"], update_carry["info"]["mu_power_share"]),
        )
