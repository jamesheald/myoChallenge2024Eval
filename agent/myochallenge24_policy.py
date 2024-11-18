from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_probability.substrates.jax as tfp
from common_policies import VectorCritic
from flax.training.train_state import TrainState
from gymnasium import spaces

# from common_policies_24 import BaseJaxPolicy
from myochallenge24_common_policies import BaseJaxPolicy
from sbx.common.policies import Flatten  # , VectorCritic#, BaseJaxPolicy
from sbx.common.type_aliases import RLTrainState
from stable_baselines3.common.type_aliases import Schedule

tfd = tfp.distributions

from functools import partial

import distrax
import mujoco
from flax.linen.initializers import constant, normal, zeros_init
from inverse_kinematics.IK import GradientDescentIK
from jax.lax import associative_scan, stop_gradient
from jax.scipy.linalg import cho_factor, cho_solve


class flow_model(nn.Module):
    h_dims_conditioner: int
    num_bijector_params: int
    num_coupling_layers: int
    a_dim: int

    def setup(self):
        # final linear layer of each conditioner initialised to zero so that the flow is initialised to the identity function
        self.conditioners = [
            nn.Sequential(
                [
                    nn.Dense(features=self.h_dims_conditioner),
                    nn.relu,
                    nn.Dense(features=self.h_dims_conditioner),
                    nn.relu,
                    nn.Dense(
                        features=self.num_bijector_params * self.a_dim,
                        bias_init=constant(jnp.log(jnp.exp(1.0) - 1.0)),
                        kernel_init=zeros_init(),
                    ),
                ]
            )
            for layer_i in range(self.num_coupling_layers)
        ]

    def __call__(self):
        def make_flow():
            mask = jnp.arange(self.a_dim) % 2  # every second element is masked
            mask = mask.astype(bool)

            def bijector_fn(params):
                shift, arg_soft_plus = jnp.split(params, 2, axis=-1)
                return distrax.ScalarAffine(
                    shift=shift - jnp.log(jnp.exp(1.0) - 1.0),
                    scale=jax.nn.softplus(arg_soft_plus) + 1e-3,
                )

            layers = []
            for layer_i in range(self.num_coupling_layers):
                layer = distrax.MaskedCoupling(
                    mask=mask,
                    bijector=bijector_fn,
                    conditioner=self.conditioners[layer_i],
                )
                layers.append(layer)
                mask = jnp.logical_not(mask)  # flip mask after each layer

            # return distrax.Inverse(distrax.Chain(layers)) # invert the flow so that the `forward` method is called with `log_prob`
            return distrax.Chain(layers)

        return make_flow()


class dynamics(nn.Module):
    h_dims_dynamics: List
    task_variables: List
    control_variables: List
    action_variables_dynamics: List
    obs_variables: List
    drop_out_rate: float = 0.1

    def setup(self):
        # dynamics_mean = [nn.Sequential([nn.Dense(features=h_dim), nn.relu]) for h_dim in self.h_dims_dynamics]
        # dynamics_mean.append(nn.Dense(features=len(self.control_variables)))
        # self.dynamics_mean = dynamics_mean

        # dynamics_var = [nn.Sequential([nn.Dense(features=h_dim), nn.relu]) for h_dim in self.h_dims_dynamics]
        # # dynamics_var.append(nn.Dense(features=len(self.control_variables))) # anisotropic noise
        # dynamics_var.append(nn.Dense(features=1)) # isotropic noise
        # self.dynamics_var = dynamics_var

        dynamics = [
            nn.Sequential([nn.Dense(features=h_dim), nn.LayerNorm(), nn.relu])
            for h_dim in self.h_dims_dynamics
        ]
        dynamics.append(nn.Dense(features=len(self.control_variables) * 2))
        self.dynamics = dynamics

        self.dropout = nn.Dropout(rate=self.drop_out_rate)

        self.inverse_softplus_sigma = self.param(
            "inverse_softplus_sigma",
            init_fn=lambda key: jnp.full((), jnp.log(jnp.exp(1.0) - 1.0)),
        )

    def __call__(self, obs, u, key, deterministic):
        def get_log_var(x):
            """
            sigma = log(1 + exp(x))
            """

            # sigma = jnp.log(1 + jnp.exp(x)) + 1e-6
            sigma = nn.softplus(x) + 1e-6
            log_var = 2 * jnp.log(sigma)

            return log_var

        ##################### dynamics loss

        # # mask task-specific variables
        # s = s.at[...,self.task_variables].set(0.)

        # x = jnp.concatenate((s, u), axis=-1) # state-dependent dynamics/synergies
        # # x = jnp.copy(u) # state-independent dynamics/synergies
        # # key, subkey = jax.random.split(key)
        # # x = self.dropout(x, False, subkey)
        # for i, fn in enumerate(self.dynamics_mean):
        #     x = fn(x)
        #     # if i == 0:
        #     #     x = self.dropout(x, False, key)
        #     key, subkey = jax.random.split(key)
        #     x = self.dropout(x, False, subkey)
        # s_prime_mean = jnp.copy(x)

        # x = jnp.copy(s)
        # for i, fn in enumerate(self.dynamics_var):
        #     x = fn(x)
        #     key, subkey = jax.random.split(key)
        #     # x = self.dropout(x, False, subkey)
        # s_prime_log_var = get_log_var(x)

        # return s_prime_mean, s_prime_log_var

        # ##################### dynamics loss

        # mask task-specific variables
        obs = obs.at[..., self.task_variables].set(0.0)

        x = jnp.concatenate(
            (obs[..., self.obs_variables], u[..., self.action_variables_dynamics]),
            axis=-1,
        )  # state-dependent dynamics/synergies
        # x = jnp.copy(u) # state-independent dynamics/synergies
        for i, fn in enumerate(self.dynamics):
            x = fn(x)
            key, subkey = jax.random.split(key)
            if i == 0:
                x = self.dropout(x, deterministic, key)
        s_prime_mean, s_prime_scale = jnp.split(x, 2, axis=-1)
        s_prime_log_var = get_log_var(s_prime_scale)

        return s_prime_mean, s_prime_log_var


# class Posterior:

#     def __init__(self, control_variables):

#         self.control_variables = control_variables
#         self.z_dim = len(control_variables)

#     def infer(self, obs, next_obs, synergies_state, dynamics_state, key):

#         def symmetrize(A):
#             return 0.5 * (A + jnp.swapaxes(A, -1, -2))

#         def psd_solve(A, b, diagonal_boost=1e-9):
#             A = symmetrize(A) + diagonal_boost * jnp.eye(A.shape[-1])
#             L, lower = cho_factor(A, lower=True)
#             x = cho_solve((L, lower), b)
#             return x

#         def get_posterior(residual_power, Jac, A, E_y_prime, s_prime_var, next_obs):

#             # prior p(z|x) = N(mu_z, Lambda^-1) - the optimal source
#             mu_z = jnp.zeros(self.z_dim)
#             Lambda = 1. / (residual_power / self.z_dim) * jnp.eye(self.z_dim) # equal power allocation assumed here

#             # likelihood p(y'|z,x) = N(A_eff z + b, L^-1)
#             A_eff = Jac @ A # effective communication channel (synergies, A, followed by Jacobian, Jac)
#             b = E_y_prime
#             L = jnp.diag(1 / s_prime_var)

#             # posterior p(z|y',x) = N(mu_posterior, Sigma_posterior)
#             Sigma_posterior = psd_solve(Lambda + A_eff.T @ L @ A_eff, np.eye(self.z_dim)) # jnp.linalg.inv(Lambda + A_eff.T @ L @ A_eff)
#             mu_posterior = Sigma_posterior @ (A_eff.T @ L @ (next_obs[...,self.control_variables] - b) + Lambda @ mu_z)

#             return mu_posterior, Sigma_posterior

#         batch_get_posterior = jax.vmap(get_posterior, in_axes=(0,0,0,0,0,0))

#         _, _, _, _, A, _, _, _, residual_power, Jac, E_y_prime, s_prime_var = synergies_state.apply_fn(synergies_state.params, obs, dynamics_state, key)

#         mu_posterior, Sigma_posterior = batch_get_posterior(residual_power, Jac, A, E_y_prime, s_prime_var, next_obs)

#         z_posterior = distrax.MultivariateNormalFullCovariance(loc=mu_posterior, covariance_matrix=Sigma_posterior)

#         return z_posterior


def posterior_inference(
    obs, next_obs, synergies_state, dynamics_state, key, control_variables
):
    def symmetrize(A):
        return 0.5 * (A + jnp.swapaxes(A, -1, -2))

    def psd_solve(A, b, diagonal_boost=1e-9):
        A = symmetrize(A) + diagonal_boost * jnp.eye(A.shape[-1])
        L, lower = cho_factor(A, lower=True)
        x = cho_solve((L, lower), b)
        return x

    def get_posterior(
        residual_power,
        Jac,
        A,
        mu_u_source,
        E_y_prime,
        s_prime_var,
        next_obs,
        control_variables,
    ):
        z_dim = len(control_variables)

        # prior p(z|x) = N(mu_z, Lambda^-1) - the optimal source
        mu_z = jnp.zeros(z_dim)
        Lambda = (
            1.0 / (residual_power / z_dim) * jnp.eye(z_dim)
        )  # equal power allocation assumed here

        # likelihood p(y'|z,x) = N(Jac eps + b, L^-1) = N(Jac (A z + mu_u) + b, L^-1) = N(Jac A z + Jac mu_u + b, L^-1)
        # (A z + mu_u)
        A_eff = (
            Jac @ A
        )  # effective communication channel (synergies, A, followed by Jacobian, Jac)
        b_eff = Jac @ mu_u_source + E_y_prime
        L = jnp.diag(1 / s_prime_var)  # isotropic noise assumed here

        # posterior p(z|y',x) = N(mu_posterior, Sigma_posterior)
        Sigma_posterior = psd_solve(
            Lambda + A_eff.T @ L @ A_eff, np.eye(z_dim)
        )  # jnp.linalg.inv(Lambda + A_eff.T @ L @ A_eff)
        mu_posterior = Sigma_posterior @ (
            A_eff.T @ L @ (next_obs[..., control_variables] - b_eff) + Lambda @ mu_z
        )

        return mu_posterior, Sigma_posterior

    batch_get_posterior = jax.vmap(get_posterior, in_axes=(0, 0, 0, 0, 0, 0, 0, None))

    _, _, _, mu_u_source, A, _, _, _, residual_power, Jac, E_y_prime, s_prime_var = (
        synergies_state.apply_fn(synergies_state.params, obs, dynamics_state, key)
    )

    mu_posterior, Sigma_posterior = batch_get_posterior(
        residual_power,
        Jac,
        A,
        mu_u_source,
        E_y_prime,
        s_prime_var,
        next_obs,
        control_variables,
    )

    z_posterior = distrax.MultivariateNormalFullCovariance(
        loc=mu_posterior, covariance_matrix=Sigma_posterior
    )

    return z_posterior


class synergies(nn.Module):
    h_dims_source: List
    action_dim: int
    z_dim: int
    source_mu: List
    power_target_per_dim: float
    num_MC_samples: int = 1
    cocontraction: str = None

    def setup(self):
        # self.power_target = self.power_target_per_dim * self.action_dim
        self.power_target = self.power_target_per_dim * self.z_dim

        source = [
            nn.Sequential([nn.Dense(features=h_dim), nn.relu])
            for h_dim in self.h_dims_source
        ]
        source.append(nn.Dense(features=self.action_dim + 1))
        self.source = source

        self.standard_basis = np.eye(self.z_dim, self.action_dim)

    def __call__(
        self, obs, dynamics_state, key, deterministic, cocontraction, squash_controls_fn
    ):
        def modified_gram_schmidt(vectors, init_i=0):
            """
            adapted from https://github.com/tensorflow/probability/blob/main/tensorflow_probability/python/math/gram_schmidt.py
            fundamental change: while_loop replaced with scan
            Args:
            vectors: A Tensor of shape `[d, n]` of `d`-dim column vectors to
              orthonormalize.

            Returns:
            A Tensor of shape `[d, n]` corresponding to the orthonormalization.
            """

            def orthogonalise_wrt_subspace(subspace_vecs, vec):
                weights = vec @ subspace_vecs
                vec = vec - subspace_vecs @ weights
                return vec

            batch_orthogonalise_wrt_subspace = jax.vmap(
                orthogonalise_wrt_subspace, in_axes=(None, 1), out_axes=(1)
            )

            def body_fn(vecs, i):
                u = jnp.nan_to_num(vecs[:, i] / jnp.linalg.norm(vecs[:, i]))
                weights = u @ vecs
                masked_weights = jnp.where(jnp.arange(num_vectors) > i, weights, 0.0)
                vecs = vecs - jnp.outer(u, masked_weights)
                return vecs, None

            num_vectors = vectors.shape[-1]
            if init_i != 0:
                null_space_basis = batch_orthogonalise_wrt_subspace(
                    vectors[:, :init_i], vectors[:, init_i:]
                )  # better to batch than scan where possible (synergies already orthonormalised)
                vectors = jnp.concatenate(
                    (vectors[:, :init_i], null_space_basis), axis=1
                )
            vectors, _ = jax.lax.scan(
                body_fn, vectors, jnp.arange(init_i, num_vectors - 1)
            )
            vec_norm = jnp.linalg.norm(vectors, axis=0, keepdims=True)
            return jnp.nan_to_num(vectors / vec_norm)

        def get_participation_ratio(Sigma):
            eig_vals, _ = jnp.linalg.eigh(Sigma)
            eig_vals = jnp.real(eig_vals)
            eff_dim = eig_vals.sum() ** 2 / (eig_vals**2).sum()

            return eff_dim

        def dynamics_forward_pass(
            u, dynamics_state, obs, deterministic, key, squash_controls_fn
        ):
            # squash the control inputs before passing them through the dynamics model
            # if len(self.source_mu) == 1:
            #     u = nn.sigmoid(u*5.)
            # else:
            #     u = jnp.concatenate([nn.sigmoid(u[...,:63]*5.),nn.tanh(u[...,63:]*3.)],axis=-1)
            u = squash_controls_fn(
                jnp.concatenate((u, jnp.zeros(17)), axis=-1)
            )  #  add dummy, it gets ignored in dynamics anyway

            y_prime_mean, s_prime_log_var = dynamics_state.apply_fn(
                dynamics_state.params, obs, u, key, deterministic
            )

            return y_prime_mean, (y_prime_mean, jnp.exp(s_prime_log_var))

        batch_get_jacobian = jax.vmap(
            jax.jacrev(dynamics_forward_pass, has_aux=True),
            in_axes=(None, None, None, None, 0, None),
        )

        def half_log_det(Sigma, diagonal_boost=1e-9):
            L = jnp.linalg.cholesky(Sigma + diagonal_boost * jnp.eye(Sigma.shape[-1]))
            half_log_det_Sigma = jnp.log(jnp.diagonal(L, axis1=-2, axis2=-1)).sum(-1)

            return half_log_det_Sigma

        def get_orthogonal_bases(A, key):
            # augment A with random vectors and use the QR decomposition to orthogonalise these vectors relative to the columns of A and each other
            A_augmented = jnp.concatenate(
                (A, jax.random.normal(key, (63, 63 - self.z_dim))), axis=1
            )

            # Q, _ = jnp.linalg.qr(A_augmented)
            # # A_perp = jnp.concatenate((A, Q[:,self.z_dim:]), axis=1)
            # A_perp = Q[:,self.z_dim:]

            A_perp = modified_gram_schmidt(A_augmented, init_i=self.z_dim)[
                :, self.z_dim :
            ]

            return A_perp

        def get_synergies(
            mu_u_source,
            obs,
            dynamics_state,
            residual_power,
            key,
            deterministic,
            squash_controls_fn,
        ):
            key, subkey = jax.random.split(key)

            # mu_u_source = jnp.zeros(self.action_dim)

            subkeys = jax.random.split(key, self.num_MC_samples)

            (Jac, (y_prime_mean, s_prime_var)) = batch_get_jacobian(
                mu_u_source[..., :63],
                dynamics_state,
                obs,
                deterministic,
                subkeys,
                squash_controls_fn,
            )

            # Jac = jax.random.normal(key, (self.num_MC_samples,self.z_dim,self.action_dim))
            # y_prime_mean = jax.random.normal(key, (self.num_MC_samples,self.z_dim))
            # s_prime_var = jax.random.normal(key, (self.num_MC_samples,self.z_dim))

            mean_Jac = Jac.mean(axis=0)

            # U, S, Vh = jnp.linalg.svd(mean_Jac, full_matrices=True) # sloooow
            # A = Vh[:self.z_dim,:].T@U[:,:self.z_dim].T
            # A_complement = Vh[self.z_dim:,:].T # orthogonal complement

            # A,_ = jax.scipy.linalg.polar(mean_Jac.T, method="qdwh") # default method=qdwh fast!, method="svd" sloooow

            # economic SVD of Jac (full_matrices=False) automatically excludes the right singular vectors that are associated with zero singular values
            # U, S, Vh = jnp.linalg.svd(mean_Jac, full_matrices=False)

            # right singular vectors as basis
            # Jac_z = mean_Jac @ Vh.T
            # Vh = Vh * jnp.where(jnp.diag(Jac_z) < 0, -1, 1)[:,None] # as expected, seems very useful
            # A = Vh.T

            # rows of jacobian as basis
            # A = (mean_Jac/jnp.linalg.norm(mean_Jac,axis=-1, keepdims=True)).T

            # rows of jacobian orthonormalised via gram schmidt as basis
            A = modified_gram_schmidt(mean_Jac.T)
            # A = gram_schmidt(mean_Jac.T)
            # A = mean_Jac.T

            # nearest orthogonal matrix to jacobian as basis
            # https://en.wikipedia.org/wiki/Singular_value_decomposition#Nearest_orthogonal_matrix
            # https://math.stackexchange.com/questions/4492668
            # A = Vh.T@U.T

            # eigenvectors of unique orthogonal projector as basis
            # https://en.wikipedia.org/wiki/Projection_(linear_algebra)#Formulas
            # P = mean_Jac.T@jnp.linalg.solve(mean_Jac@mean_Jac.T,mean_Jac) # unique orthogonal projector matrix
            # P = Vh.T@Vh
            # eig_vals, eig_vecs = jnp.linalg.eigh(P)
            # A = eig_vecs[:,-self.z_dim:] # use the eigenvectors of P that have nonzero eigenvalues as the orthonormal basis

            # standard basis vectors projected onto unique orthogonal projector as basis
            # https://en.wikipedia.org/wiki/Projection_(linear_algebra)#Formulas
            # P = mean_Jac.T@jnp.linalg.solve(mean_Jac@mean_Jac.T,mean_Jac) # unique orthogonal projector matrix
            # P = Vh.T@Vh # unique orthogonal projector matrix
            # q = jax.vmap(lambda P, e: P @ e, in_axes=(None,0), out_axes=1)(P,self.standard_basis)
            # A = gram_schmidt(q)
            # U, S, Vh = jnp.linalg.svd(q, full_matrices=False)
            # nearest orthogonal matrix to q as basis
            # https://en.wikipedia.org/wiki/Singular_value_decomposition#Nearest_orthogonal_matrix
            # A = Vh.T@U.T

            if self.num_MC_samples > 1:
                eps = 0.0
                explore_var = (Jac - Jac.mean(axis=0, keepdims=True)).var() + eps
                key, subkey = jax.random.split(key)
                A_complement = get_orthogonal_bases(A, key)
            elif self.num_MC_samples == 1:
                explore_var = 0.0
                A_complement = jnp.ones((63, 63 - self.z_dim))  # not actually used
            # if self.num_MC_samples == 1:
            #     eps = 1e-12
            #     explore_cov = eps*np.eye(self.action_dim)
            # elif self.num_MC_samples > 1:
            #     eps = 1e-6
            #     explore_cov = (eps + self.explore_coeff*(Jac-Jac.mean(axis=0, keepdims=True)).var())*np.eye(self.action_dim)
            # explore_cov = (jax.vmap(lambda J: J.T @ J)(Jac).mean(axis=0) - mean_Jac.T@mean_Jac)/self.z_dim + eps*np.eye(self.action_dim)
            # explore_cov = jax.vmap(lambda J: J.T @ J / J.shape[0], in_axes=(1,))(Jac).mean(axis=0) - jnp.outer(Jac.mean(axis=(0,1)),Jac.mean(axis=(0,1))) + eps*np.eye(self.action_dim)
            # eigenvalues, eigenvectors = jnp.linalg.eig(explore_cov)
            # A_explore = eigenvectors @ jnp.diag(jnp.sqrt(eigenvalues))

            # Sigma_u_source = self.power_target * A @ jnp.eye(self.z_dim) @ A.T # equal power allocation assumed here
            Sigma_u_source = (
                residual_power / self.z_dim * A @ jnp.eye(self.z_dim) @ A.T
            )  # equal power allocation assumed here

            # p_ratio_source = get_participation_ratio(Sigma_u_source)

            # y_dim = self.z_dim
            # # Sigma_y_prime = Jac @ Sigma_u_source @ Jac.T + jnp.diag(s_prime_var)
            # # Sigma_y_prime = mean_Jac @ Sigma_u_source @ mean_Jac.T + jnp.diag(s_prime_var)
            # # Sigma_y_prime = self.power_target * jnp.diag(S) @ jnp.eye(self.z_dim) @ jnp.diag(S).T + jnp.diag(s_prime_var) # nan for reasons I dont understand with # Sigma_y_prime = Jac @ Sigma_u_source @ Jac.T + jnp.diag(s_prime_var)
            # Sigma_y_prime = residual_power / self.z_dim * jnp.diag(S) @ jnp.eye(self.z_dim) @ jnp.diag(S).T + jnp.diag(s_prime_var.mean(axis=0)) # nan for reasons I dont understand with # Sigma_y_prime = Jac @ Sigma_u_source @ Jac.T + jnp.diag(s_prime_var)
            # entropy_marginal_transition = half_log_det(Sigma_y_prime) - half_log_det(jnp.diag(s_prime_var.mean(axis=0))) # + 0.5 * y_dim * jnp.log(2*jnp.pi*jnp.exp(1.))

            # A_scaled = A * jnp.sqrt(residual_power / self.z_dim) # equal power allocation assumed here
            # A_scaled = A @ jnp.sqrt(residual_power[None,:] / self.z_dim) # equal power allocation assumed here

            return (
                A,
                explore_var,
                A_complement,
                0.0,
                0.0,
                Jac,
                y_prime_mean,
                s_prime_var,
            )

        batch_get_synergies = jax.vmap(
            get_synergies, in_axes=(0, 0, None, 0, 0, None, None)
        )

        # x = jnp.copy(obs)
        # for fn in self.source:
        #     x = fn(x)
        # mu_u_source = jnp.copy(x)

        # x = jnp.copy(obs)
        # for fn in self.source:
        #     x = fn(x)
        # mu_u_source_dir, mu_u_source_mag = jnp.split(x, [self.action_dim], axis=-1)

        # # norm(mu)**2 + trace(Sigma) <= power_target = power_target_per_dim * action_dim

        # mu_u_source_unit_norm = mu_u_source_dir/jnp.linalg.norm(mu_u_source_dir, axis=-1)[:,None]
        # mu_norm = jnp.sqrt(self.power_target)*nn.sigmoid(mu_u_source_mag) # much better in practice than jnp.sqrt(self.power_target*nn.sigmoid(mu_u_source_mag)), and mu_power_share = nn.sigmoid(mu_u_source_mag)
        # mu_u_source = mu_norm * mu_u_source_unit_norm # mu_u_source has a squared norm of mu_norm**2
        # residual_power = self.power_target-mu_norm**2

        # mu_u_source = jnp.zeros((obs.shape[0], self.action_dim))
        if self.cocontraction is not None:
            mu_u_source = (
                jnp.ones((obs.shape[0], self.action_dim))
                * nn.sigmoid(cocontraction)
                * -1.0
            )
        else:
            if len(self.source_mu) == 1:
                mu_u_source = (
                    jnp.ones((obs.shape[0], self.action_dim)) * self.source_mu[0]
                )  # -0.5 and -0.4 decent
            else:
                mu_u_source = jnp.array(self.source_mu)[None, :].repeat(
                    obs.shape[0], axis=0
                )
        mu_u_source_mag = jnp.zeros((obs.shape[0], 1))
        residual_power = self.power_target * jnp.ones((obs.shape[0], 1))

        keys = jax.random.split(key, obs.shape[0])
        (
            A,
            explore_var,
            A_complement,
            p_ratio_source,
            entropy_marginal_transition,
            Jac,
            y_prime_mean,
            s_prime_var,
        ) = batch_get_synergies(
            mu_u_source,
            obs,
            dynamics_state,
            residual_power,
            keys,
            deterministic,
            squash_controls_fn,
        )

        mu_power_share = nn.sigmoid(mu_u_source_mag) ** 2

        return (
            entropy_marginal_transition,
            p_ratio_source,
            mu_u_source,
            A,
            explore_var,
            A_complement,
            mu_power_share,
            residual_power,
            Jac,
            y_prime_mean,
            s_prime_var,
        )


class power_coef(nn.Module):
    power_init: float = 0.0
    constraint_type: str = "equality"

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        if self.constraint_type == "equality":  # power_coef can be positive or negative
            power_coef = self.param(
                "power_coef", init_fn=lambda key: jnp.full((), self.power_init)
            )
        elif self.constraint_type == "inequality":  # power_coef is non-negative
            inverse_softplus_power_coef = self.param(
                "inverse_softplus_power_coef",
                init_fn=lambda key: jnp.full(
                    (), jnp.log(jnp.exp(self.power_init) - 1.0)
                ),
            )
            power_coef = nn.softplus(inverse_softplus_power_coef)
        return power_coef


class source(nn.Module):
    h_dims_source: List
    h_dims_dynamics: List
    h_dims_inverse: List
    h_dims_conditioner: int
    num_coupling_layers: int
    task_variables: List
    control_variables: List
    a_dim: int
    push_forward_type: str
    state_dependent_base_variance: bool = False
    init_axes_scales: float = -3.0
    power_coefficient: float = 1.0
    init_power_coeff: float = 0.01
    power_target: float = 1e-4
    drop_out_rate: float = 0.1
    log_std_init: float = 0.0
    power_for_var: bool = True
    stop_grad_posterior_bijector: bool = False
    normalise_scales: bool = True

    def setup(self):
        num_control_variables = len(self.control_variables)

        dynamics = [
            nn.Sequential([nn.Dense(features=h_dim), nn.relu])
            for h_dim in self.h_dims_dynamics
        ]
        dynamics.append(nn.Dense(features=num_control_variables * 2))
        # dynamics.append(nn.Dense(features=num_control_variables))
        self.dynamics = dynamics

        self.dropout = nn.Dropout(rate=self.drop_out_rate)

        source_policy = [
            nn.Sequential([nn.Dense(features=h_dim), nn.relu])
            for h_dim in self.h_dims_source
        ]  # [nn.Dense(features=h_dim), nn.LayerNorm(), nn.relu]
        if self.state_dependent_base_variance:
            # initialise the final layer output to 0 so that the inititial mean and log_var is 0
            source_policy.append(
                nn.Dense(
                    features=self.a_dim + 1,
                    bias_init=zeros_init(),
                    kernel_init=zeros_init(),
                )
            )
        else:
            # initialise the final layer output to 0 so that the inititial mean and log_var is 0
            source_policy.append(
                nn.Dense(
                    features=self.a_dim,
                    bias_init=zeros_init(),
                    kernel_init=zeros_init(),
                )
            )
        self.source_policy = source_policy

        self.house_holder = self.param(
            "house_holder", normal(), (self.a_dim, self.a_dim)
        )
        if self.power_for_var:
            # self.decoder_axes_scales = self.param('decoder_axes_scales', constant(0.), (self.a_dim)) # softmax
            self.decoder_axes_scales = self.param(
                "decoder_axes_scales",
                constant(jnp.log(jnp.exp(1.0) - 1.0)),
                (self.a_dim),
            )  # softplus
        else:
            self.decoder_axes_scales = self.param(
                "decoder_axes_scales", constant(self.init_axes_scales), (self.a_dim)
            )  # for elbow, worse with -3.5 or -2.5, when used with power dual or power fixed 0.05
        self.decoder_b = self.param("decoder_b", zeros_init(), (self.a_dim))

        self.flow = flow_model(
            h_dims_conditioner=self.h_dims_conditioner,
            num_bijector_params=2,
            num_coupling_layers=self.num_coupling_layers,
            a_dim=self.a_dim,
        )

        inverse = [
            nn.Sequential([nn.Dense(features=h_dim), nn.relu])
            for h_dim in self.h_dims_inverse
        ]
        inverse.append(nn.Dense(features=self.a_dim * 2))
        self.inverse = inverse

        # self.power_coef = self.param('power_coef', constant(jnp.log(jnp.exp(self.init_power_coeff)-1.)), (1,))

    def __call__(self, s, key, dynamics_params, power_coef):
        def log_likelihood_diagonal_Gaussian(x, mu, log_var):
            """
            Calculate the log likelihood of x under a diagonal Gaussian distribution
            var_min is added to the variances for numerical stability
            """
            log_var = stabilise_variance(log_var)

            # calculate the log likelihood
            log_likelihood = -0.5 * (
                log_var + jnp.log(2 * jnp.pi) + (x - mu) ** 2 / jnp.exp(log_var)
            )

            return log_likelihood

        batch_log_likelihood_diagonal_Gaussian = jax.vmap(
            log_likelihood_diagonal_Gaussian, in_axes=(None, 0, 0)
        )

        def stabilise_variance(log_var, var_min=1e-6):
            """
            var_min is added to the variances for numerical stability
            """
            return jnp.log(jnp.exp(log_var) + var_min)

        def sample_diag_Gaussian(mean, log_var, key):
            """
            sample from a diagonal Gaussian distribution
            """
            log_var = stabilise_variance(log_var)

            return mean + jnp.exp(0.5 * log_var) * jax.random.normal(key, mean.shape)

        def get_log_var(x):
            """
            sigma = log(1 + exp(x))
            """

            sigma = jnp.log(1 + jnp.exp(x)) + 1e-4
            log_var = 2 * jnp.log(sigma)

            return log_var

        def symmetrise(A, diagonal_boost=1e-9):
            return 0.5 * (A + A.T) + diagonal_boost * jnp.eye(A.shape[-1])

        @jax.vmap
        def process_house_holder(x):
            x /= jnp.linalg.norm(x)
            x = jnp.eye(x.size) - 2 * jnp.outer(x, x)

            return x

        def construct_channel(epsilon=0.0):
            orthogonal_matrix = associative_scan(
                jax.vmap(jnp.matmul), process_house_holder(self.house_holder)
            )[-1]

            # lower bound the axes scales
            # if decoder_axes_scales become very small, the entropy may go below the target_entropy
            # this can causes problems for SAC, as it doesn't optimise these variables
            # decoder_axes_scales = jnp.clip(nn.softplus(self.decoder_axes_scales),1e-2,None)
            decoder_axes_scales = nn.softplus(self.decoder_axes_scales)

            if self.normalise_scales:
                # 4 options:
                # softplus with decoder_axes_scales**2
                # softplus with decoder_axes_scales
                # no softplus with decoder_axes_scales**2
                # softmax (initialised eg at 0)

                # # enforce a floor of epislon on the normalised scales
                # normalised_scales = decoder_axes_scales**2 / jnp.sum(decoder_axes_scales**2)
                # transformed_scales = normalised_scales * (1-epsilon*self.a_dim) + epsilon
                # scales = jnp.sqrt(self.a_dim * transformed_scales)

                decoder_axes_scales = jnp.sqrt(
                    self.a_dim
                    * decoder_axes_scales**2
                    / jnp.sum(decoder_axes_scales**2)
                )

                # decoder_axes_scales = jnp.sqrt(self.a_dim * nn.softmax(self.decoder_axes_scales))

                decoder_axes_scales = jnp.clip(decoder_axes_scales, epsilon, None)

                # assert jnp.all(scales == scales2), "scales does not equal scales2 source"

                # scales = jnp.sqrt(self.power_target * self.a_dim * decoder_axes_scales**2 / jnp.sum(decoder_axes_scales**2))

            # the columns of orthogonal_matrix are my axes, described in the coordinates of the standard basis
            # the individual elements of z_u are coordinates along those axes
            A = (
                orthogonal_matrix @ jnp.diag(decoder_axes_scales)
            )  # scale the columns, which scales the z-coordinates (https://www.youtube.com/watch?v=P2LTAUO1TdA)
            # A = self.house_holder

            # # testing role of orthogonal_matrix!!!!!
            # A = jnp.diag(decoder_axes_scales)

            # # testing role of orthogonal_matrix!!!!!
            # A = orthogonal_matrix @ jnp.eye(self.a_dim)

            return A, decoder_axes_scales

        def get_effective_dimensionality(Sigma):
            eig_vals, _ = jnp.linalg.eigh(Sigma)
            eig_vals = jnp.real(eig_vals)
            eff_dim = eig_vals.sum() ** 2 / (eig_vals**2).sum()

            return eff_dim

        def get_pushforward_dist(A, b, x_mean, x_log_var):
            push_mean = A @ x_mean + b
            push_cov = A @ jnp.diag(jnp.exp(x_log_var)) @ A.T

            return push_mean, push_cov

        def sample_action_and_predict_next_state(
            s, a_mean, a_log_var, decoder_A, decoder_b, a_cov, key, dynamics_params
        ):
            a_key, s_key, dropout_key = jax.random.split(key, 3)

            # push_mean, push_cov = get_pushforward_dist(decoder_A, decoder_b, a_mean, a_log_var)
            # base_dist_a = distrax.MultivariateNormalFullCovariance(push_mean, push_cov)
            # dist_a = distrax.Transformed(base_dist_a, distrax.Block(distrax.Tanh(), 1))
            # squashed_a_i, ll_a = dist_a.sample_and_log_prob(seed=a_key)

            base_dist_a = distrax.MultivariateNormalDiag(
                loc=a_mean, scale_diag=jnp.exp(0.5 * a_log_var)
            )
            if self.push_forward_type == "linear":
                bijector = distrax.UnconstrainedAffine(matrix=decoder_A, bias=decoder_b)
            elif self.push_forward_type == "nonlinear":
                bijector = self.flow()

            if self.power_for_var:
                uniform_scaler = distrax.UnconstrainedAffine(
                    matrix=jnp.sqrt(self.power_target) * jnp.eye(self.a_dim),
                    bias=jnp.zeros(self.a_dim),
                )
                bijector = distrax.Chain([bijector, uniform_scaler])

            dist_a = distrax.Transformed(
                base_dist_a, distrax.Chain([distrax.Block(distrax.Tanh(), 1), bijector])
            )
            squashed_a_i, ll_a = dist_a.sample_and_log_prob(seed=a_key)

            x = jnp.concatenate((s, squashed_a_i), axis=-1)
            # x = jnp.copy(squashed_a_i)
            for i, fn in enumerate(self.dynamics):
                x = fn.apply(
                    {"params": dynamics_params["params"]["dynamics_" + str(i)]}, x
                )
                if i == 0:
                    x = self.dropout(x, False, dropout_key)
            # delta_s_mean = x
            # delta_s_scale = 1.
            delta_s_mean, delta_s_scale = jnp.split(x, 2, axis=-1)
            delta_s_log_var = get_log_var(delta_s_scale)
            delta_s_i = sample_diag_Gaussian(delta_s_mean, delta_s_log_var, s_key)
            s_prime_i = s[:, self.control_variables] + delta_s_i

            x = jnp.concatenate((s, s_prime_i), axis=-1)  # s_prime not delta
            for fn in self.inverse:
                x = fn(x)
            a_inverse_mean, a_inverse_scale = jnp.split(x, 2, axis=-1)
            a_inverse_log_var = get_log_var(a_inverse_scale)
            # if self.power_for_var:
            #     a_inverse_log_var = jnp.clip(a_inverse_log_var, None, jnp.log(self.power_target))
            # else:
            #     a_inverse_log_var = jnp.clip(a_inverse_log_var, None, 0) # for elbow, clipping seems to have negligible impact
            a_inverse_log_var = jnp.clip(
                a_inverse_log_var, None, 0
            )  # for elbow, clipping seems to have negligible impact
            # ll_a_inverse = log_likelihood_diagonal_Gaussian(a_i, a_inverse_mean, a_inverse_log_var).sum()

            # push_mean, push_cov = get_pushforward_dist(decoder_A, decoder_b, a_inverse_mean, a_inverse_log_var)
            # base_dist_a_inverse = distrax.MultivariateNormalFullCovariance(push_mean, push_cov)
            # dist_a_inverse = distrax.Transformed(base_dist_a_inverse, distrax.Block(distrax.Tanh(), 1))
            # ll_a_inverse = dist_a_inverse.log_prob(jnp.clip(squashed_a_i, -1.+1e-6, 1.-1e-6))

            base_dist_a_inverse = distrax.MultivariateNormalDiag(
                loc=a_inverse_mean, scale_diag=jnp.exp(0.5 * a_inverse_log_var)
            )

            if self.stop_grad_posterior_bijector:
                posterior_bijector = stop_gradient(bijector)
            else:
                posterior_bijector = bijector

            dist_a_inverse = distrax.Transformed(
                base_dist_a_inverse,
                distrax.Chain([distrax.Block(distrax.Tanh(), 1), posterior_bijector]),
            )
            ll_a_inverse = dist_a_inverse.log_prob(
                jnp.clip(squashed_a_i, -1.0 + 1e-6, 1.0 - 1e-6)
            )

            return squashed_a_i, s_prime_i, ll_a_inverse, ll_a

        # mask task-specific variables (e.g. pos_error)
        s = s.at[:, self.task_variables].set(0.0)

        x = jnp.copy(s)
        for fn in self.source_policy:
            x = fn(x)
        if self.state_dependent_base_variance:
            a_mean, a_scale = jnp.split(x, [self.a_dim], axis=-1)
            a_log_var = jnp.ones(a_mean.shape) * get_log_var(a_scale)
        else:
            a_mean = jnp.copy(x) * 0.0
            # if self.power_for_var:
            #     a_log_var = jnp.ones(self.a_dim) * jnp.log(self.power_target)
            # else:
            #     a_log_var = jnp.zeros(self.a_dim)
            a_log_var = jnp.zeros(self.a_dim)

        decoder_A, decoder_axes_scales = construct_channel()

        if self.state_dependent_base_variance:
            if self.power_for_var:
                effective_dimensionality = (
                    (
                        jnp.exp(get_log_var(a_scale))
                        * decoder_axes_scales**2
                        * self.power_target
                    ).sum(-1)
                    ** 2
                    / (
                        (
                            jnp.exp(get_log_var(a_scale))
                            * decoder_axes_scales**2
                            * self.power_target
                        )
                        ** 2
                    ).sum(-1)
                ).mean()
            else:
                effective_dimensionality = (
                    (jnp.exp(get_log_var(a_scale)) * decoder_axes_scales**2).sum(-1)
                    ** 2
                    / (
                        (jnp.exp(get_log_var(a_scale)) * decoder_axes_scales**2) ** 2
                    ).sum(-1)
                ).mean()
        else:
            a_cov = decoder_A @ jnp.diag(jnp.exp(a_log_var)) @ decoder_A.T
            effective_dimensionality = get_effective_dimensionality(a_cov)

        squashed_a_i, s_prime_i, ll_a_inverse, ll_a = (
            sample_action_and_predict_next_state(
                s,
                a_mean,
                a_log_var,
                decoder_A,
                self.decoder_b * 0.0,
                [],
                key,
                dynamics_params,
            )
        )  # for elbow, decoder_b = 0 is (marginally) worse

        avg_power = (squashed_a_i**2).mean(axis=-1)

        mutual_information = ll_a_inverse - ll_a
        mutual_information /= self.a_dim

        power_penalty = power_coef * (
            avg_power - self.power_target
        )  # KKT stationarity condition

        # power_penalty = stop_gradient(nn.softplus(self.power_coef)) * avg_power

        # i don't think either of below matter, because gradient of expectation = expectation of grad

        # average power < P, average over dimensions of u, allowing u to be greater than P in some dims of u (this is what i want, synergies)
        # power_penalty = stop_gradient(nn.softplus(self.power_coef)) * (avg_power-self.power_target) # KKT stationarity condition
        # power_coef_loss = -nn.softplus(self.power_coef) * stop_gradient(avg_power-self.power_target) # KKT complementary slackness condition

        # average power < P, average over dimensions of u AND states, allowing u to be greater than P in some dims of u and/or states
        # power_penalty = stop_gradient(nn.softplus(self.power_coef)) * (avg_power.mean()-self.power_target) # KKT stationarity condition
        # power_coef_loss = -nn.softplus(self.power_coef) * stop_gradient(avg_power.mean()-self.power_target) # KKT complementary slackness condition

        # # L2 penalty on power
        # power_penalty = self.power_coefficient * avg_power # for elbow, with axes scale init softplus(-3), decent if fixed at 0.005, 0.01, 0.05 (best at 0.05), worse at 0.1 or 0.001
        # power_coef_loss = 0.

        if self.normalise_scales:
            loss = -mutual_information

        else:
            loss = (
                -mutual_information + power_penalty
            )  # + self.power_coefficient * (avg_power-self.power_target)**2 # + power_coef_loss

        return (
            loss,
            ll_a_inverse,
            ll_a,
            effective_dimensionality,
            avg_power,
            decoder_axes_scales,
        )


class Actor(nn.Module):
    net_arch: Sequence[int]
    action_dim: int
    z_dim: int
    push_forward_type: str
    h_dims_conditioner: int
    num_coupling_layers: int
    obs_variables: List
    source_mu: List
    # log_std_min: float = -20
    unscale_action: Callable
    std_max: float = 1.0
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    base_covariance_spherical: bool = False
    normalise_scales: bool = True
    # power_target_per_dim: float = 1e-4
    power_for_var: bool = False
    cocontraction: str = None
    explore_coeff: float = 1.0
    MPL_acts: int = 17

    def setup(self):
        if len(set(self.source_mu)) == 1 and self.source_mu[0] == 0:
            self.squash_controls_fn = lambda u: u
        if len(self.source_mu) == 1:
            self.squash_controls_fn = lambda u: nn.sigmoid(u * 5.0)
        else:
            self.squash_controls_fn = (
                lambda u: jnp.concatenate(
                    [
                        nn.sigmoid(u[..., :63] * 5.0),
                        self.unscale_action(nn.tanh(u))[..., 63:],
                    ],
                    axis=-1,
                )
            )  # for MPL, this should be between [low,high] not [-1,1] - unscale does this

        self.flatten = Flatten()

        # self.power_target = self.power_target_per_dim * self.z_dim

        policy = [
            nn.Sequential([nn.Dense(features=h_dim), nn.relu])
            for h_dim in self.net_arch
        ]  # [nn.Dense(features=h_dim), nn.LayerNorm(), nn.relu]
        # if self.base_covariance_spherical:
        #     policy.append(nn.Dense(features=self.z_dim+1))
        # else:
        #     # policy.append(nn.Dense(features=self.z_dim*2+(self.action_dim-self.z_dim))) # policy dependent anistropic noise in orthogonal complement - this makes no sense i think as the axes of orthogonal complement are abirtrary
        #      # policy.append(nn.Dense(features=self.z_dim*2+1)) # policy dependent istropic noise in orthogonal complement
        #      policy.append(nn.Dense(features=self.z_dim*2+1)) # no policy dependent component to noise in orthogonal complement (e.g. uncertainty based or none explicitly)
        # self.policy = policy
        if self.cocontraction == "state_dependent":
            policy.append(nn.Dense(features=(self.z_dim + self.MPL_acts) * 2 + 1))
        else:
            policy.append(nn.Dense(features=(self.z_dim + self.MPL_acts) * 2))
        self.policy = policy

        if self.cocontraction == "state_independent":
            self.u_mu = self.param("u_mu", constant(-np.log((1 / 0.4) - 1)), (1,))

    # @nn.compact
    def __call__(
        self, obs: jnp.ndarray, synergies_state, dynamics_state, key, deterministic
    ) -> tfd.Distribution:  # type: ignore[name-defined] # mu_u_source, A, explore_var, A_complement,
        # def get_participation_ratio(A, Sigma):

        #     eig_vals, _ = jnp.linalg.eigh(Sigma)
        #     eig_vals = jnp.real(eig_vals)
        #     eff_dim = eig_vals.sum()**2/(eig_vals**2).sum()

        #     return eff_dim

        # def get_transition_mean(u, dynamics_state, x, key):

        #     # squash the control inputs before passing them through the dynamics model
        #     u = nn.tanh(u)

        #     y_prime_mean, _ = dynamics_state.apply_fn(dynamics_state.params, x, u, key)

        #     return y_prime_mean

        # batch_get_transition_mean = jax.vmap(get_transition_mean, in_axes=(None,None,None,0))

        # def get_transition_mean_and_uncertainty(u, dynamics_state, x, key, num_MC_samples=5):

        #     # subkeys = jax.random.split(key, self.num_MC_samples)
        #     subkeys = jax.random.split(key, num_MC_samples)

        #     y_prime_mean = batch_get_transition_mean(u, dynamics_state, x, subkeys)

        #     E_y_prime = y_prime_mean.mean(axis=0)

        #     predictive_variance = (y_prime_mean**2).mean(axis=0) - E_y_prime**2

        #     epistemic_uncertainty = predictive_variance.mean() # average predictive variance across control variable dimensions

        #     return E_y_prime, epistemic_uncertainty

        # def half_log_det(Sigma, diagonal_boost = 1e-9):

        #     L = jnp.linalg.cholesky(Sigma + diagonal_boost * jnp.eye(Sigma.shape[-1]))
        #     half_log_det_Sigma = jnp.log(jnp.diagonal(L, axis1 = -2, axis2 = -1)).sum(-1)

        #     return half_log_det_Sigma

        # def get_orthogonal_bases(A, key):

        #     # augment A with random vectors and use the QR decomposition to orthogonalise these vectors relative to the columns of A and each other
        #     A_augmented = jnp.concatenate((A, jax.random.normal(key, (self.action_dim, self.action_dim-self.z_dim))), axis=1)
        #     Q, _ = jnp.linalg.qr(A_augmented)

        #     A_perp = jnp.concatenate((A, Q[:,self.z_dim:]), axis=1)

        #     return A_perp

        # def get_synergies(a_mean, log_std, log_std_full, x, dynamics_state, key):

        #     key, subkey = jax.random.split(key)

        #     mu_u_source = jnp.zeros(self.action_dim)

        #     (Jac, epistemic_uncertainty) = jax.jacrev(get_transition_mean_and_uncertainty, has_aux=True)(mu_u_source, dynamics_state, x, subkey)

        #     # option 1
        #     # A = jnp.linalg.pinv(Jac)

        #     # economic SVD of Jac (full_matrices=False) automatically excludes the right singular vectors that are associated with zero singular values
        #     U, S, Vh = jnp.linalg.svd(Jac, full_matrices=False)

        #     # option 2
        #     # A = Vh.T @ U.T

        #     # option 3
        #     # resolve the sign ambiguity of the right singular vectors (columns of V/rows of Vh) by ensuring the first element of each right singular vector is positive
        #     # Vh = Vh * jnp.where(Vh[:,0] < 0, -1, 1)[:,None]
        #     # A = Vh.T

        #     # option 4
        #     # resolve the sign ambiguity of the right singular vectors (columns of V/rows of Vh) by ensuring the the diagonal elements of the Jacobian dy/dz are positive
        #     # i think this probably makes the most sense
        #     Jac_z = Jac @ Vh.T
        #     Vh = Vh * jnp.where(jnp.diag(Jac_z) < 0, -1, 1)[:,None]
        #     A = Vh.T

        #     # option 5
        #     # resolve the sign ambiguity of the right singular vectors (columns of V/rows of Vh) by multiplying each right singular vector by the sign of the first element of the corresponding left singular vector
        #     # i think this makes no sense as the first element of the left singular vector is abitrary - why not the second or third element - the results will depend on this arbitrary choice
        #     # Vh = Vh * jnp.where(U[0,:] < 0, -1, 1)[:,None]
        #     # A = Vh.T

        #     # normalise synergy vector lengths
        #     # A = A/jnp.linalg.norm(A,axis=0)[None,:]

        #     A_perp = get_orthogonal_bases(A, key)

        #     Sigma_u_source = self.power_target * A @ jnp.eye(self.z_dim) @ A.T
        #     # Sigma_u_source = A @ jnp.eye(self.z_dim) @ A.T

        #     p_ratio_source = get_participation_ratio(A, Sigma_u_source)

        #     y_dim = self.z_dim
        #     Sigma_y_prime = Jac @ Sigma_u_source @ Jac.T
        #     entropy_marginal_transition =  0.5 * y_dim * jnp.log(2*jnp.pi*jnp.exp(1.)) + half_log_det(Sigma_y_prime)

        #     # base_dist = distrax.MultivariateNormalDiag(loc=a_mean, scale_diag=jnp.exp(log_std))
        #     # if self.power_for_var: # this to include uniform scaling as part of base distribution (e.g. in z-space entropy)
        #     #     uniform_scaler = distrax.UnconstrainedAffine(matrix=jnp.sqrt(self.power_target)*jnp.eye(self.z_dim), bias=jnp.zeros(self.z_dim))
        #     #     base_dist = distrax.Transformed(base_dist, uniform_scaler)

        #     # bijector = distrax.Chain([distrax.Block(distrax.Tanh(), 1), distrax.UnconstrainedAffine(matrix=A, bias=jnp.zeros(self.action_dim))])
        #     # pi_dist = distrax.Transformed(base_dist, bijector)

        #     Sigma_u_pi = self.power_target * A @ jnp.diag(jnp.exp(2*log_std)) @ A.T

        #     p_ratio_pi = get_participation_ratio(A, Sigma_u_pi)

        #     pi_base_loc = jnp.sqrt(self.power_target) * A @ a_mean

        #     # pi_base_Sigma = Sigma_u_pi + jnp.eye(self.action_dim) * jnp.exp(log_std_full) * jnp.sqrt(self.power_target) # isotropic
        #     # pi_base_Sigma = Sigma_u_pi + self.power_target * A_perp[:,self.z_dim:] @ (jnp.exp(2*log_std_full) * jnp.eye(self.action_dim-self.z_dim)) @ A_perp[:,self.z_dim:].T # orthogonal complement
        #     pi_base_Sigma = Sigma_u_pi

        #     orthogonal_Sigma = self.power_target * A_perp[:,self.z_dim:] @ (jnp.exp(2*log_std_full) * jnp.eye(self.action_dim-self.z_dim)) @ A_perp[:,self.z_dim:].T + jnp.eye(self.action_dim) * 1e-6 # policy dependent istropic noise in orthogonal complement
        #     # orthogonal_Sigma = self.power_target * A_perp[:,self.z_dim:] @ jnp.diag(jnp.exp(2*log_std_full)) @ A_perp[:,self.z_dim:].T + jnp.eye(self.action_dim) * 1e-6 # policy anistropic dependent noise in orthogonal complement - this makes no sense i think as the axes of orthogonal complement are abirtrary
        #     # orthogonal_Sigma = A_perp[:,self.z_dim:] @ (5**2*epistemic_uncertainty * jnp.eye(self.action_dim-self.z_dim)) @ A_perp[:,self.z_dim:].T + jnp.eye(self.action_dim) * 1e-6 # dynamics uncertainty based noise in orthogonal complement

        #     return pi_base_loc, pi_base_Sigma, orthogonal_Sigma, Jac, A, A_perp, p_ratio_source, p_ratio_pi, entropy_marginal_transition, epistemic_uncertainty

        # batch_get_synergies = jax.vmap(get_synergies, in_axes=(0,0,0,0,None,0))

        x = self.flatten(obs[..., self.obs_variables])
        for fn in self.policy:
            x = fn(x)
        if self.base_covariance_spherical:
            a_mean, log_std = jnp.split(x, [self.z_dim], axis=-1)
            log_std = jnp.repeat(
                jnp.clip(log_std, self.log_std_min, self.log_std_max),
                self.z_dim,
                axis=1,
            )
        else:
            if self.cocontraction == "state_dependent":
                a_mean, logit_std, cocontraction = jnp.split(
                    x,
                    [self.z_dim + self.MPL_acts, (self.z_dim + self.MPL_acts) * 2],
                    axis=-1,
                )
            else:
                a_mean, logit_std = jnp.split(x, 2, axis=-1)
                if self.cocontraction == "state_independent":
                    cocontraction = self.u_mu
                else:
                    cocontraction = jnp.zeros((obs.shape[0], 1))
            # log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)

            # a_mean, log_std, log_std_full = jnp.split(x, [self.z_dim, self.z_dim*2], axis=-1)
            # log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
            # log_std_full = jnp.clip(log_std_full, self.log_std_min, self.log_std_max)
            # std = nn.sigmoid(log_std) * jnp.exp(self.log_std_max)

        # keys = jax.random.split(key, x.shape[0])
        # pi_base_loc, pi_base_Sigma, orthogonal_Sigma ,Jac, A, A_perp, p_ratio_source, p_ratio_pi, entropy_marginal_transition, epistemic_uncertainty = batch_get_synergies(a_mean, log_std, log_std_full, obs, dynamics_state, keys)

        (
            _,
            _,
            mu_u_source,
            A,
            explore_var,
            A_complement,
            mu_power_share,
            residual_power,
            Jac,
            _,
            _,
        ) = synergies_state.apply_fn(
            synergies_state.params,
            obs,
            dynamics_state,
            key,
            deterministic,
            cocontraction,
            self.squash_controls_fn,
        )

        # mu_u_source = jnp.array([-0.4 for _ in range(63)] + [0 for _ in range(17)])
        # explore_var = abs(jax.random.normal(key, (obs.shape[0],)) * .01)
        # A = jax.random.normal(jax.random.PRNGKey(0), (obs.shape[0],self.action_dim, self.z_dim))
        # A, _ = jnp.linalg.qr(A)
        # A_complement = jax.random.normal(jax.random.PRNGKey(0), (obs.shape[0],self.action_dim, self.action_dim-self.z_dim))
        # A_complement, _ = jnp.linalg.qr(A_complement)
        # mu_power_share = jax.random.normal(key, (obs.shape[0],1))
        # residual_power = jax.random.normal(key, (obs.shape[0],1))
        # Jac = jax.random.normal(key, (obs.shape[0],1,self.z_dim,self.action_dim))

        # base_dist = distrax.MultivariateNormalDiag(loc=a_mean, scale_diag=jnp.exp(log_std))
        # # base_dist = distrax.MultivariateNormalDiag(loc=a_mean, scale_diag=jnp.exp(log_std)*jnp.sqrt(residual_power / self.z_dim))
        # if self.power_for_var: # this to include uniform scaling as part of base distribution (e.g. in z-space entropy)
        #     # uniform_scaler = distrax.UnconstrainedAffine(matrix=jnp.sqrt(self.power_target)*jnp.eye(self.z_dim), bias=jnp.zeros(self.z_dim))
        #     uniform_scaler = distrax.UnconstrainedAffine(matrix=jnp.sqrt(residual_power[:,:,None] / self.z_dim)*jnp.eye(self.z_dim)[None,:,:], bias=jnp.zeros(self.z_dim))
        #     base_dist = distrax.Transformed(base_dist, uniform_scaler)
        #     # base_dist = distrax.MultivariateNormalDiag(loc=a_mean, scale_diag=jnp.sqrt(residual_power / self.z_dim)*jnp.clip(nn.softplus(log_std), jnp.exp(self.log_std_min), jnp.exp(self.log_std_max)))
        #     # base_dist = distrax.MultivariateNormalDiag(loc=a_mean, scale_diag=jnp.sqrt(residual_power / self.z_dim)*nn.tanh(log_std))

        # base_dist = distrax.MultivariateNormalDiag(loc=a_mean[...,:self.z_dim], scale_diag=nn.sigmoid(logit_std[...,:self.z_dim])*self.std_max)
        # MPL_dist = distrax.MultivariateNormalDiag(loc=a_mean[...,self.z_dim:], scale_diag=nn.sigmoid(logit_std[...,self.z_dim:])*1.)

        base_dist = distrax.MultivariateNormalDiag(
            loc=a_mean, scale_diag=nn.sigmoid(logit_std) * self.std_max
        )

        # base_dist = distrax.MultivariateNormalFullCovariance(loc=pi_base_loc, covariance_matrix=pi_base_Sigma)
        # pi_dist = distrax.Transformed(base_dist, distrax.Block(distrax.Tanh(), 1))

        # pi_base_cov = jax.vmap(lambda A_perp, log_stds: self.power_target * A_perp @ jnp.diag(jnp.exp(2*log_stds)) @ A_perp.T)(A_perp, jnp.concatenate((log_std,jnp.repeat(log_std_full,self.action_dim-self.z_dim,axis=1)),axis=1)) # policy dependent istropic noise in orthogonal complement
        # pi_base_cov = jax.vmap(lambda A_perp, variances: self.power_target * A_perp @ jnp.diag(variances) @ A_perp.T)(A_perp, jnp.concatenate((jnp.exp(2*log_std),jnp.repeat(5**2*epistemic_uncertainty[:,None]+1e-6,self.action_dim-self.z_dim,axis=1)),axis=1)) # dynamics uncertainty based noise in orthogonal complement
        # pi_base_dist = distrax.MultivariateNormalFullCovariance(loc=mu_u_source+jnp.concatenate((a_mean,jnp.zeros((obs.shape[0],self.action_dim-self.z_dim))),axis=1), covariance_matrix=pi_base_cov)
        # pi_dist = distrax.Transformed(pi_base_dist, distrax.Block(distrax.Tanh(), 1))

        # explore_dist = distrax.MultivariateNormalFullCovariance(loc=jnp.zeros((obs.shape[0],self.action_dim)), covariance_matrix=jnp.eye(self.action_dim)*1e-6)
        # explore_dist = distrax.MultivariateNormalFullCovariance(loc=jnp.zeros((obs.shape[0],self.action_dim)), covariance_matrix=explore_cov)
        explore_dist = distrax.MultivariateNormalDiag(
            scale_diag=jnp.repeat(
                self.explore_coeff * jnp.sqrt(explore_var)[:, None],
                63 - self.z_dim,
                axis=1,
            )
        )

        return (
            base_dist,
            explore_dist,
            A,
            mu_u_source,
            A_complement,
            self.squash_controls_fn,
        )
        # return base_dist, pi_dist, A_scaled, mu_u_source


class SACPolicy(BaseJaxPolicy):
    action_space: spaces.Box  # type: ignore[assignment]

    def __init__(
        self,
        MjModel,
        max_num_mode_switches,
        vec_stats,
        control_variables: List,
        task_variables: List,
        obs_variables: List,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        init_axes_scales: float,
        power_coefficient: float,
        init_power_coeff: float,
        std_max: float,
        dynamics_dropout: float,
        cocontraction: bool,
        num_MC_samples: int,
        explore_coeff: float,
        dynamics_learning_rate: float,
        source_mu: List,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        dropout_rate: float = 0.0,
        layer_norm: bool = False,
        activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu,
        use_sde: bool = False,
        # Note: most gSDE parameters are not used
        # this is to keep API consistent with SB3
        log_std_init: float = -3,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class=None,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Callable[..., optax.GradientTransformation] = optax.adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=False,
        )
        self.dropout_rate = dropout_rate
        self.layer_norm = layer_norm
        if net_arch is not None:
            if isinstance(net_arch, list):
                self.net_arch_pi = self.net_arch_qf = net_arch
            else:
                self.net_arch_pi = net_arch["pi"]
                self.net_arch_qf = net_arch["qf"]
        else:
            self.net_arch_pi = self.net_arch_qf = [256, 256]
        self.n_critics = n_critics
        self.use_sde = use_sde
        self.activation_fn = activation_fn

        self.key = self.noise_key = jax.random.PRNGKey(0)

        self.MjModel = MjModel
        self.max_num_mode_switches = max_num_mode_switches
        self.vec_stats = vec_stats
        self.control_variables = control_variables
        self.task_variables = task_variables
        self.obs_variables = obs_variables
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

    def build(
        self, key: jax.Array, lr_schedule: Schedule, qf_learning_rate: float
    ) -> jax.Array:
        (
            key,
            actor_key,
            qf_key,
            dropout_key,
            source_key,
            dynamics_key,
            power_key,
            synergies_key,
        ) = jax.random.split(key, 8)
        # Keep a key for the actor
        key, self.key = jax.random.split(key, 2)
        # Initialize noise
        self.reset_noise()

        if isinstance(self.observation_space, spaces.Dict):
            obs = jnp.array(
                [
                    spaces.flatten(
                        self.observation_space, self.observation_space.sample()
                    )
                ]
            )
        else:
            obs = jnp.array([self.observation_space.sample()])
        action = jnp.array([self.action_space.sample()])
        if action.size < len(self.control_variables):
            z_action = jnp.copy(action)
        else:
            z_action = action[:, self.control_variables]

        #####################################

        push_forward_type = "linear"

        power_for_var = True

        normalise_scales = True

        #####################################

        if self.obs_variables == []:
            self.obs_variables = list(range(self.observation_space.shape[-1]))

        self.dynamics = dynamics(
            h_dims_dynamics=[256, 256],
            task_variables=self.task_variables,
            control_variables=self.control_variables,
            drop_out_rate=self.dynamics_dropout,
            obs_variables=self.obs_variables,
            action_variables_dynamics=list(range(63)),
        )

        self.dynamics_state = TrainState.create(
            apply_fn=self.dynamics.apply,
            params=self.dynamics.init(dynamics_key, obs, action, dynamics_key, True),
            # apply_fn=partial(lambda dynamics_apply, obs_variables, params, s, u, key, deterministic: dynamics_apply(params, s[...,obs_variables], u, key, deterministic), self.dynamics.apply, self.obs_variables),
            # params=self.dynamics.init(dynamics_key, obs[...,self.obs_variables], action, dynamics_key, True),
            # optax.chain(
            # optax.clip_by_global_norm(max_grad_norm), # TODO?
            tx=self.optimizer_class(
                learning_rate=self.dynamics_learning_rate,  # type: ignore[call-arg] # lr_schedule(1)
                **self.optimizer_kwargs,
            ),
            # ),
        )

        # if len(self.source_mu) == 1:
        #     self.squash_controls_fn = jax.tree_util.Partial(lambda u: nn.sigmoid(u*5.))
        # else:
        #     self.squash_controls_fn = jax.tree_util.Partial(lambda u: jnp.concatenate([nn.sigmoid(u[...,:63]*5.),nn.tanh(u[...,63:]*3.)],axis=-1))

        self.synergies = synergies(
            h_dims_source=[256, 256],
            action_dim=int(np.prod(self.action_space.shape)),
            z_dim=len(self.control_variables),
            source_mu=self.source_mu,
            power_target_per_dim=0.05,
            cocontraction=self.cocontraction,
            num_MC_samples=self.num_MC_samples,
        )

        self.synergies_state = TrainState.create(
            apply_fn=self.synergies.apply,
            params=self.synergies.init(
                synergies_key,
                obs,
                self.dynamics_state,
                synergies_key,
                True,
                jnp.zeros((obs.shape[0], 1)),
                lambda u: u,
            ),
            # apply_fn=partial(lambda synergies_apply, obs_variables, params, obs, dynamics_state, key, deterministic, cocontraction: synergies_apply(params, obs[...,obs_variables], dynamics_state, key, deterministic, cocontraction), self.synergies.apply, self.obs_variables),
            # params=self.synergies.init(synergies_key, obs[...,self.obs_variables], self.dynamics_state, synergies_key, True, jnp.zeros((obs.shape[0],1))),
            # optax.chain(
            # optax.clip_by_global_norm(max_grad_norm), # TODO?
            tx=self.optimizer_class(
                learning_rate=self.dynamics_learning_rate,  # type: ignore[call-arg] # lr_schedule(1)
                **self.optimizer_kwargs,
            ),
            # ),
        )

        # wrap the function with Partial so it is a valid argument to a jitted function
        # https://github.com/google/jax/issues/1443#issuecomment-1527813792
        self.posterior = jax.tree_util.Partial(
            posterior_inference, control_variables=self.control_variables
        )

        self.power_coef = power_coef(self.init_power_coeff)

        self.power_coef_state = TrainState.create(
            apply_fn=self.power_coef.apply,
            params=self.power_coef.init(power_key)["params"],
            tx=self.optimizer_class(
                learning_rate=lr_schedule(1),  # type: ignore[call-arg]
                **self.optimizer_kwargs,
            ),
        )

        # self.source = source(h_dims_source=[256,256],
        #              h_dims_inverse=[256,256],
        #              h_dims_dynamics=self.dynamics.h_dims_dynamics,
        #              h_dims_conditioner=256,
        #              num_coupling_layers=2,
        #              task_variables=self.task_variables,
        #              control_variables=self.control_variables,
        #              state_dependent_base_variance=False,
        #              init_axes_scales=self.init_axes_scales,
        #              power_coefficient=self.power_coefficient,
        #              init_power_coeff=self.init_power_coeff,
        #              power_target=self.power_target,
        #              a_dim=int(np.prod(self.action_space.shape)),
        #              drop_out_rate=self.dynamics_dropout,
        #              push_forward_type=push_forward_type,
        #              power_for_var=power_for_var,
        #              normalise_scales=normalise_scales)

        # # use a different learning rate for the decoder_axes_scales params
        # # https://github.com/google-deepmind/optax/issues/59
        # # https://dm-haiku.readthedocs.io/en/latest/api.html
        # import haiku as hk
        # # a pytree with the same structure as params with boolean values at the leaves
        # source_params = self.source.init(source_key, obs, source_key, self.dynamics_state.params, 0.)
        # mask = hk.data_structures.map(lambda module_name, name, value: name == 'decoder_axes_scales' or name == 'house_holder', source_params)
        # not_mask = jax.tree_map(lambda x: not x, mask)

        # self.source_state = TrainState.create(
        #     apply_fn=self.source.apply,
        #     params=source_params,
        #     tx= optax.chain(
        #         # only applies to leaves where mask is True:
        #         optax.masked(self.optimizer_class(
        #                      learning_rate=self.dynamics_learning_rate,  # type: ignore[call-arg] # lr_schedule(1)
        #                      **self.optimizer_kwargs), mask),
        #         # only applies to leaves where not_mask is True:
        #         optax.masked(self.optimizer_class(
        #                      learning_rate=lr_schedule(1),  # type: ignore[call-arg] # lr_schedule(1)
        #                      **self.optimizer_kwargs), not_mask)
        #         )
        #         # self.optimizer_class(
        #         #     learning_rate=lr_schedule(1),  # type: ignore[call-arg] # lr_schedule(1)
        #         #     **self.optimizer_kwargs,
        #         # ),
        #     # ),
        # )

        self.source_state = None

        # if power_for_var:
        # self.source_state.params['params']['source_policy_2']['kernel'] = self.source_state.params['params']['source_policy_2']['kernel'] * 0.
        # log_std_max = jnp.log(jnp.sqrt(self.power_target))
        # else:
        # log_std_max = 0.
        # log_std_max = np.log(1.) # np.log(10), np.log(1)

        # testing this to see role of power!!!
        # log_std_max = jnp.log(jnp.sqrt(self.power_target))

        self.actor = Actor(
            action_dim=int(np.prod(self.action_space.shape)),
            z_dim=len(self.control_variables),
            # net_arch=self.net_arch_pi,
            net_arch=[256, 256],  # self.source.h_dims_source
            push_forward_type=push_forward_type,
            h_dims_conditioner=256,  # self.source.h_dims_conditioner
            num_coupling_layers=2,  # self.source.num_coupling_layers
            activation_fn=self.activation_fn,
            std_max=self.std_max,
            normalise_scales=normalise_scales,
            # power_target_per_dim=self.power_target,
            power_for_var=power_for_var,
            cocontraction=self.cocontraction,
            explore_coeff=self.explore_coeff,
            obs_variables=self.obs_variables,
            source_mu=self.source_mu,
            unscale_action=self.unscale_action,
        )
        # Hack to make gSDE work without modifying internal SB3 code
        self.actor.reset_noise = self.reset_noise

        # mu_u_source = jnp.array([-0.4 for _ in range(63)] + [0 for _ in range(17)])
        # explore_var = abs(jax.random.normal(key, (obs.shape[0],)) * .01)
        # A = jax.random.normal(jax.random.PRNGKey(0), (obs.shape[0],self.action_dim, self.z_dim))
        # A, _ = jnp.linalg.qr(A)
        # A_complement = jax.random.normal(jax.random.PRNGKey(0), (obs.shape[0],self.action_dim, self.action_dim-self.z_dim))
        # A_complement, _ = jnp.linalg.qr(A_complement)

        self.actor_state = TrainState.create(
            apply_fn=self.actor.apply,
            params=self.actor.init(
                actor_key,
                obs,
                self.synergies_state,
                self.dynamics_state,
                actor_key,
                True,
            ),  # jnp.zeros((1,80)), jnp.zeros((1,80,14)), jnp.zeros((1,)), jnp.zeros((1,80,66)),
            # apply_fn=partial(lambda actor_apply, obs_variables, params, obs, synergies_state, dynamics_state, key, deterministic: actor_apply(params, obs[...,obs_variables], synergies_state, dynamics_state, key, deterministic), self.actor.apply, self.obs_variables),
            # params=self.actor.init(actor_key, obs[...,self.obs_variables], self.synergies_state, self.dynamics_state, actor_key, True),
            tx=self.optimizer_class(
                learning_rate=lr_schedule(1),  # type: ignore[call-arg]
                **self.optimizer_kwargs,
            ),
        )

        self.qf = VectorCritic(
            obs_variables=self.obs_variables,
            dropout_rate=self.dropout_rate,
            use_layer_norm=self.layer_norm,
            net_arch=self.net_arch_qf,
            n_critics=self.n_critics,
            activation_fn=self.activation_fn,
        )

        self.qf_state = RLTrainState.create(
            # apply_fn=self.qf.apply,
            apply_fn=partial(
                lambda obs_variables, params, obs, action, rngs: self.qf.apply(
                    params, obs[..., obs_variables], action, rngs=rngs
                ),
                self.obs_variables,
            ),
            params=self.qf.init(
                {"params": qf_key, "dropout": dropout_key},
                # obs,
                obs[..., self.obs_variables],
                action,  # a space Q function
                # z_action, # z space Q function
            ),
            target_params=self.qf.init(
                {"params": qf_key, "dropout": dropout_key},
                # obs,
                obs[..., self.obs_variables],
                action,  # a space Q function
                # z_action, # z space Q function
            ),
            tx=self.optimizer_class(
                learning_rate=qf_learning_rate,  # type: ignore[call-arg]
                **self.optimizer_kwargs,
            ),
        )

        self.actor.apply = jax.jit(self.actor.apply)  # type: ignore[method-assign]
        self.qf.apply = jax.jit(  # type: ignore[method-assign]
            self.qf.apply,
            static_argnames=("dropout_rate", "use_layer_norm"),
        )

        self.phase_two = np.array(False)
        self.phase_three = np.array(False)
        self.phase_four = np.array(False)
        self.phase_five = np.array(False)

        MjData = mujoco.MjData(self.MjModel)
        jacp = np.zeros((3, self.MjModel.nv))  # translation jacobian
        jacr = np.zeros((3, self.MjModel.nv))  # rotational jacobian
        # eef_name = "prosthesis/palm"
        eef_name = "prosthesis/middle0"
        self.prosthesis_ids = np.arange(4)
        self.body_id = self.MjModel.body(eef_name).id
        self.ik = GradientDescentIK(
            self.MjModel, MjData, step_size=0.1, tol=0, alpha=0.1, jacp=jacp, jacr=jacr
        )

        self.target_qpos = np.zeros(4)

        self.touching_mpl_count = 0
        self.touching_myo_count = 0

        return key

    def reset_noise(self, batch_size: int = 1) -> None:
        """
        Sample new weights for the exploration matrix, when using gSDE.
        """
        self.key, self.noise_key = jax.random.split(self.key, 2)

    def forward(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        return self._predict(obs, deterministic=deterministic)

    def _predict(
        self, observation: np.ndarray, phase, deterministic: bool = False
    ) -> np.ndarray:  # type: ignore[override]
        if deterministic:
            return BaseJaxPolicy.select_action(
                self.actor_state.params,
                self.actor_state,
                observation,
                self.noise_key,
                self.synergies_state,
                self.dynamics_state,
                phase,
            )
        # Trick to use gSDE: repeat sampled noise by using the same noise key
        if not self.use_sde:
            self.reset_noise()
        action, _, _ = BaseJaxPolicy.sample_action(
            self.actor_state.params,
            self.actor_state,
            observation,
            self.noise_key,
            self.synergies_state,
            self.dynamics_state,
            phase,
        )
        return action
