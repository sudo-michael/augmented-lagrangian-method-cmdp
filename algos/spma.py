import time
from typing import Optional
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from flax import struct

from algos.base_algo import Algo
from environments.tabular_env import TabularEnv


def construct_features(env, env_id, feature_str):
    # assert env_id in ["ConstrainedCliffWorld"], "unknown envirnoment features"
    # match feature_dim:
    #     case 40:
    #         num_tiles = 5
    #         tiling_size = 1
    #     case 50:
    #         num_tiles = 6
    #         tiling_size = 1
    #     case 60:
    #         num_tiles = 4
    #         tiling_size = 3
    #     case 80:
    #         num_tiles = 5
    #         tiling_size = 3
    #     case 100:
    #         num_tiles = 6
    #         tiling_size = 3
    #     case _:
    #         raise ValueError(f"unsupported feature_dim: {feature_dim}")
    if feature_str == "one_hot":
        d = env.S * env.A
        features = np.zeros((env.S, env.A, d))
        for s in range(env.S):
            for a in range(env.A):
                features[s, a, s * env.A + a] = 1
    else:
        raise ValueError(f"unsupported feature_str: {feature_str}")

    return features


class Actor(struct.PyTreeNode):
    theta: jnp.array
    features: jnp.array
    eta: float
    armijo_line_search_eta_max: float = 1e5
    armijo_line_search_c: float = 0.5
    armijo_line_search_beta: float = 0.9

    @classmethod
    def create(
        cls,
        actor_key,
        d: int,
        features: jnp.ndarray,
        eta,
        armijo_line_search_eta_max,
        armijo_line_search_c,
        armijo_line_search_beta,
    ):
        theta = jax.random.normal(actor_key, d)
        return cls(
            theta=theta,
            features=features,
            eta=eta,
            armijo_line_search_eta_max=armijo_line_search_eta_max,
            armijo_line_search_c=armijo_line_search_c,
            armijo_line_search_beta=armijo_line_search_beta,
        )

    @jax.jit
    def policy(self, theta):
        return jax.nn.softmax(jnp.einsum("sad,d->sa", self.features, theta), axis=1)

    @jax.jit
    def loss_fn(self, theta, adv_hat_pit, pit):
        logits = jnp.einsum("sad,d->sa", self.features, theta)
        pitheta = jax.nn.softmax(logits, axis=-1)
        # prevent div by 0 error / log(0)
        eps = 1e-10
        ratio = jnp.maximum(eps, pitheta / (pit + eps))
        etap = self.eta
        # pit * jnp.log(ratio) = (ratio - 1 - jnp.log(ratio))
        return jnp.mean(pit * adv_hat_pit - 1 / etap * (ratio - 1 - jnp.log(ratio)))

    @jax.jit
    def update(
        self,
        adv_hat_pit,
        pit,
    ):
        theta_t = self.theta
        grad = jax.grad(self.loss_fn)(theta_t, adv_hat_pit, pit)

        def armijo_condition_not_satisfied(eta):
            return self.loss_fn(theta_t + eta * grad, adv_hat_pit, pit) < (
                self.loss_fn(theta_t, adv_hat_pit, pit)
                + self.armijo_line_search_c * eta * jnp.linalg.norm(grad) ** 2
            )

        def reduce_eta(eta):
            return eta * self.armijo_line_search_beta

        step_size = jax.lax.while_loop(
            armijo_condition_not_satisfied, reduce_eta, self.armijo_line_search_eta_max
        )

        theta_tp1 = theta_t + step_size * grad

        # ML: Reza suggests to use this for debugging --- maybe will use later
        # kl div
        # pi_tp1 = jax.nn.softmax(jnp.einsum("sad,d->sa", self.features, theta_tp1))
        # ratio = pi_tp1 / (pit + 1e-10)
        # # kl div(pi_t, pi_tp1)
        # kl_div = (ratio - 1) - jnp.log(ratio)

        return (
            self.replace(theta=theta_tp1),
            jnp.linalg.norm(jax.grad(self.loss_fn)(theta_tp1, adv_hat_pit, pit)),
            1.8 * step_size,  # trick
        )


class Critic(struct.PyTreeNode):
    omega: jnp.array
    features: jnp.array
    c: float
    armijo_line_search_eta_max: float = 1e5
    armijo_line_search_c: float = 0.5
    armijo_line_search_beta: float = 0.9

    @classmethod
    def create(
        cls,
        critic_key,
        d: int,
        features: jnp.ndarray,
        c: float,
        armijo_line_search_eta_max,
        armijo_line_search_c,
        armijo_line_search_beta,
    ):
        """Args:
        d (int): number of parameters
        features (S x A x d): features
        c (float):
        """
        omega = jax.random.normal(critic_key, d)
        return cls(
            omega=omega,
            features=features,
            c=c,
            armijo_line_search_eta_max=armijo_line_search_eta_max,
            armijo_line_search_c=armijo_line_search_c,
            armijo_line_search_beta=armijo_line_search_beta,
        )

    @jax.jit
    def estimate_a_hat(self, pit):
        q_hat_pit = jnp.einsum("sad,d->sa", self.features, self.omega)
        v_hat_pit = jnp.einsum("sa,sa->s", pit, q_hat_pit)
        adv_hat_pit = q_hat_pit - v_hat_pit.reshape(-1, 1)
        return adv_hat_pit

    @jax.jit
    def loss_fn(self, omega, adv_pit, pit):
        # E.3 https://arxiv.org/pdf/2305.15249#page=7.52
        q_hat_pit = jnp.einsum("sad,d->sa", self.features, omega)
        v_hat_pit = jnp.einsum("sa,sa->s", q_hat_pit, pit)
        a_hat_pit = q_hat_pit - v_hat_pit.reshape(-1, 1)

        # eps = 1e-10
        # HACK: if x <= 0, log(x) is undefined. Maybe masking out the gradient / log is the right idea
        #       but in the ACPG code base it is clipped instead: jnp.maximum(1 - self.c * (adv_pit - a_hat_pit), eps)
        # inner_term = 1 - self.c * (adv_pit - a_hat_pit)
        # safe_log_inner_term = jnp.log(jnp.where(inner_term > 0.0, inner_term, eps))

        # loss = (1 / self.c) * (inner_term * safe_log_inner_term).mean()
        return jnp.mean((adv_pit - a_hat_pit) ** 2)

    @jax.jit
    def update(self, adv_pit, pit):
        """
        Args:
            a_pit: (estimated) advantage function
            pi_t: policy probability
            d_pit: state visitation distribution
            eta: step-size
        """

        omega_t = self.omega
        grad = jax.grad(self.loss_fn)(omega_t, adv_pit, pit)

        def armijo_condition_not_satisfied(eta):
            return self.loss_fn(omega_t - eta * grad, adv_pit, pit) > (
                self.loss_fn(omega_t, adv_pit, pit)
                - self.armijo_line_search_c * eta * jnp.linalg.norm(grad) ** 2
            )

        def reduce_eta(eta):
            return eta * self.armijo_line_search_beta

        step_size = jax.lax.while_loop(
            armijo_condition_not_satisfied, reduce_eta, self.armijo_line_search_eta_max
        )

        omega_tp1 = omega_t - step_size * grad

        return (
            self.replace(omega=omega_tp1),
            jnp.linalg.norm(self.loss_fn(omega_tp1, adv_pit, pit)),
            1.8 * step_size,  # trick for faster line-search termination
        )


class StochasticSPMA(Algo):
    iterations_outer: int
    iterations_inner: int
    init_lambda: float
    init_tau: float
    beta: float
    max_tau: float
    eta: float
    c: float
    iterations_actor: int
    target_grad_norm_actor: float
    iterations_critic: int
    target_grad_norm_critic: float
    tau_stratergy: int
    num_trajs: int
    armijo_line_search_eta_max: float
    armijo_line_search_c: float
    armijo_line_search_beta: float
    functional_representation: str = struct.field(pytree_node=False)
    actor_feature_str: str = struct.field(pytree_node=False)
    critic_feature_str: str = struct.field(pytree_node=False)
    actor: Actor
    reward_critic: Critic
    alm_utility_critic: Critic
    actor_features: jnp.ndarray

    @classmethod
    def create(
        cls,
        iterations_outer,
        iterations_inner,
        init_lambda,
        init_tau,
        beta,
        max_tau,
        eta,
        c,
        tau_stratergy,
        num_trajs,
        iterations_actor,
        iterations_critic,
        target_grad_norm_actor,
        target_grad_norm_critic,
        armijo_line_search_eta_max,
        armijo_line_search_c,
        armijo_line_search_beta,
        functional_representation,
        actor_feature_str,
        critic_feature_str,
    ):
        return cls(
            iterations_outer=iterations_outer,
            iterations_inner=iterations_inner,
            init_lambda=init_lambda,
            init_tau=init_tau,
            beta=beta,
            max_tau=max_tau,
            eta=eta,
            c=c,
            tau_stratergy=tau_stratergy,
            num_trajs=num_trajs,
            iterations_actor=iterations_actor,
            iterations_critic=iterations_critic,
            target_grad_norm_actor=target_grad_norm_actor,
            target_grad_norm_critic=target_grad_norm_critic,
            functional_representation=functional_representation,
            actor_feature_str=actor_feature_str,
            critic_feature_str=critic_feature_str,
            armijo_line_search_eta_max=armijo_line_search_eta_max,
            armijo_line_search_c=armijo_line_search_c,
            armijo_line_search_beta=armijo_line_search_beta,
            actor=None,
            reward_critic=None,
            alm_utility_critic=None,
            actor_features=None,
        )

    # @jax.jit
    def grad_aug_lag_exact(self, theta, lmbda, tau, env):
        pi = self.actor.policy(theta)

        # JIT-compatible version
        cons = env.constraint(pi)
        condition = cons >= env.b + lmbda / tau
        Gamma_pi = jnp.where(
            condition, env.r, env.r + lmbda * env.c - tau * (cons - env.b) * env.c
        )

        q_Gamma_pi = env.calc_qpi(pi, Gamma_pi)

        d_pi = env.calc_dpi(pi)

        @jax.grad
        def pg(theta):
            return jnp.dot(
                d_pi,
                jnp.einsum("sa, sa->s", q_Gamma_pi, self.actor.policy(theta)),
            )

        return pg(theta)

    def calc_alm_utility(self, env, theta, lmbda, tau, key, sample_fn):
        qr_pit, qc_pit = env.estimate_q_pi_mc(key, self.num_trajs, sample_fn, theta)
        return self._calc_alm_utility_core(env, theta, lmbda, tau, qr_pit, qc_pit)

    @jax.jit
    def _calc_alm_utility_core(self, env, theta, lmbda, tau, qr_pit, qc_pit):
        pit = self.actor.policy(theta)
        vr_pit = jnp.einsum("sa,sa->s", pit, qr_pit)
        vc_pit = jnp.einsum("sa,sa->s", pit, qc_pit)

        cons_hat = vc_pit.mean()
        use_only_reward = cons_hat >= env.b + lmbda / tau
        adv_Gamma_pit = jnp.where(
            use_only_reward,
            qr_pit - vr_pit.reshape(-1, 1),
            (qr_pit - vr_pit.reshape(-1, 1))
            + lmbda * (qc_pit - vc_pit.reshape(-1, 1))
            - tau * (cons_hat - env.b) * (qc_pit - vc_pit.reshape(-1, 1)),
        )
        return adv_Gamma_pit, use_only_reward

    def actor_update(self, adv_pit: jnp.ndarray, pit):
        eta_max = self.armijo_line_search_eta_max
        actor = self.actor
        for i in range(self.iterations_actor):
            actor, actor_grad_norm, eta_max = actor.update(
                adv_pit,
                pit,
            )

            if actor_grad_norm < self.target_grad_norm_actor:
                break
        total_actor_iters = i
        last_step_size = eta_max / 1.8
        last_actor_grad_norm = actor_grad_norm.item()
        return actor, total_actor_iters, last_step_size, last_actor_grad_norm

    def critic_update(self, critic: Critic, adv_pit: jnp.ndarray, pit: jnp.ndarray):
        eta_max = self.armijo_line_search_eta_max
        for i in range(self.iterations_critic):
            # print(f'iter: {i=}, {critic.omega}')
            critic, critic_grad_norm, eta_max = critic.update(
                adv_pit,
                pit,
            )
            if critic_grad_norm < self.target_grad_norm_critic:
                break
        total_critic_iters = i
        last_step_size = eta_max / 1.8
        last_critic_grad_norm = critic_grad_norm.item()
        return critic, total_critic_iters, last_step_size, last_critic_grad_norm

    def spma_update(self, env: TabularEnv, theta, lmbda, tau, key, sample_fn):
        key, adv_key = jax.random.split(key)
        qr_pit, qc_pit = env.estimate_q_pi_mc(key, self.num_trajs, sample_fn, theta)

        pit = self.actor.policy(theta)

        # print(f'omega before critic update: {self.reward_critic.omega}')
        # (
        #     new_reward_critic,
        #     total_critic_iters,
        #     last_critic_step_size,
        #     last_critic_grad_norm,
        # ) = self.critic_update(self.reward_critic, qr_pit, pit)
        # self = self.replace(reward_critic=new_reward_critic)
        # adv_hat_pit = self.reward_critic.estimate_a_hat(pit)
        # taget
        total_critic_iters = 0
        last_critic_step_size = 0
        last_critic_grad_norm = 0
        qr_pit = env.calc_qpi(pit, env.r)
        vr_pit = jnp.einsum("sa, sa->s", qr_pit, pit)
        adv_pit = jnp.einsum("sa, s->sa", qr_pit, vr_pit)
        # print(last_critic_grad_norm)
        # print(jnp.mean((adv_pit - adv_hat_pit)**2))
        # print(f'omega after critic update: {self.reward_critic.omega}')
        # breakpoint()

        (
            new_actor,
            total_actor_iters,
            last_actor_step_size,
            last_actor_grad_norm,
        ) = self.actor_update(
            adv_pit,
            pit,
        )
        self = self.replace(actor=new_actor)

        info = {
            "total_critic_iters": total_critic_iters,
            "last_critic_step_size": last_critic_step_size,
            "last_critic_grad_norm": last_critic_grad_norm,
            "total_actor_iters": total_actor_iters,
            "last_actor_step_size": last_actor_step_size,
            "last_actor_grad_norm": last_actor_grad_norm,
        }

        return self, info

    def train(
        self,
        log_dir: str,
        env: TabularEnv,
        env_id: str,
        pi_star: Optional[jnp.ndarray] = None,
        seed: Optional[int] = 0,
        verbose: Optional[bool] = False,
    ):
        features_actor = construct_features(env, env_id, self.actor_feature_str)
        features_critic = construct_features(env, env_id, self.critic_feature_str)

        key = jax.random.PRNGKey(seed)
        key, actor_key, reward_critic_key, alm_utility_critic_key = jax.random.split(
            key, 4
        )

        # fmt:off
        actor = Actor.create(
            actor_key=actor_key,
            d=env.S * env.A
            if self.actor_feature_str == "one_hot" else features_actor.shape[-1],
            features=jnp.array(features_actor), eta=self.eta,
            armijo_line_search_beta=self.armijo_line_search_beta,
            armijo_line_search_c=self.armijo_line_search_c,
            armijo_line_search_eta_max=self.armijo_line_search_eta_max,
        )
        reward_critic = Critic.create(
            critic_key=reward_critic_key,
            d=env.S * env.A if self.critic_feature_str == "one_hot" else features_critic.shape[-1],
            features=jnp.array(features_critic),
            c=self.c,
            armijo_line_search_beta=self.armijo_line_search_beta,
            armijo_line_search_c=self.armijo_line_search_c,
            armijo_line_search_eta_max=self.armijo_line_search_eta_max,
        )
        alm_utility_critic = Critic.create(
            critic_key=alm_utility_critic_key,
            d=env.S * env.A if self.critic_feature_str == "one_hot" else features_critic.shape[-1], features=jnp.array(features_critic),
            c=self.c,
            armijo_line_search_beta=self.armijo_line_search_beta,
            armijo_line_search_c=self.armijo_line_search_c,
            armijo_line_search_eta_max=self.armijo_line_search_eta_max,
        )
        self = self.replace(
            actor=actor, reward_critic=reward_critic, alm_utility_critic=alm_utility_critic, actor_features=actor.features
        )
        # fmt:on

        v_r_pis = []
        v_c_pis = []
        lambdas = []
        taus = []
        gradient_evaluations_per_iteration = []

        lambda_t = self.init_lambda
        tau_t = self.init_tau

        pi_t = self.actor.policy(self.actor.theta)
        v_r = env.objective(pi_t)
        v_c = env.constraint(pi_t)

        v_r_pis.append(v_r.item())
        v_c_pis.append(v_c.item())
        taus.append(tau_t)
        lambdas.append(lambda_t)

        start_time = time.time()

        def sample_fn(key, theta: jnp.ndarray, state: jnp.ndarray):
            logits = jnp.einsum("ad,d->a", self.actor_features[state], theta)
            return jax.random.categorical(key, logits)

        # DO STUFF HERE
        for t in tqdm.tqdm(range(self.iterations_outer), desc="T", leave=False):
            theta_t = actor.theta
            self, info = self.spma_update(
                env, actor.theta, lambda_t, tau_t, key, sample_fn
            )

            # log things
            pi_t = self.actor.policy(theta_t)
            v_r = env.objective(pi_t)
            v_c = env.constraint(pi_t)

            v_r_pis.append(v_r.item())
            v_c_pis.append(v_c.item())
            taus.append(tau_t)
            lambdas.append(lambda_t)
            gradient_evaluations_per_iteration.append(self.iterations_inner)

        end_time = time.time()

        log_info = {
            "v_r_pis": v_r_pis,
            "v_c_pis": v_c_pis,
            "lambas": lambdas,
            "taus": taus,
            "gradient_evaluations_per_iteration": gradient_evaluations_per_iteration,
        }
        pi_T = self.actor.policy(theta_t)

        self.save_log(
            log_info=log_info,
            log_dir=log_dir,
            env=env,
            env_id=env_id,
            pi_T=pi_T,
            pi_star=pi_star,
            seed=seed,
            elapsed_time=end_time - start_time,
        )

        return True
