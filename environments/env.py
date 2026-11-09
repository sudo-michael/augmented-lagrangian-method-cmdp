import numpy as np
import scipy.optimize


class Env:
    def __init__(
        self,
        P: np.ndarray,
        r: np.ndarray,
        c: np.ndarray,
        b: float,
        rho: np.ndarray,
        gamma: float,
    ):
        self.S: int = P.shape[0]
        self.A: int = P.shape[1]
        self.P: np.ndarray = P
        self.r: np.ndarray = r
        self.c: np.ndarray = c
        self.b: float = b
        self.rho: np.ndarray = rho
        self.gamma: float = gamma

    def calc_vpi(self, pi: np.ndarray, r: np.ndarray) -> np.ndarray:
        """
        Computes v^pi = (I_S - gamma P_pi)^{-1} r_pi
        where r is an arbitrary 'reward' vector

        ref: https://vaswanis.github.io/419_983-F23/L4.pdf
        """
        p_pi = np.einsum("xay,xa->xy", self.P, pi)
        r_pi = np.einsum("xa,xa->x", r, pi)
        v_pi = np.linalg.solve(np.eye(self.S) - self.gamma * p_pi, r_pi)
        return v_pi

    def calc_qpi(self, pi: np.ndarray, r: np.ndarray) -> np.ndarray:
        """
        Computes q^pi(s, a) = r(s, a) + gamma * sum_{s'} p(s' | s, a) * v^pi(s')
        """
        v_pi = self.calc_vpi(pi, r)
        q_pi = r + self.gamma * np.einsum("xay,y->xa", self.P, v_pi)
        return q_pi

    def calc_dpi(self, pi: np.ndarray) -> np.ndarray:
        """
        Computes the normalized state occupancy measure
            $d^pi = (1 - gamma) * rho * (I - gamma P_pi^T)^{-1}
        """
        p_pi = np.einsum("xay,xa->xy", self.P, pi)
        d_pi = (1 - self.gamma) * np.linalg.solve(
            np.eye(self.S) - self.gamma * p_pi.T, self.rho
        )
        # for addressing numerical errors
        d_pi /= d_pi.sum()
        return d_pi

    def calc_q_star(self, num_iters=1000) -> np.ndarray:
        q = np.zeros((self.S, self.A))
        for i in range(num_iters):
            q_new = self.r + np.einsum("xay,y->xa", self.gamma * self.P, q.max(axis=1))
            q = q_new.copy()
        return q

    def calc_v_star(self, num_iters=1000) -> np.ndarray:
        v = np.zeros(self.S)
        for i in range(num_iters):
            v_new = self.r + np.einsum("xay,y->xa", self.gamma * self.P, v)
            v_new = v_new.max(axis=1)
            v = v_new.copy()
        return v

    def calc_pi_star(self, num_iters=1000) -> np.ndarray:
        q_star = self.calc_q_star(num_iters=num_iters)
        pi_star = np.zeros((self.S, self.A))
        pi_star[range(self.S), q_star.argmax(1)] = 1
        return pi_star

    def objective(self, pi: np.ndarray):
        return self.calc_vpi(pi, self.r) @ self.rho

    def constraint(self, pi: np.ndarray):
        return self.calc_vpi(pi, self.c) @ self.rho

    def solve_dual(self) -> np.ndarray:
        """
        Dual LP problem a CMDP
        max_{mu} sum_{s, a} mu(s, a) r(s, a) / (1 - gamma)
        s.t. sum_{a} mu(s', a) - gamma sum_{s ,a} P(s' | s, a) mu(s, a) = (1 - gamma) \rho(s') [for all s' in S]
             sum_{s, a} mu(s, a) c(s, a) / (1 - gamma) >= b
             <=>
             -sum_{s, a} mu(s, a) c(s, a) / (1 - gamma) <= -b
             mu(s, a) >= 0

        ref: https://proceedings.mlr.press/v206/gladin23a/gladin23a.pdf
             https://github.com/arushijain94/CoinBettingPolitex/blob/master/Tabular/tabularCMDPEnv.py
        """

        A_eq = np.zeros((self.S, self.S * self.A))
        for s_prime in range(self.S):
            # add a block of  lengh A accross so that (A @ mu)[s'] = \sum_{a} \mu(s', a)
            A_eq[s_prime, self.A * s_prime : self.A * (s_prime + 1)] = 1
            A_eq[s_prime] -= self.gamma * self.P[:, :, s_prime].flatten()
        b_eq = (1 - self.gamma) * self.rho

        A_ub = -self.c.reshape((1, -1)) / (1 - self.gamma)
        b_ub = -self.b

        res = scipy.optimize.linprog(
            -self.r.flatten() / (1 - self.gamma),
            A_eq=A_eq,
            b_eq=b_eq,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=[(0, None)] * self.S * self.A,
        )

        if not res.success:
            print(res.message)
            breakpoint()

        mu_star = res.x.reshape((self.S, self.A))
        pi_star = mu_star / mu_star.sum(axis=1, keepdims=True)
        return pi_star

    def print_performance(self, pi):
        print(f"V^π_r(ρ): {self.calc_vpi(pi, self.r) @ self.rho:0.8f}")
        print(f"V^π_c(ρ)={self.calc_vpi(pi, self.c) @ self.rho:0.8f} >= {self.b}=b")
