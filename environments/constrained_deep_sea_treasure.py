from environments.env import Env
import numpy as np
import matplotlib.pyplot as plt


class ConstrainedDeepSeaTreasure(Env):
    """
    # Deep Sea Treasure
    ------------------------    20 is the goal state
    | 4 | 9 | 14 | 19 | 24 |    0, 5, 10, 15, 20 are terminal states
    ------------------------    4 is the start state
    | 3 | 8 | 13 | 18 | 23 |
    ------------------------
    | 2 | 7 | 12 | 17 | 22 |    all transitions are deterministic
    ------------------------    Actions: 0=left, 1=right
    | 1 | 6 | 11 | 16 | 21 |
    ------------------------    rewards are all -0.01/5 except at state 20
    | 0 | 5 | 10 | 15 | 20 |    reward for going into the goal state is +1
    ------------------------    costs are all 0, except for entering state 17 where it is +1
    """

    def __init__(self, b: float = -0.1, gamma: float = 0.9):
        terminal_states = [0, 5, 10, 15, 20]
        P = np.zeros((25, 2, 25))
        for state_idx in range(25):
            for action_idx in range(2):
                if state_idx in terminal_states:  # terminal states
                    new_state_idx = state_idx
                else:  # move according to the deterministic dynamics
                    x_new = x_old = state_idx // 5
                    y_new = y_old = state_idx % 5
                    if action_idx == 0:  # left
                        x_new = np.clip(x_old - 1, 0, 4)
                    elif action_idx == 1:  # right
                        x_new = np.clip(x_old + 1, 0, 4)
                    y_new = y_old - 1
                    new_state_idx = 5 * x_new + y_new
                P[state_idx, action_idx, new_state_idx] = 1

        c = np.zeros((25, 2))
        # c[1, 0] = c[6, 0] = -100  # positive cost for reaching bomb
        # c[23, 0] = c[13, 1] = -100  # positive cost for reaching bomb
        c[1, 0] = c[6, 0] = -2  # positive cost for reaching bomb
        c[23, 0] = c[13, 1] = -2  # positive cost for reaching bomb

        r = (-0.01 / 5) * np.ones((25, 2))
        r[16, 1] = r[21, 1] = +1  # positive reward for finding the goal terminal state
        for s in terminal_states:
            r[s, :] = 0
        rho = np.ones(25) / 25

        super().__init__(P, r, c, b, rho, gamma)

    def plot_pi(self, pi, filename="deepseatreasure.png"):
        plt.figure(figsize=(4, 4), dpi=200)
        color_list = ["tab:orange", "tab:blue", "tab:red", "tab:green"]
        x_diff = [0.3, 0.7, 0.5, 0.5]
        y_diff = [0.5, 0.5, 0.3, 0.7]

        xlim = 5
        ylim = 5
        # make grid
        for x in range(xlim + 1):
            plt.plot((x, x), (0, ylim), linewidth=0.5, color="black")
        for y in range(ylim + 1):
            plt.plot((0, xlim), (y, y), linewidth=0.5, color="black")

        # show the transition dynamics
        # for state_idx in range(xlim * ylim):
        #     x_old, y_old = state_idx // ylim, state_idx % ylim
        #     for action_idx in range(2):
        #         if np.sum(self.P[state_idx, action_idx]) == 0:
        #             continue
        #         next_state_idx = self.P[state_idx, action_idx].argmax()
        #         x_new, y_new = next_state_idx // ylim, next_state_idx % ylim
        #         plt.quiver(
        #             x_old + x_diff[action_idx],
        #             y_old + y_diff[action_idx],
        #             x_new - x_old,
        #             y_new - y_old,
        #             color=color_list[action_idx],
        #             width=0.007,
        #             headwidth=2,
        #             headlength=4,
        #             scale=1,
        #             scale_units="xy",
        #             linewidth=0.1,
        #         )

        # show the dynamics of policy
        for s_x in range(xlim):
            for s_y in range(ylim):
                state_idx = s_x * ylim + s_y
                for action_idx, action_prob in enumerate(pi[state_idx]):
                    x_old, y_old = state_idx // ylim, state_idx % ylim

                    if np.sum(self.P[state_idx, action_idx]) == 0:
                        continue
                    next_state_idx = self.P[state_idx, action_idx].argmax()

                    x_new, y_new = next_state_idx // ylim, next_state_idx % ylim
                    plt.quiver(
                        x_old + x_diff[action_idx],
                        y_old + y_diff[action_idx],
                        (x_new - x_old) * action_prob,
                        (y_new - y_old) * action_prob,
                        color=color_list[action_idx],
                        width=0.007,
                        headwidth=2,
                        headlength=4,
                        scale=1,
                        scale_units="xy",
                        linewidth=0.1,
                    )

        # show the state indices
        for state_idx in range(25):
            x = state_idx // ylim
            y = state_idx % ylim
            plt.text(
                x + 0.3, y + 0.3, str(state_idx), fontsize="large", fontweight="bold"
            )

        plt.scatter(
            4.5,
            0.5,
            s=1500,
            color="tab:green",
            marker="s",
            label="treasure",
            alpha=0.5,
        )
        plt.scatter(
            3.5,
            2.5,
            s=1500,
            color="tab:red",
            marker="s",
            label="fail",
            alpha=0.5,
        )
        plt.scatter(
            0.5,
            0.5,
            s=1500,
            color="tab:red",
            marker="s",
            label="fail",
            alpha=0.5,
        )

        plt.title(f"Constrained Deep Sea Treasure (b={self.b})")
        plt.savefig(filename, bbox_inches="tight")
