from environments.env import Env
import numpy as np
import matplotlib.pyplot as plt


class ConstrainedCliffWorld(Env):
    """
    Simplified CliffWorld
    -------------------         4 is the goal state
    | 4 | 9 | 14 | 19 |         20 is terminal state reached only via the state 4
    -------------------         3, 2, 1 are chasms
    | 3 | 8 | 13 | 18 |         0 is the start state
    -------------------
    | 2 | 7 | 12 | 17 |         all transitions are determinitic
    -------------------         Actions: 0=down, 1=up, 2=left, 3=right
    | 1 | 6 | 11 | 16 |
    ------------------------    rewards are all zeros, except for going into the goal state is +1
    | 0 | 5 | 10 | 15 | 20 |    costs are all zeros, except for going into the chasms is +1
    ------------------------
    """

    def __init__(self, b: float = -0.17, gamma: float = 0.9):
        P = np.zeros((21, 4, 21))
        for state_idx in range(21):
            for action_idx in range(4):
                # if state_idx in [1, 2, 3]:  # chasms: reset to start state 0
                #     new_state_idx = 0
                if state_idx == 4:  # goal state: agent always goes to 20
                    new_state_idx = 20
                elif state_idx == 20:  # terminal state
                    new_state_idx = 20
                else:  # move according to the deterministic dynamics
                    x_new = x_old = state_idx // 5
                    y_new = y_old = state_idx % 5
                    if action_idx == 0:  # Down
                        y_new = np.clip(y_old - 1, 0, 4)
                    elif action_idx == 1:  # Up
                        y_new = np.clip(y_old + 1, 0, 4)
                    elif action_idx == 2:  # Left
                        x_new = np.clip(x_old - 1, 0, 3)
                    elif action_idx == 3:  # Right
                        x_new = np.clip(x_old + 1, 0, 3)
                    new_state_idx = 5 * x_new + y_new

                P[state_idx, action_idx, new_state_idx] = 1

        c = np.zeros((21, 4))
        c[1, :] = -1
        c[2, :] = -1
        c[3, :] = -1
        # c[0, 1] = -15
        # c[1, :] = -15
        # c[2, :] = -15
        # c[3, :] = -15
        # c[3, 1] = 0
        r = np.zeros((21, 4))
        r[4, :] = +1  # positive reward for finding the goal terminal state

        # ML: original cliff world environment rewards
        # r = np.zeros((21, 4))
        # r[1, :] = r[2, :] = r[3, :] = -100  # negative reward for falling into chasms
        # r[4, :] = +1  # positive reward for finding the goal terminal state

        rho = np.ones(21) / 21

        # ML: original cliff world environment init state distribution
        # rho = np.zeros(21)
        # rho[0] = 1

        super().__init__(P, r, c, b, rho, gamma)

    def plot_pi(self, pi, filename="cliffworld.png"):
        plt.figure(figsize=(4, 4), dpi=200)

        xlim = 4
        ylim = 5
        x_center = np.arange(xlim) + 0.5
        y_center = np.arange(ylim) + 0.5

        for x in x_center:
            for y in y_center:
                if int((x - 0.5) * ylim + (y - 0.5)) == 0:
                    plt.scatter(
                        x,
                        y,
                        s=1500,
                        color="tab:blue",
                        marker="s",
                        label="start",
                        alpha=0.5,
                    )
                elif int((x - 0.5) * ylim + (y - 0.5)) in [1, 2, 3]:
                    plt.scatter(
                        x,
                        y,
                        s=1500,
                        color="tab:red",
                        marker="x",
                        label="chasm",
                        alpha=0.5,
                    )
                elif int((x - 0.5) * ylim + (y - 0.5)) == 4:
                    plt.scatter(
                        x,
                        y,
                        s=1500,
                        color="tab:green",
                        marker="s",
                        label="end",
                        alpha=0.5,
                    )
                else:
                    plt.scatter(x, y, s=0.1, color="purple")

        for x in range(xlim + 1):
            plt.plot((x, x), (0, ylim), linewidth=0.5, color="black")

        for y in range(ylim + 1):
            plt.plot((0, xlim), (y, y), linewidth=0.5, color="black")

        for s_x in range(xlim):
            for s_y in range(ylim):
                x_center = s_x + 0.5
                y_center = s_y + 0.5
                s = s_x * ylim + s_y
                diff = pi[s] / 2

                plt.quiver(
                    x_center,
                    y_center,
                    0,
                    -diff[0],
                    color="tab:orange",
                    width=0.013,
                    headwidth=2,
                    headlength=4,
                    scale=1,
                    scale_units="xy",
                    linewidth=1,
                )
                # up
                plt.quiver(
                    x_center,
                    y_center,
                    0,
                    +diff[1],
                    color="tab:blue",
                    width=0.013,
                    headwidth=2,
                    headlength=4,
                    scale=1,
                    scale_units="xy",
                    linewidth=1,
                )
                # left
                plt.quiver(
                    x_center,
                    y_center,
                    -diff[2],
                    0,
                    color="tab:red",
                    width=0.013,
                    headwidth=2,
                    headlength=4,
                    scale=1,
                    scale_units="xy",
                    linewidth=1,
                )
                # right
                plt.quiver(
                    x_center,
                    y_center,
                    +diff[3],
                    0,
                    color="tab:green",
                    width=0.013,
                    headwidth=2,
                    headlength=4,
                    scale=1,
                    scale_units="xy",
                    linewidth=1,
                )
        plt.title(f"Constrained CliffWorld (b={self.b})")
        plt.savefig(filename, bbox_inches="tight")
