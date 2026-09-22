import numpy as np
import gymnasium as gym
from gymnasium import spaces

from hill_cartpole_physics import make_hill, rk4_step, critical_force

class HillCartPoleEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        render_mode=None,
        m_c=1.0,
        m_p=0.1,
        pole_length=1.0,
        max_force=10.0,
        force_ratio=None,
        hill_amplitude=1.0,
        hill_freq=1.0,
        theta_threshold_deg=12.0,
        max_episode_steps=1000,
        dt=0.02,
        continuous=True,
        sparse_reward=True,
        shaping_scale=5.0,
    ):
        super().__init__()
        self.m_c, self.m_p = m_c, m_p
        self.l = pole_length / 2.0
        self.I = (1 / 3) * m_p * self.l**2
        self.params = dict(m_c=m_c, m_p=m_p, l=self.l, I=self.I)

        self.critical_force = critical_force(m_c, m_p, hill_amplitude, hill_freq)
        if force_ratio is not None:
            max_force = force_ratio * self.critical_force
        self.max_force = max_force
        self.force_ratio_used = max_force / self.critical_force
        self.dt = dt
        self.max_episode_steps = max_episode_steps
        self.continuous = continuous
        self.sparse_reward = sparse_reward
        self.shaping_scale = shaping_scale

        self.h, self.hprime, self.hprime2 = make_hill(hill_amplitude, hill_freq)
        self.x_start = 0.0
        self.x_goal = np.pi / hill_freq
        self.x_fail_limit = -2 * self.x_goal

        self.theta_threshold = np.deg2rad(theta_threshold_deg)

        high = np.array([2 * abs(self.x_goal) + 1.0, np.pi, 10.0, 20.0],
                         dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        if continuous:
            self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(3)

        self.render_mode = render_mode
        self.state = None
        self.steps = 0

    def _get_obs(self):
        return np.array(self.state, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        theta0 = self.np_random.uniform(-0.05, 0.05)
        self.state = np.array([self.x_start, theta0, 0.0, 0.0])
        self.prev_x = self.x_start
        self.steps = 0
        return self._get_obs(), {}

    def step(self, action):
        if self.continuous:
            F = float(np.clip(action[0], -1.0, 1.0)) * self.max_force
        else:
            F = {0: -self.max_force, 1: 0.0, 2: self.max_force}[int(action)]

        self.state = rk4_step(self.state, F, self.hprime, self.hprime2,
                               self.params, self.dt)

        self.state[1] = (self.state[1] + np.pi) % (2 * np.pi) - np.pi
        x, theta, xdot, thetadot = self.state
        self.steps += 1

        pole_fell = abs(theta) > self.theta_threshold
        fell_off = x < self.x_fail_limit
        reached_goal = (x >= self.x_goal) and not pole_fell

        terminated = bool(pole_fell or fell_off or reached_goal)
        truncated = self.steps >= self.max_episode_steps

        if self.sparse_reward:
            if reached_goal:
                reward = 100.0
            elif pole_fell or fell_off:
                reward = -100.0
            else:
                reward = -1.0
        else:
            SHAPING_SCALE = self.shaping_scale
            progress = SHAPING_SCALE * (x - self.prev_x) / self.x_goal
            reward = progress - 0.1 * abs(theta)
            self.prev_x = x
            if reached_goal:
                reward += 100.0
            if pole_fell or fell_off:
                reward -= 100.0

        info = {"reached_goal": reached_goal, "pole_fell": pole_fell}
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        x, theta, _, _ = self.state
        xs = np.linspace(self.x_fail_limit - 1, self.x_goal + 1, 400)
        ys = self.h(xs)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(xs, ys, color="saddlebrown", lw=2)
        ax.axvline(self.x_goal, color="green", ls="--", lw=1, label="cilj")

        cart_w, cart_h = 0.3, 0.15
        cart_y = self.h(x)
        ax.add_patch(plt.Rectangle((x - cart_w / 2, cart_y), cart_w, cart_h,
                                    color="steelblue"))
        pivot = (x, cart_y + cart_h)
        pole_len = 2 * self.l
        tip = (pivot[0] + pole_len * np.sin(theta),
               pivot[1] + pole_len * np.cos(theta))
        ax.plot([pivot[0], tip[0]], [pivot[1], tip[1]], color="black", lw=3)
        ax.plot(*tip, "o", color="red", markersize=6)

        ax.set_xlim(self.x_fail_limit - 1, self.x_goal + 1)
        ax.set_ylim(min(ys) - 1, max(ys) + 2 * self.l + 1)
        ax.set_aspect("equal")
        ax.legend(loc="upper right")

        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)
        return frame

    def set_theta_threshold(self, deg):
        self.theta_threshold = np.deg2rad(deg)

    def close(self):
        pass
