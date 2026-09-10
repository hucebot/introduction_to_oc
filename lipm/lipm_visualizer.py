import numpy as np
import matplotlib.pyplot as plt


def lipm_plot(ns, tf, integer_pattern,
              r, rdot, uu, zmp_ref,
              left_hist, right_hist,
              block=True):

    dt = tf / ns

    t_state = np.arange(ns + 1) * dt
    t_ctrl = np.arange(ns) * dt

    left_hist = np.asarray(left_hist)
    right_hist = np.asarray(right_hist)
    zmp_ref = np.asarray(zmp_ref)

    fig = plt.figure(figsize=(14, 8))

    gs = fig.add_gridspec(
        nrows=3,
        ncols=2,
        width_ratios=[2.0, 1.2],
        wspace=0.3
    )

    ax_x = fig.add_subplot(gs[0, 0])
    ax_y = fig.add_subplot(gs[1, 0], sharex=ax_x)
    ax_v = fig.add_subplot(gs[2, 0], sharex=ax_x)

    ax_foot = fig.add_subplot(gs[:, 1])

    # ==================================================
    # X (LEFT COLUMN) — KEEP FOOTSTEP MARKERS
    # ==================================================

    ax_x.plot(t_state, r[0, :], label=r'$r_x$ (CoM)')
    ax_x.plot(t_ctrl, uu[0, :], label=r'$z_x$ (ZMP)')
    ax_x.plot(t_ctrl, zmp_ref[:, 0], label=r'$z_x$ ref')

    for k in range(len(integer_pattern)):

        alpha_l = 1.0
        alpha_r = 1.0

        if integer_pattern[k] == 1:
            alpha_r = 0.15
        elif integer_pattern[k] == 2:
            alpha_l = 0.15

        ax_x.scatter(t_ctrl[k], left_hist[k, 0],
                     marker='s', s=60, color='tab:blue', alpha=alpha_l)

        ax_x.scatter(t_ctrl[k], right_hist[k, 0],
                     marker='s', s=60, color='tab:orange', alpha=alpha_r)

    ax_x.set_ylabel('Position [m]')
    ax_x.grid(True)
    ax_x.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)

    # ==================================================
    # Y (LEFT COLUMN) — KEEP FOOTSTEP MARKERS
    # ==================================================

    ax_y.plot(t_state, r[1, :], label=r'$r_y$ (CoM)')
    ax_y.plot(t_ctrl, uu[1, :], label=r'$z_y$ (ZMP)')
    ax_y.plot(t_ctrl, zmp_ref[:, 1], label=r'$z_y$ ref')

    for k in range(len(integer_pattern)):

        alpha_l = 1.0
        alpha_r = 1.0

        if integer_pattern[k] == 1:
            alpha_r = 0.15
        elif integer_pattern[k] == 2:
            alpha_l = 0.15

        ax_y.scatter(t_ctrl[k], left_hist[k, 1],
                     marker='s', s=60, color='tab:blue', alpha=alpha_l)

        ax_y.scatter(t_ctrl[k], right_hist[k, 1],
                     marker='s', s=60, color='tab:orange', alpha=alpha_r)

    ax_y.set_ylabel('Position [m]')
    ax_y.grid(True)
    ax_y.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)

    # ==================================================
    # VELOCITY
    # ==================================================

    ax_v.plot(t_state, rdot[0, :], label=r'$\dot r_x$')
    ax_v.plot(t_state, rdot[1, :], label=r'$\dot r_y$')

    ax_v.set_xlabel('Time [s]')
    ax_v.set_ylabel('Velocity [m/s]')
    ax_v.grid(True)
    ax_v.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)

    # ==================================================
    # RIGHT PLOT — FOOTSTEPS + CoM + ZMP
    # ==================================================

    # foot trajectories
    ax_foot.plot(left_hist[:, 0], left_hist[:, 1],
                 '--', color='tab:blue', alpha=0.5, label='Left foot')

    ax_foot.plot(right_hist[:, 0], right_hist[:, 1],
                 '--', color='tab:orange', alpha=0.5, label='Right foot')

    # CoM (actual)
    ax_foot.plot(r[0, :], r[1, :],
                 color='green', alpha=1.0, linewidth=2, label='CoM')

    # ZMP (actual)
    ax_foot.plot(uu[0, :], uu[1, :],
                 color='red', alpha=1.0, linewidth=2, label='ZMP')

    # reference ZMP
    ax_foot.plot(zmp_ref[:, 0], zmp_ref[:, 1],
                 'k', linewidth=2, label='ZMP ref')

    # foot contact markers
    for k in range(len(integer_pattern)):

        alpha_l = 1.0
        alpha_r = 1.0

        if integer_pattern[k] == 1:
            alpha_r = 0.15
        elif integer_pattern[k] == 2:
            alpha_l = 0.15

        ax_foot.scatter(left_hist[k, 0], left_hist[k, 1],
                        marker='s', s=100, color='tab:blue', alpha=alpha_l)

        ax_foot.scatter(right_hist[k, 0], right_hist[k, 1],
                        marker='s', s=100, color='tab:orange', alpha=alpha_r)

    # ==================================================
    # CENTERED SQUARE VIEW
    # ==================================================

    pts_x = np.concatenate([
        left_hist[:, 0],
        right_hist[:, 0],
        r[0, :],
        uu[0, :],
        zmp_ref[:, 0]
    ])

    pts_y = np.concatenate([
        left_hist[:, 1],
        right_hist[:, 1],
        r[1, :],
        uu[1, :],
        zmp_ref[:, 1]
    ])

    xc = 0.5 * (np.min(pts_x) + np.max(pts_x))
    yc = 0.5 * (np.min(pts_y) + np.max(pts_y))

    side = max(np.max(pts_x) - np.min(pts_x),
               np.max(pts_y) - np.min(pts_y))

    side = max(side, 0.5) * 1.2

    ax_foot.set_xlim(xc - side / 2, xc + side / 2)
    ax_foot.set_ylim(yc - side / 2, yc + side / 2)

    ax_foot.set_box_aspect(1)
    ax_foot.set_aspect('equal')

    ax_foot.set_xlabel('x [m]')
    ax_foot.set_ylabel('y [m]')
    ax_foot.set_title('Footsteps / CoM / ZMP')
    ax_foot.grid(True)
    ax_foot.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)

    plt.tight_layout(rect=(0.0, 0.0, 0.84, 1.0))
    plt.show(block=block)


class LIPMVisualizer:
    """Efficient online visualizer for LIPM MPC rollouts.

    The class creates artists once, then updates their data in-place for each
    new MPC sample. This avoids repeated re-plotting inside loops.
    """

    def __init__(self, ns, tf, interactive=True, history_size=None):
        self.ns = int(ns)
        self.tf = float(tf)
        self.dt = self.tf / self.ns
        self.interactive = bool(interactive)
        self.history_size = int(history_size) if history_size is not None else self.ns

        self.t_state = np.arange(self.history_size + 1) * self.dt
        self.t_ctrl = np.arange(self.history_size) * self.dt

        self.integer_hist = np.zeros(self.history_size, dtype=int)
        self.r_hist = np.full((2, self.history_size + 1), np.nan)
        self.rdot_hist = np.full((2, self.history_size + 1), np.nan)
        self.uu_hist = np.full((2, self.history_size), np.nan)
        self.zmp_ref_hist = np.full((self.history_size, 2), np.nan)
        self.left_hist = np.full((self.history_size, 2), np.nan)
        self.right_hist = np.full((self.history_size, 2), np.nan)

        self.k = 0
        self.state_count = 0
        self.ctrl_count = 0

        self._setup_figure()

        if self.interactive:
            plt.ion()
            self.fig.show()
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

    def _setup_figure(self):
        self.fig = plt.figure(figsize=(14, 8))
        gs = self.fig.add_gridspec(
            nrows=3,
            ncols=2,
            width_ratios=[2.0, 1.2],
            wspace=0.3,
        )

        self.ax_x = self.fig.add_subplot(gs[0, 0])
        self.ax_y = self.fig.add_subplot(gs[1, 0], sharex=self.ax_x)
        self.ax_v = self.fig.add_subplot(gs[2, 0], sharex=self.ax_x)
        self.ax_foot = self.fig.add_subplot(gs[:, 1])

        # Left column
        (self.line_rx,) = self.ax_x.plot([], [], label=r"$r_x$ (CoM)")
        (self.line_ux,) = self.ax_x.plot([], [], label=r"$z_x$ (ZMP)")
        (self.line_zx_ref,) = self.ax_x.plot([], [], label=r"$z_x$ ref")
        self.scatter_lx = self.ax_x.scatter([], [], marker="s", s=60, color="tab:blue")
        self.scatter_rx = self.ax_x.scatter([], [], marker="s", s=60, color="tab:orange")
        self.ax_x.set_ylabel("Position [m]")
        self.ax_x.set_xlim(0.0, self.tf)
        self.ax_x.grid(True)
        self.ax_x.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)

        (self.line_ry,) = self.ax_y.plot([], [], label=r"$r_y$ (CoM)")
        (self.line_uy,) = self.ax_y.plot([], [], label=r"$z_y$ (ZMP)")
        (self.line_zy_ref,) = self.ax_y.plot([], [], label=r"$z_y$ ref")
        self.scatter_ly = self.ax_y.scatter([], [], marker="s", s=60, color="tab:blue")
        self.scatter_ry = self.ax_y.scatter([], [], marker="s", s=60, color="tab:orange")
        self.ax_y.set_ylabel("Position [m]")
        self.ax_y.set_xlim(0.0, self.tf)
        self.ax_y.grid(True)
        self.ax_y.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)

        (self.line_vx,) = self.ax_v.plot([], [], label=r"$\dot r_x$")
        (self.line_vy,) = self.ax_v.plot([], [], label=r"$\dot r_y$")
        self.ax_v.set_xlabel("Time [s]")
        self.ax_v.set_ylabel("Velocity [m/s]")
        self.ax_v.set_xlim(0.0, self.tf)
        self.ax_v.grid(True)
        self.ax_v.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)

        # Right plot
        (self.line_left_path,) = self.ax_foot.plot([], [], "--", color="tab:blue", alpha=0.5, label="Left foot")
        (self.line_right_path,) = self.ax_foot.plot([], [], "--", color="tab:orange", alpha=0.5, label="Right foot")
        (self.line_com_xy,) = self.ax_foot.plot([], [], color="green", linewidth=2, label="CoM")
        (self.line_zmp_xy,) = self.ax_foot.plot([], [], color="red", linewidth=2, label="ZMP")
        (self.line_zmp_ref_xy,) = self.ax_foot.plot([], [], "k", linewidth=2, label="ZMP ref")
        self.scatter_lxy = self.ax_foot.scatter([], [], marker="s", s=100, color="tab:blue")
        self.scatter_rxy = self.ax_foot.scatter([], [], marker="s", s=100, color="tab:orange")

        self.ax_foot.set_box_aspect(1)
        self.ax_foot.set_aspect("equal")
        self.ax_foot.set_xlabel("x [m]")
        self.ax_foot.set_ylabel("y [m]")
        self.ax_foot.set_title("Footsteps / CoM / ZMP")
        self.ax_foot.grid(True)
        self.ax_foot.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)

        self.fig.tight_layout(rect=(0.0, 0.0, 0.84, 1.0))

    @staticmethod
    def _alpha_masks(integer_pattern):
        integer_pattern = np.asarray(integer_pattern, dtype=int)
        left_alpha = np.where(integer_pattern == 2, 0.15, 1.0)
        right_alpha = np.where(integer_pattern == 1, 0.15, 1.0)
        return left_alpha, right_alpha

    @staticmethod
    def _rgba(rgb, alpha):
        out = np.zeros((len(alpha), 4), dtype=float)
        out[:, 0] = rgb[0]
        out[:, 1] = rgb[1]
        out[:, 2] = rgb[2]
        out[:, 3] = alpha
        return out

    @staticmethod
    def _update_y_limits(ax, y_data, pad_ratio=0.1, min_span=1e-2):
        y_data = np.asarray(y_data)
        y_data = y_data[np.isfinite(y_data)]
        if y_data.size == 0:
            return
        y_min = float(np.min(y_data))
        y_max = float(np.max(y_data))
        span = max(y_max - y_min, min_span)
        pad = span * pad_ratio
        ax.set_ylim(y_min - pad, y_max + pad)

    def _update_foot_limits(self, x_data, y_data):
        x_data = np.asarray(x_data)
        y_data = np.asarray(y_data)
        mask = np.isfinite(x_data) & np.isfinite(y_data)
        if not np.any(mask):
            return

        x_data = x_data[mask]
        y_data = y_data[mask]
        xc = 0.5 * (np.min(x_data) + np.max(x_data))
        yc = 0.5 * (np.min(y_data) + np.max(y_data))
        side = max(np.max(x_data) - np.min(x_data), np.max(y_data) - np.min(y_data))
        side = max(float(side), 0.5) * 1.2
        self.ax_foot.set_xlim(xc - side / 2.0, xc + side / 2.0)
        self.ax_foot.set_ylim(yc - side / 2.0, yc + side / 2.0)

    def update_mpc(self, integer_phase, r_k, rdot_k, uu_k,
                   zmp_ref_k, left_k, right_k, redraw=True):
        """Append one MPC sample and update all artists efficiently."""
        if self.k < self.history_size:
            idx = self.k
            self.k += 1
            self.ctrl_count = max(self.ctrl_count, idx + 1)
            self.state_count = max(self.state_count, idx + 1)
        else:
            self.integer_hist[:-1] = self.integer_hist[1:]
            self.uu_hist[:, :-1] = self.uu_hist[:, 1:]
            self.zmp_ref_hist[:-1, :] = self.zmp_ref_hist[1:, :]
            self.left_hist[:-1, :] = self.left_hist[1:, :]
            self.right_hist[:-1, :] = self.right_hist[1:, :]
            self.r_hist[:, :-1] = self.r_hist[:, 1:]
            self.rdot_hist[:, :-1] = self.rdot_hist[:, 1:]
            idx = self.history_size - 1
            self.ctrl_count = self.history_size
            self.state_count = self.history_size

        self.integer_hist[idx] = int(integer_phase)
        self.r_hist[:, idx] = np.asarray(r_k).reshape(2,)
        self.rdot_hist[:, idx] = np.asarray(rdot_k).reshape(2,)
        self.uu_hist[:, idx] = np.asarray(uu_k).reshape(2,)
        self.zmp_ref_hist[idx, :] = np.asarray(zmp_ref_k).reshape(2,)
        self.left_hist[idx, :] = np.asarray(left_k).reshape(2,)
        self.right_hist[idx, :] = np.asarray(right_k).reshape(2,)

        # Keep one additional state point for time-series continuity.
        state_idx = min(idx + 1, self.history_size)
        self.r_hist[:, state_idx] = self.r_hist[:, idx]
        self.rdot_hist[:, state_idx] = self.rdot_hist[:, idx]
        self.state_count = min(max(self.state_count, state_idx + 1), self.history_size + 1)

        n_ctrl = self.ctrl_count
        n_state = self.state_count

        t_ctrl = self.t_ctrl[:n_ctrl]
        t_state = self.t_state[:n_state]

        self.line_rx.set_data(t_state, self.r_hist[0, :n_state])
        self.line_ux.set_data(t_ctrl, self.uu_hist[0, :n_ctrl])
        self.line_zx_ref.set_data(t_ctrl, self.zmp_ref_hist[:n_ctrl, 0])

        self.line_ry.set_data(t_state, self.r_hist[1, :n_state])
        self.line_uy.set_data(t_ctrl, self.uu_hist[1, :n_ctrl])
        self.line_zy_ref.set_data(t_ctrl, self.zmp_ref_hist[:n_ctrl, 1])

        self.line_vx.set_data(t_state, self.rdot_hist[0, :n_state])
        self.line_vy.set_data(t_state, self.rdot_hist[1, :n_state])

        self.line_left_path.set_data(self.left_hist[:n_ctrl, 0], self.left_hist[:n_ctrl, 1])
        self.line_right_path.set_data(self.right_hist[:n_ctrl, 0], self.right_hist[:n_ctrl, 1])
        self.line_com_xy.set_data(self.r_hist[0, :n_state], self.r_hist[1, :n_state])
        self.line_zmp_xy.set_data(self.uu_hist[0, :n_ctrl], self.uu_hist[1, :n_ctrl])
        self.line_zmp_ref_xy.set_data(self.zmp_ref_hist[:n_ctrl, 0], self.zmp_ref_hist[:n_ctrl, 1])

        phase = self.integer_hist[:n_ctrl]
        left_alpha, right_alpha = self._alpha_masks(phase)

        self.scatter_lx.set_offsets(np.column_stack((t_ctrl, self.left_hist[:n_ctrl, 0])))
        self.scatter_rx.set_offsets(np.column_stack((t_ctrl, self.right_hist[:n_ctrl, 0])))
        self.scatter_ly.set_offsets(np.column_stack((t_ctrl, self.left_hist[:n_ctrl, 1])))
        self.scatter_ry.set_offsets(np.column_stack((t_ctrl, self.right_hist[:n_ctrl, 1])))
        self.scatter_lxy.set_offsets(self.left_hist[:n_ctrl, :])
        self.scatter_rxy.set_offsets(self.right_hist[:n_ctrl, :])

        self.scatter_lx.set_facecolors(self._rgba((0.121, 0.466, 0.705), left_alpha))
        self.scatter_rx.set_facecolors(self._rgba((1.0, 0.498, 0.054), right_alpha))
        self.scatter_ly.set_facecolors(self._rgba((0.121, 0.466, 0.705), left_alpha))
        self.scatter_ry.set_facecolors(self._rgba((1.0, 0.498, 0.054), right_alpha))
        self.scatter_lxy.set_facecolors(self._rgba((0.121, 0.466, 0.705), left_alpha))
        self.scatter_rxy.set_facecolors(self._rgba((1.0, 0.498, 0.054), right_alpha))

        self._update_y_limits(
            self.ax_x,
            np.concatenate((self.r_hist[0, :n_state], self.uu_hist[0, :n_ctrl], self.zmp_ref_hist[:n_ctrl, 0])),
        )
        self._update_y_limits(
            self.ax_y,
            np.concatenate((self.r_hist[1, :n_state], self.uu_hist[1, :n_ctrl], self.zmp_ref_hist[:n_ctrl, 1])),
        )
        self._update_y_limits(
            self.ax_v,
            np.concatenate((self.rdot_hist[0, :n_state], self.rdot_hist[1, :n_state])),
        )

        foot_x = np.concatenate((
            self.left_hist[:n_ctrl, 0],
            self.right_hist[:n_ctrl, 0],
            self.r_hist[0, :n_state],
            self.uu_hist[0, :n_ctrl],
            self.zmp_ref_hist[:n_ctrl, 0],
        ))
        foot_y = np.concatenate((
            self.left_hist[:n_ctrl, 1],
            self.right_hist[:n_ctrl, 1],
            self.r_hist[1, :n_state],
            self.uu_hist[1, :n_ctrl],
            self.zmp_ref_hist[:n_ctrl, 1],
        ))
        self._update_foot_limits(foot_x, foot_y)

        if redraw:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

    def plot(self, integer_pattern, r, rdot, uu, zmp_ref, left_hist, right_hist, block=True):
        """One-shot plotting entrypoint for compatibility with offline workflows."""
        return lipm_plot(
            ns=self.ns,
            tf=self.tf,
            integer_pattern=integer_pattern,
            r=r,
            rdot=rdot,
            uu=uu,
            zmp_ref=zmp_ref,
            left_hist=left_hist,
            right_hist=right_hist,
            block=block,
        )

    def close(self):
        plt.close(self.fig)
