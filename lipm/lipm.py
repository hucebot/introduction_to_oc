import numpy as np
import riccati
from lipm_visualizer import LIPMVisualizer, lipm_plot
import time
import joy as joystick

class lipm_problem():
    def __init__(self, ns:int, tf:float, h:float=83, Q:float=1e-1, Qfin:float=1e6, R:float=1e1):
        # States
        self.nr = 2  # CoM position size
        self.nrdot = 2  # CoM velocity size
        self.nx = self.nr + self.nrdot  # State dimension

        nz = 2  # ZMP position
        self.nu = nz  # Control dimension

        self.ns = ns
        self.dt = tf/ns

        self.w = np.sqrt(9.81 / h)

        # Cost Function Params
        self.Q = Q
        self.Qfin = Qfin
        self.R = R

        self.left_hist = []
        self.right_hist = []
        self.zmp_hist = []

    @staticmethod
    def _rotz(theta):
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([[c, -s], [s, c]])

    def dynamics(self):
        A = np.zeros((self.nx, self.nx))
        A[0:2, 2:4] = np.eye(2)
        A[2:4, 0:2] = self.w**2 * np.eye(2)

        B = np.zeros((self.nx, self.nu))
        B[2:4, 0:2] = -self.w**2 * np.eye(2)
        return A, B


    def running_cost_gradient(self, zref, vref):
        """
        Gradient running cost for single stage
        """
        g = np.zeros((self.nx + self.nu, 1))
        g[2:4, 0] = -self.Q * vref
        g[4:6, 0] = -self.R * zref
        return g
    
    def final_cost_gradient(self, vref):
        """
        Gradient final cost
        """
        g = np.zeros((self.nx, 1))
        g[2:4, 0] = -self.Qfin * vref
        return g


    def Euler(self, A, B, dt):
        """
        Euler integration for a single stage (node)
        """
        E = np.zeros((self.nx, 2 * self.nx + self.nu))
        E[:, 0:self.nx] = np.eye(self.nx) + dt * A
        E[:, self.nx:self.nx + self.nu] = dt * B
        E[:, self.nx + self.nu:] = -np.eye(self.nx)
        return E

    def init(self, step_pattern:list, v_des, 
             left_init, right_init,
             max_step_x=np.inf, max_step_y=np.inf,
             max_feet_distance=np.inf):
        Qk = np.zeros((self.nx, self.nx))
        Qk[self.nr:self.nr + self.nr, self.nr:self.nr + self.nr] = self.Q * np.eye(self.nrdot)
        Rk = np.zeros((self.nu, self.nu))
        Rk[0:self.nu, 0:self.nu] = self.R * np.eye(self.nu)
        Qf = np.zeros((self.nx, self.nx))
        Qf[self.nr:self.nr + self.nr, self.nr:self.nr + self.nr] = self.Qfin * np.eye(self.nrdot)
        
        self.left_hist, self.right_hist, self.zmp_hist = self.generate_gait_intervals(
            step_pattern=step_pattern,
            velocity=v_des,
            dt=self.dt,
            left_init=left_init,
            right_init=right_init,
            max_step_x=max_step_x,
            max_step_y=max_step_y,
            max_feet_distance=max_feet_distance,
        )

        Q = []
        R = []
        S = []
        grad = []
        A = []
        B = []
        b = []
        for k in range(0, self.ns):
            Q.append(Qk)
            R.append(Rk)
            S.append(np.zeros((self.nu, self.nx)))
            grad.append(self.running_cost_gradient(self.zmp_hist[k], v_des[0:2]))
            Ak, Bk = self.dynamics()
            E = self.Euler(Ak, Bk, self.dt)
            A.append(E[:, 0:self.nx])
            B.append(E[:, self.nx:self.nx + self.nu])
            b.append(np.zeros((self.nx, 1)))

        gradf = self.final_cost_gradient(v_des[0:2])

        return Q, Qf, R, S, grad, gradf, A, B, b
    
    def update(self, step_pattern:list, v_des, 
             left_init, right_init,
             max_step_x=np.inf, max_step_y=np.inf,
             max_feet_distance=np.inf):
        
        self.left_hist, self.right_hist, self.zmp_hist = self.generate_gait_intervals(
            step_pattern=step_pattern,
            velocity=v_des,
            dt=self.dt,
            left_init=left_init,
            right_init=right_init,
            max_step_x=max_step_x,
            max_step_y=max_step_y,
            max_feet_distance=max_feet_distance,
        )

        grad = []
        b = []
        for k in range(0, self.ns):
            grad.append(self.running_cost_gradient(self.zmp_hist[k], v_des[0:2]))
            b.append(np.zeros((self.nx, 1)))

        gradf = self.final_cost_gradient(v_des[0:2])

        return grad, gradf, b
        

    def generate_gait_intervals(self, step_pattern, velocity, dt,
                                left_init, right_init,
                                max_step_x=np.inf, max_step_y=np.inf,
                                max_feet_distance=np.inf):
        DS = 0
        LS = 1
        RS = 2

        vx, vy, omega = velocity

        left = np.array(left_init, dtype=float).reshape(3,)
        right = np.array(right_init, dtype=float).reshape(3,)

        max_step_x = float(max_step_x)
        max_step_y = float(max_step_y)
        max_feet_distance = float(max_feet_distance)
        tol = 1e-9
        pass_margin = 1e-4

        # Nominal relative placement from the provided initial poses.
        rel_r_in_l0 = self._rotz(-left[2]) @ (right[:2] - left[:2])
        rel_l_in_r0 = self._rotz(-right[2]) @ (left[:2] - right[:2])

        left_hist = []
        right_hist = []
        zmp_hist = []

        k = 0
        while k < len(step_pattern):
            phase = step_pattern[k]
            # ------------------------------------------------
            # DOUBLE SUPPORT
            # ------------------------------------------------
            if phase == DS:
                left_hist.append(left.copy())
                right_hist.append(right.copy())

                zmp_hist.append(0.5 * (left[:2] + right[:2]))

                k += 1
                continue

            # ------------------------------------------------
            # LEFT SUPPORT
            # ------------------------------------------------
            if phase == LS:
                start = k
                while k < len(step_pattern) and step_pattern[k] == LS:
                    k += 1

                n = k - start

                right_start = right.copy()

                target = right.copy()
                desired_step_local = np.array([vx * n * dt, vy * n * dt])
                limited_step_local = np.array([
                    np.clip(desired_step_local[0], -max_step_x, max_step_x),
                    np.clip(desired_step_local[1], -max_step_y, max_step_y),
                ])
                limited_rel_local = rel_r_in_l0 + limited_step_local
                if abs(vx) > tol:
                    desired_sign = np.sign(vx)
                    min_pass_x = max(abs(limited_step_local[0]), pass_margin)
                    if limited_rel_local[0] * desired_sign >= 0.0:
                        limited_rel_local[0] = desired_sign * max(abs(limited_rel_local[0]), min_pass_x)

                # Keep right foot on its nominal side relative to left foot.
                side_sign = np.sign(rel_r_in_l0[1])
                if abs(side_sign) > tol:
                    limited_rel_local[1] = side_sign * max(abs(limited_rel_local[1]), pass_margin)

                # Keep feet within a maximum separation radius.
                rel_norm = np.linalg.norm(limited_rel_local)
                if rel_norm > max_feet_distance > 0.0:
                    limited_rel_local = (max_feet_distance / rel_norm) * limited_rel_local

                target[:2] = left[:2] + self._rotz(left[2]) @ limited_rel_local
                target[2] += omega * n * dt

                for i in range(n):
                    alpha = (i + 1) / n

                    right_interp = ((1 - alpha) * right_start + alpha * target)

                    left_hist.append(left.copy())
                    right_hist.append(right_interp)

                    zmp_hist.append(left[:2].copy())

                right = target
                continue

            # ------------------------------------------------
            # RIGHT SUPPORT
            # ------------------------------------------------
            if phase == RS:
                start = k

                while k < len(step_pattern) and step_pattern[k] == RS:
                    k += 1

                n = k - start

                left_start = left.copy()

                target = left.copy()
                desired_step_local = np.array([vx * n * dt, vy * n * dt])
                limited_step_local = np.array([
                    np.clip(desired_step_local[0], -max_step_x, max_step_x),
                    np.clip(desired_step_local[1], -max_step_y, max_step_y),
                ])
                limited_rel_local = rel_l_in_r0 + limited_step_local
                if abs(vx) > tol:
                    desired_sign = np.sign(vx)
                    min_pass_x = max(abs(limited_step_local[0]), pass_margin)
                    if limited_rel_local[0] * desired_sign >= 0.0:
                        limited_rel_local[0] = desired_sign * max(abs(limited_rel_local[0]), min_pass_x)

                # Keep left foot on its nominal side relative to right foot.
                side_sign = np.sign(rel_l_in_r0[1])
                if abs(side_sign) > tol:
                    limited_rel_local[1] = side_sign * max(abs(limited_rel_local[1]), pass_margin)

                # Keep feet within a maximum separation radius.
                rel_norm = np.linalg.norm(limited_rel_local)
                if rel_norm > max_feet_distance > 0.0:
                    limited_rel_local = (max_feet_distance / rel_norm) * limited_rel_local

                target[:2] = right[:2] + self._rotz(right[2]) @ limited_rel_local
                target[2] += omega * n * dt

                for i in range(n):
                    alpha = (i + 1) / n

                    left_interp = ((1 - alpha) * left_start + alpha * target)

                    left_hist.append(left_interp)
                    right_hist.append(right.copy())

                    zmp_hist.append(right[:2].copy())

                left = target
                continue

        return left_hist, right_hist, zmp_hist

"""
Trajectory Optimzation test
"""
def main(mpc=False):
    ns = 40
    tf = 3.

    integer_pattern = []
    for i in range(ns):
        if i >= 10 and i < 20:
            integer_pattern.append(1)
        elif i >= 20 and i < 30:
            integer_pattern.append(2)
        elif i >= 30 or i < 10:
            integer_pattern.append(0)

    lipm = lipm_problem(ns=ns, tf=tf, h=0.83)
    x0 = np.zeros((lipm.nx, 1))

    foot_distance=0.2
    left_init =  [0.0, +foot_distance / 2.0, 0.0]
    right_init = [0.0, -foot_distance / 2.0, 0.0]

    solver = riccati.Riccati()

    k = 0

    Q = None
    Qf = None
    R = None
    S = None
    A = None
    B = None

    if not mpc:
        vd = np.array([0.1, 0., 0.])
        Q, Qf, R, S, grad, gradf, A, B, b = lipm.init(integer_pattern, vd, 
                                                    left_init=left_init, right_init=right_init,
                                                    max_step_x=0.3, max_step_y=0.3)
        x,u,f = solver.solve(x0=x0,Q=Q, Qf=Qf, R=R, S=S, grad=grad, gradf=gradf, A=A, B=B, b=b)

        # Extract r and rdot
        r = np.zeros((2, ns + 1))
        rdot = np.zeros((2, ns + 1))
        uu = np.zeros((2, ns))
        for i in range(ns + 1):
            r[:, i] = np.array(x[i][0:lipm.nr]).flatten()
            rdot[:, i] = np.array(x[i][lipm.nr:lipm.nx]).flatten()
            if i < ns:
                uu[:, i] = np.array(u[i][:]).flatten()

        lipm_plot(ns=ns, tf=tf, integer_pattern=integer_pattern, r=r, rdot=rdot, uu=uu, zmp_ref=lipm.zmp_hist, left_hist=lipm.left_hist, right_hist=lipm.right_hist)
    else:
        viz = LIPMVisualizer(ns=ns, tf=tf, interactive=True, history_size=None)
        joy = joystick.Joystick()
        joy.start()
        while True:
            integer_pattern[30:] = integer_pattern[10:20] # copy the left support phase to the right support phase for continuous walking 

            vx, vy, wz = joy.get(alpha_lin=0.5, alpha_ang=1.)

            vd = np.array([vx, vy, 0.])
            #print(f"vd = {vd}")
            if Q is None:
                Q, Qf, R, S, grad, gradf, A, B, b = lipm.init(integer_pattern, vd, 
                                                    left_init=left_init, right_init=right_init,
                                                    max_step_x=0.3, max_step_y=0.3)
            else:
                grad, gradf, b = lipm.update(integer_pattern, vd, 
                                                    left_init=left_init, right_init=right_init,
                                                    max_step_x=0.3, max_step_y=0.3)
            
            x, u, f = solver.solve(x0, Q=Q, Qf=Qf, R=R, S=S, grad=grad, gradf=gradf, A=A, B=B, b=b)

            # Extract r and rdot
            r = np.zeros((2, ns + 1))
            rdot = np.zeros((2, ns + 1))
            uu = np.zeros((2, ns))
            for i in range(ns + 1):
                r[:, i] = np.array(x[i][0:lipm.nr]).flatten()
                rdot[:, i] = np.array(x[i][lipm.nr:lipm.nx]).flatten()
                if i < ns:
                    uu[:, i] = np.array(u[i][:]).flatten()

            viz.update_mpc(integer_phase=integer_pattern[0], 
                           r_k=r[:,0], rdot_k=rdot[:,0], uu_k=uu[:,0], 
                           zmp_ref_k=lipm.zmp_hist[0], 
                           left_k=lipm.left_hist[0][0:2], 
                           right_k=lipm.right_hist[0][0:2], redraw=True)

            x0 = x[1]
            if k < 10:
                integer_pattern.pop(0)
                integer_pattern.append(integer_pattern[10-k])
                k += 1
            else:
                integer_pattern = np.roll(integer_pattern, -1)
            left_init = lipm.left_hist[1]
            right_init = lipm.right_hist[1]

            #time.sleep(0.1)



if __name__ == "__main__":
    main(mpc=True)
