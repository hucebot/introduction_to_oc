import numpy as np
from collections import deque

class Riccati():
    def __init__(self):
        pass

    def riccati_backward(self, Q, S, R, q, r, P, p, A, B, b):
        Q_ = Q + A.T @ P @ A
        R_ = R + B.T @ P @ B
        S_ = S + B.T @ P @ A

        l = P @ b + p
        q_ = q + A.T @ l
        r_ = r + B.T @ l

        R_i = np.linalg.pinv(R_)
        P_ = Q_ - S_.T @ R_i @ S_
        p_ = q_ - S_.T @ R_i @ r_

        K = -R_i @ S_
        k = -R_i @ r_

        return P_, p_, K, k

    def riccati_forward(self, x0, K, k, A, B, b):
        u = K @ x0 + k
        x = A @ x0 + B @ u + b

        return x, u

    def solve(self, x0, Q:list, Qf:np.array, R:list, S:list, grad:list, gradf:np.array, A:list, B:list, b:list):
        ns = len(Q)
        nx = Q[0].shape[0]
        nu = R[0].shape[0]

        x = []
        u = []
        for i in range(0, ns):
            x.append(np.zeros((nx, 1)))
            u.append(np.zeros((nu, 1)))
        x.append(np.zeros((nx, 1)))

        P = Qf
        p = gradf
        K = deque()
        d = deque()

        for k in range(ns - 1, -1, -1):
            qk = grad[k][0:nx]
            rk = grad[k][nx:nx + nu]
            P, p, Kk, dk = self.riccati_backward(Q[k], S[k], R[k], qk, rk, P, p, A[k], B[k], b[k])
            K.appendleft(Kk)
            d.appendleft(dk)

        x[0] = x0
        for k in range(0, ns):
            x[k + 1], u[k] = self.riccati_forward(x[k], K[k], d[k], A[k], B[k], b[k])

        f = 0.
        for k in range(ns):
            f += 0.5 * x[k].T @ Q[k] @ x[k] + 0.5 * u[k].T @ R[k] @ u[k] + x[k].T @ S[k].T @ u[k] - grad[k].T @ np.vstack((x[k], u[k]))
        f += 0.5 * x[ns].T @ Qf @ x[ns] - gradf.T @ x[ns]

        return x, u, f.ravel()[0]
