import numpy as np
from numba import boolean, float64, int64, njit
from numba.experimental import jitclass
from numpy.linalg import cholesky, inv

from .utils import l1, l2

ZERO = 0.0


@njit(cache=True)
def soft_threshold(y, alpha):
    ret = np.zeros(y.shape)
    ret_full = np.sign(y) * np.maximum(np.abs(y) - alpha, 0.0)
    ret = ret_full
    return ret


@njit(cache=True)
def grad(theta, snl, s1nl, Q_inv):
    grad = Q_inv @ (theta @ snl - s1nl)
    return grad


@njit(cache=True)
def _f(theta, k, Q_chol_inv, y, z, l1_lam, l2_lam):
    return (
        np.sum(np.square(Q_chol_inv @ (y - z @ theta.T).T)) / 2
        + l1(l1_lam, theta[:, k:])
        + l2(l2_lam, theta[:, k:])
    )


@njit(cache=True)
def _iter(
    y,
    z,
    snl,
    s1nl,
    Q_chol_inv,
    Q_inv,
    theta,
    Qbz,
    l_k,
    l1_lam,
    l2_lam,
    l2_lam_step,
    diag_mat,
    k,
    k_nl,
    max_iter,
    ptol,
):
    x_acc = theta.copy()
    x_p = theta.copy()
    x_iter = theta.copy()
    w = 1.0
    w_p = 1.0
    I = np.eye(k, k + k_nl)
    l1_lam_step = l1_lam / l_k
    l1_A = np.ones_like(theta[:, :k]) * ZERO
    l1_F = np.ones_like(theta[:, k:]) * l1_lam_step
    l1_mat = np.hstack((l1_A, l1_F))
    l2_A = np.ones_like(theta[:, :k]) * ZERO
    l2_F = np.ones_like(theta[:, k:]) * l2_lam_step
    l2_mat = np.hstack((l2_A, l2_F))
    f_p = _f(x_p, k, Q_chol_inv, y, z, l1_lam, l2_lam)

    for iter in range(max_iter):
        g = grad(x_acc, snl, s1nl, Q_inv)
        x_iter = (
            soft_threshold(
                x_acc - (g + Qbz + l2_mat * (x_acc - I)) * diag_mat / l_k - I, l1_mat
            )
            + I
        )

        f = _f(x_iter, k, Q_chol_inv, y, z, l1_lam, l2_lam)
        conv2 = abs(f - f_p)

        if conv2 < ptol:
            break

        if f > f_p:
            w = 1.0
            x_acc = x_p.copy()
        else:
            w_p = w
            w = (1 + np.sqrt(1 + 4 * np.power(w, 2))) / 2
            x_acc = x_iter + ((w_p - 1.0) / w) * (x_iter - x_p)
            x_p = x_iter.copy()
            f_p = f

    if f_p < f:
        x_iter = x_p
        f = f_p

    return np.nan_to_num(x_iter, copy=False), f


spec = [
    ("ptol", float64),
    ("l1_lam", float64),
    ("l2_lam", float64),
    ("l2_lam_step", float64),
    ("sparse", boolean),
    ("max_iter", int64),
]


@jitclass(spec)
class PG:
    def __init__(self, ptol, l1_lam, l2_lam, max_iter=1000):
        self.ptol = ptol
        self.l1_lam = l1_lam
        self.l2_lam = l2_lam
        self.l2_lam_step = l2_lam * 2
        if l1_lam == 0:
            self.sparse = False
        else:
            self.sparse = True
        self.max_iter = max_iter

    def fit(self, theta, y, b, z, Szznl, Sz1znl, Q, k, k_nl):
        diag_mat = np.hstack((np.eye(k), np.ones((k, k_nl))))
        theta = theta * diag_mat
        if self.sparse:
            Q_chol_inv = inv(cholesky(Q + ZERO * np.eye(Q.shape[0])))
            Q_inv = Q_chol_inv.T @ Q_chol_inv
            Qbz = Q_inv @ np.outer(b, np.sum(z, axis=0))
            l1_lam = self.l1_lam
            l2_lam = self.l2_lam
            l2_lam_step = self.l2_lam_step
            l_k = (
                np.sqrt(np.sum(np.power(Q_inv, 2)) * np.sum(np.power(Szznl[k:, :], 2)))
                + l2_lam_step
            )
            param, f_s = _iter(
                y,
                z,
                Szznl,
                Sz1znl,
                Q_chol_inv,
                Q_inv,
                theta,
                Qbz,
                l_k,
                l1_lam,
                l2_lam,
                l2_lam_step,
                diag_mat,
                k,
                k_nl,
                self.max_iter,
                self.ptol,
            )
        else:
            param = Sz1znl @ inv(Szznl)
            param = param * diag_mat

        return param[:, :k], param[:, k:]
