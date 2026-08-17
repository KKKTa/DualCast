import gc
import math
import multiprocessing
import os
import pickle
import time
import warnings

from numba.core.errors import NumbaPerformanceWarning

warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", category=NumbaPerformanceWarning)

import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy

import itertools_len as itertools
import lmfit
import matplotlib.pyplot as plt
import numpy as np
import tensorly as tl
from numba import njit, prange
from numpy.linalg import pinv
from tensorly.cp_tensor import cp_to_tensor
from tensorly.metrics.regression import RMSE
from tqdm import tqdm

from .proximal_gradient import PG
from .utils import *

# np.seterr(all='raise')

h = 1.0e-5
INF = 1.0e20


@njit(cache=True)
def _defunc(
    z, A, F, C, b, d, T, kq, kl, k_nl, comb_list, stabilize=False, threshold=1e5
):
    for t in range(T - 1):
        non = make_state_vec(z[t], kq, kl, k_nl, comb_list)
        next = A @ z[t] + F @ non + b
        if np.any(np.isnan(next)):
            mask_nan = np.isnan(next)
            next[mask_nan] = z[t][mask_nan].copy()

        if stabilize:
            if t >= 5:
                for i in range(len(next)):
                    mean_val = np.mean(z[: t + 1, i])
                    std_val = np.std(z[: t + 1, i])

                    limit = 3.0 * std_val + 1e-5
                    upper_bound = mean_val + limit
                    lower_bound = mean_val - limit

                    if next[i] > upper_bound:
                        next[i] = upper_bound
                    elif next[i] < lower_bound:
                        next[i] = lower_bound

        z[t + 1] = np.clip(next, -threshold, threshold)

    return z, z @ C.T + d


def update_mu0(Ez, limit_val=5):
    return np.clip(Ez[0].copy(), a_min=-limit_val, a_max=limit_val)


def update_Q0(Ez, Ezz, kq, kl, covariance_type):
    Q0 = Ezz[0] - np.outer(Ez[0], Ez[0])

    if covariance_type == "full":
        pass
    elif covariance_type == "diag":
        Q0 = np.diag(np.diag(Q0))
    elif covariance_type == "isotropic":
        k = Q0.shape[0]
        Q0 = np.diag(np.full(k, np.trace(Q0) / k))

    return Q0


@njit(parallel=True)
def update_AF(
    A,
    F,
    Q,
    y,
    b,
    method,
    z,
    Sz1z,
    SzzT,
    Szznl,
    Sz1znl,
    kq,
    kl,
    k_nl,
):
    # bound_nl = int(kl / 2)
    # bound_nl = kl  # linear
    # bound_nl = -1  # nonlinear
    bound_nl = 3  # nonlinear

    A_blocks = extract_diag_blocks(A, kq, kq, kl)
    F_blocks = extract_diag_blocks(F, kq, k_nl, kl)
    init = np.concatenate((A_blocks, F_blocks), axis=1)

    A = np.zeros((kq * kl, kq * kl))
    F = np.zeros((kq * kl, k_nl * kl))

    for blk in prange(kl):
        i_start = blk * kq
        i_end = (blk + 1) * kq

        j_start = blk * k_nl
        j_end = (blk + 1) * k_nl

        # linear dynamical systems
        if blk < bound_nl:
            A_tmp = Sz1z[i_start:i_end, i_start:i_end] @ pinv(
                SzzT[i_start:i_end, i_start:i_end]
            )

            A[i_start:i_end, i_start:i_end] = A_tmp.copy()
        # nonlinear dynamical systems
        else:
            params = init[:, :, blk]
            A_tmp, F_tmp = method.fit(
                params,
                y[:, i_start:i_end],
                b[i_start:i_end],
                z[:, :, blk],
                Szznl[:, :, blk],
                Sz1znl[:, :, blk],
                Q[i_start:i_end, i_start:i_end],
                kq,
                k_nl,
            )
            A[i_start:i_end, i_start:i_end] = A_tmp.copy()
            F[i_start:i_end, j_start:j_end] = F_tmp.copy()

    return A, F


def update_C(C, d, Ez, Szz, Sxz, I, R, kq, clam, covariance_type):
    if len(I) == 1:
        C_ = (Sxz - np.outer(d, np.sum(Ez, axis=0))) @ pinv(Szz)
        C = [C_]

    else:
        if covariance_type == "full":
            omg = np.eye(np.prod(I)) @ pinv(R)
        elif covariance_type == "diag":
            omg = 1 / np.diag(R)
        elif covariance_type == "isotropic":
            omg = 1 / R[0, 0]

        # omg = np.eye(np.prod(I)) @ pinv(R)
        psi = Szz
        phi = Sxz
        ups = np.outer(d, np.sum(Ez, axis=0))

        C = update_multilinear_operator(C, omg, psi, phi, ups, clam, covariance_type)

    return C


def update_b(Ez, Eznl, A, F):
    return np.mean(Ez[1:] - Ez[:-1] @ A.T - Eznl[:-1] @ F.T, axis=0)


def update_d(vecX, Ez, C):
    return np.mean(vecX - Ez @ kronecker(C, reverse=True).T, axis=0)


@njit(cache=True)
def iter_Q(Ez, Szz, Ezz, Ez1z, cov_type, matA, F, b, j_m, kq, kl, k_nl, comb_list):
    Ez1z_T = Ez1z.transpose(0, 2, 1)

    if cov_type == "full":
        Q = Szz - Ezz[0]
        for t in range(len(Ez) - 1):
            A_nl = F @ j_m[t]
            ofset = (
                F @ make_state_vec(Ez[t], kq, kl, k_nl, comb_list) + b - A_nl @ Ez[t]
            )
            A_nl += matA
            val = A_nl @ (Ez1z_T[t] - np.outer(Ez[t], ofset)) + np.outer(
                Ez[t + 1], ofset
            )
            Q += A_nl @ Ezz[t] @ A_nl.T + np.outer(ofset, ofset) - val - val.T

    elif cov_type == "diag":
        Q = np.diag(Szz) - np.diag(Ezz[0])
        for t in range(len(Ez) - 1):
            A_nl = F @ j_m[t]
            ofset = (
                F @ make_state_vec(Ez[t], kq, kl, k_nl, comb_list) + b - A_nl @ Ez[t]
            )
            A_nl += matA
            val = np.diag(A_nl @ (Ez1z_T[t] - np.outer(Ez[t], ofset))) + np.diag(
                np.outer(Ez[t + 1], ofset)
            )
            Q += (
                np.diag(A_nl @ Ezz[t] @ A_nl.T)
                + np.diag(np.outer(ofset, ofset))
                - 2 * val
            )
        Q = np.diag(Q)

    elif cov_type == "isotropic":
        L = Szz.shape[0]
        Q = np.trace(Szz) - np.trace(Ezz[0])
        for t in range(len(Ez) - 1):
            A_nl = F @ j_m[t]
            ofset = (
                F @ make_state_vec(Ez[t], kq, kl, k_nl, comb_list) + b - A_nl @ Ez[t]
            )
            A_nl += matA
            val = np.trace(A_nl @ (Ez1z_T[t] - np.outer(Ez[t], ofset))) + np.trace(
                np.outer(Ez[t + 1], ofset)
            )
            Q += (
                np.trace(A_nl @ Ezz[t] @ A_nl.T)
                + np.trace(np.outer(ofset, ofset))
                - 2 * val
            )
        Q /= L
        Q = Q * np.eye(L)

    Q = (Q + Q.T) / 2

    return Q


def update_Q(model, Ez, Szz, Ezz, Ez1z, cov_type):
    kq, kl = model.ranks
    k_nl = model.k_nl
    Q = iter_Q(
        Ez,
        Szz,
        Ezz,
        Ez1z,
        cov_type,
        model.A,
        model.F,
        model.b,
        model.j_m,
        kq,
        kl,
        k_nl,
        model.comb_list,
    )
    return Q / (len(Ez) - 1)


def update_R(vecX, Ez, Sxx, Szz, Sxz, matC, d, cov_type):
    if cov_type == "full":
        val = matC @ (Sxz.T - np.einsum("ij,l->jl", Ez, d)) + np.einsum(
            "ij,l->jl",
            vecX,
            d,
        )
        R = Sxx - val - val.T + matC @ Szz @ matC.T + np.outer(d, d) * len(vecX)

    elif cov_type == "diag":
        val = np.diag(matC @ (Sxz.T - np.einsum("ij,l->jl", Ez, d))) + np.diag(
            np.einsum("ij,l->jl", vecX, d),
        )
        R = (
            np.diag(Sxx)
            - 2 * val
            + np.diag(matC @ Szz @ matC.T)
            + np.diag(np.outer(d, d) * len(vecX))
        )
        R = np.diag(R)

    elif cov_type == "isotropic":
        N = Sxx.shape[0]
        val = np.trace(matC @ (Sxz.T - np.einsum("ij,l->jl", Ez, d))) + np.trace(
            np.einsum("ij,l->jl", vecX, d),
        )
        R = (
            np.trace(Sxx)
            - 2 * val
            + np.trace(matC @ Szz @ matC.T)
            + np.trace(np.outer(d, d) * len(vecX))
        )
        R /= N
        R = R * np.eye(N)

    R = (R + R.T) / 2
    R += np.eye(R.shape[0]) * 1e-10

    return R / len(vecX)


def fit_each(
    model_org,
    tensor,
    max_iter,
    hyper_param,
    init,
    dim_list=None,
    return_model=False,
    multi=True,
    seq_len=None,
    ts=None,
    initialize_S_factors=False,
    valid_tensor=None,
    dt=4,
):
    if dim_list is None:
        dim_list = model_org.dim_list
    loss_list = []
    vali_err_list = []

    Models = []
    Results = []

    init_tensor = tensor[:seq_len]

    for dim_poly in dim_list:
        model = deepcopy(model_org)
        model = model.initialize(
            init_tensor,
            hyper_param,
            dim_poly,
            seq_len=seq_len,
            init=init,
            initialize_S_factors=initialize_S_factors,
        )
        if ts != None:
            model.ts = ts
        if initialize_S_factors:
            if model.ablation_seasonality:
                Xd = init_tensor
                Xs = np.zeros(init_tensor.shape)
            else:
                Xd, Xs, _ = STL_decomp(init_tensor, period=int(model.n_season / 2))
        else:
            if model.ablation_seasonality:
                Xs = np.zeros(init_tensor.shape)
            else:
                Xs = model.predict_seasonal_tensor(n_sample=model.T, ts=model.ts)
            Xd = init_tensor - Xs
        model.iter = 0
        total_iter = 0
        gd_method = PG(
            ptol=model.ptol,
            l1_lam=model.l1_lam,
            l2_lam=model.l2_lam,
        )
        history_loss = [INF]
        model_d = INF
        loss_d = INF
        ascent = 0

        tic = time.time()
        vecXd = tensor_to_vec(Xd, sequential=True)
        Sxx = vecXd.T @ vecXd

        # model estimation
        for inner_itr in range(max_iter):
            try:
                # inference step
                model.forward(vecXd)
                model.backward()

                oldA = model.A.copy()
                oldF = model.F.copy()
                oldC = model.C.copy()

                # learning step
                model.solve(vecXd, Sxx, gd_method)

                pconv = model.check_convergence(oldA, oldF, oldC)
                if pconv == "pconv":
                    conv = "pconv"
                    break

                loss = model.score(vecXd)
                model.iter += 1
                total_iter += 1

            except (np.linalg.LinAlgError, FloatingPointError, ValueError) as e:
                # print(f"hyper_param: {hyper_param}, dim_poly: {dim_poly}")
                print("fit_each:", e, flush=True)

                traceback.print_exc()

                conv = "except"
                break

            history_loss.append(loss)
            if loss < loss_d:
                model_d = deepcopy(model)
                loss_d = loss

            if (
                abs(history_loss[-1] - history_loss[-2]) / abs(history_loss[-2])
                < model.tol
            ):
                conv = "conv1"
                break
            elif history_loss[-1] > history_loss[-2]:
                ascent += 1
                if ascent > model.MAX_ASCENT:
                    conv = "conv2"
                    break
            else:
                ascent = 0

        else:
            conv = "max"

        if not model_d.ablation_seasonality:
            try:
                _, vecXd = model_d.gen(T=seq_len)
                Xd = vec_to_tensor(vecXd, model_d.I, sequential=True)
                Xs = init_tensor - Xd
                mean_Xs = compute_mean_tensor(Xs, model_d.n_season, t=model_d.ts)
                if initialize_S_factors:
                    S = model_d.S_factors
                    model_d.Pcomp = [
                        unfolding_dot_khatri_rao(mean_Xs, S, mode)
                        for mode in range(len(S))
                    ]
                    model_d.Qcomp = [compute_accum(S, mode) for mode in range(len(S))]
            except (np.linalg.LinAlgError, FloatingPointError, AttributeError):
                model_d = INF

        result = make_result(model_d, hyper_param, dim_poly, init_tensor, loss_d)
        result["conv"] = conv
        loss_list.append(result["err"])

        if isinstance(model_d, float):
            result["vali_err"] = INF
            vali_err_list.append(result["vali_err"])
            Results.append(result)
            Models.append(model_d)
            continue

        if return_model:
            model_d.history_loss = history_loss
            model_d.conv = conv
            model_d.iter = total_iter
            model_d.loss = loss_d
            model_d.error = result["err"]

        # validation
        if valid_tensor is not None:
            try:
                lf = model_d.lf
                X = np.concatenate((tensor, valid_tensor), axis=0)
                preds = []
                trues = []

                for t in range(seq_len + dt, len(X) - lf, dt):
                    ts = t - seq_len
                    Xc = X[t - seq_len : t]
                    true = X[t : t + lf]

                    model_d.update(Xc, ts, seq_len, dt)

                    pred, _, _ = model.predict(
                        seq_len + lf, ts, mu0=model.mu0, shape=Xc.shape[1:]
                    )
                    _, pred = np.split(pred, [seq_len])
                    preds.append(pred)
                    trues.append(true)

                preds = np.array(preds)
                trues = np.array(trues)

                error = RMSE(trues, preds)
            except Exception as e:
                print("exception while validation", e)
                error = INF
            result["vali_err"] = error

            vali_err_list.append(error)
            # print(f'valid time: {time.time()-tic}', flush=True)

        Results.append(result)
        Models.append(model_d)
        # print()

    if valid_tensor is None:
        best_i = np.argmin(loss_list)
    else:
        best_i = np.argmin(vali_err_list)

    best_result = Results[best_i]
    if return_model:
        best_result["md"] = deepcopy(Models[best_i])

    if multi:
        del tensor
        del valid_tensor
        del model
        del model_org
        del Sxx
        del gd_method
        del Models
        gc.collect()
    # print(best_result)

    # print(hyper_param, flush=True)

    return best_result


@njit(cache=True)
def _forward(
    vecX,
    T,
    mu0,
    N,
    ranks,
    L,
    matA,
    F,
    Q,
    Q0,
    b,
    matC,
    R,
    d,
    j_delta,
    k_nl,
    comb_list,
):
    llh = 0.0
    kq, kl = ranks
    mu = np.empty((T, L))
    mu_t = np.zeros((T, L))
    mu_o = np.zeros((T, N))
    V = np.zeros((T, L, L))
    P = np.zeros((T, L, L))
    A_nl_mu = np.zeros((T, L, L))

    inv_R = pinv(R)

    for t in range(T):
        if t == 0:
            KP = Q0
            mu_t[0] = mu0
        else:
            non_linear_term = make_state_vec(mu[t - 1], kq, kl, k_nl, comb_list)
            mu_t[t] = matA @ mu[t - 1] + F @ non_linear_term + b
            A_nl_mu[t - 1] = jacobian(
                mu[t - 1], matA, F, j_delta, kq, kl, k_nl, comb_list
            )
            P[t] = A_nl_mu[t - 1] @ V[t - 1] @ A_nl_mu[t - 1].T + Q
            KP = P[t]
        C_T_inv_R = matC.T @ inv_R

        inv_KP = pinv(KP)
        inv_V_t = inv_KP + C_T_inv_R @ matC
        V[t] = pinv(inv_V_t)

        K = V[t] @ C_T_inv_R

        mu_o[t] = matC @ mu_t[t] + d
        dlt = vecX[t] - mu_o[t]
        mu[t] = mu_t[t] + K @ dlt

        _, logdet_R = np.linalg.slogdet(R)
        _, logdet_KP = np.linalg.slogdet(KP)
        _, logdet_inv_V_t = np.linalg.slogdet(inv_V_t)

        logdet_sgm = logdet_R + logdet_KP - logdet_inv_V_t

        df_term1 = dlt.T @ inv_R @ dlt
        df_term2 = dlt.T @ C_T_inv_R.T @ V[t] @ C_T_inv_R @ dlt
        df = 0.5 * (df_term1 - df_term2)

        llh += -0.5 * N * np.log(2 * np.pi) - 0.5 * logdet_sgm - df

    A_nl_mu[t] = jacobian(mu[t], matA, F, j_delta, kq, kl, k_nl, comb_list)

    return mu, mu_t, mu_o, V, P, A_nl_mu, llh


@njit(cache=True)
def _backward(
    T,
    ranks,
    L,
    mu,
    mu_t,
    V,
    P,
    A_nl,
    ks,
    k_d2,
    k_d3,
    k_nl,
    j_delta,
    comb_list,
    ignore_ho_noise,
):
    kq, kl = ranks
    Ez = np.zeros((T, L))
    Ezz = np.zeros((T, L, L))
    Ez1z = np.zeros((T, L, L))
    Eznl = np.zeros((T, k_nl * kl))
    aug_z = np.zeros((T, kq + k_nl, kl))
    j_m = np.zeros((T, k_nl * kl, L))
    Szznl = np.zeros((kq + k_nl, kq + k_nl, kl))
    Sz1znl = np.zeros((kq, kq + k_nl, kl))

    Ws = np.zeros((T, L, L))

    Vhat = V[-1].copy()
    Ez[-1] = mu[-1].copy()
    Ezz[-1] = Vhat + np.outer(Ez[-1], Ez[-1])

    Ws[-1] = Vhat.copy()

    for t in range(T - 2, -1, -1):
        J = V[t] @ A_nl[t].T @ pinv(P[t + 1])
        Ez[t] = mu[t] + J @ (Ez[t + 1] - mu_t[t + 1])
        C_lag_full = Vhat @ J.T
        Ez1z[t] = C_lag_full + np.outer(Ez[t + 1], Ez[t])
        Vhat = V[t] + J @ (Vhat - P[t + 1]) @ J.T
        Ws[t] = Vhat.copy()
        Ezz[t], Eznl[t] = moment(Ez[t], Vhat, kq, kl, ks, k_d2, k_d3, k_nl)
        j_m[t] = diff_m(Ez[t], j_delta, kq, kl, k_nl, comb_list)

        # ★追加: 厳密な行列をブロックごとに計算して加算
        if not ignore_ho_noise and ks[0] == -1:
            for b in range(kl):
                idx_start = b * kq
                idx_end = (b + 1) * kq

                # 各種ブロックを抽出
                mu_next_b = Ez[t + 1, idx_start:idx_end]
                mu_curr_b = Ez[t, idx_start:idx_end]
                W_curr_b = Vhat[idx_start:idx_end, idx_start:idx_end]
                C_lag_b = C_lag_full[idx_start:idx_end, idx_start:idx_end]

                # 蓄積
                Szznl[:, :, b] += exact_Szznl(mu_curr_b, W_curr_b, kq, k_nl, comb_list)
                Sz1znl[:, :, b] += exact_Sz1znl(
                    mu_next_b, mu_curr_b, W_curr_b, C_lag_b, kq, k_nl, comb_list
                )

    aug_z = np.concatenate(
        (
            np.ascontiguousarray(Ez.reshape(T, kl, kq).transpose(0, 2, 1)),
            np.ascontiguousarray(Eznl.reshape(T, kl, k_nl).transpose(0, 2, 1)),
        ),
        axis=1,
    )

    return (
        Ez,
        Eznl,
        aug_z,
        Ezz,
        Ez1z,
        j_m / (2 * h),
        Ws,
        Szznl,
        Sz1znl,
    )


class DualCast:
    def __init__(
        self,
        initial_state_cov="full",
        transition_cov="full",
        observation_cov="full",
        dim_list=[2],
        tol=2.0e-3,
        ptol=1.0e-10,
        th=1.0e-2,
        n_season=52,
        num_works=-1,
        print_log=False,
        verbose=False,
        random_state=42,
        sparsity=True,
        ablation_seasonality=False,
        ablation_blocking=False,
        ablation_shift=False,
        ignore_ho_noise=True,
    ):
        covariance_types = ["full", "isotropic", "diag"]

        assert initial_state_cov in covariance_types
        assert transition_cov in covariance_types
        assert observation_cov in covariance_types

        self.init_state_cov = initial_state_cov
        self.trans_cov = transition_cov
        self.obs_cov = observation_cov
        self.history = []

        self.dim_list = dim_list
        self.tol = tol
        self.ptol = ptol
        self.th = th
        self.n_season = n_season
        if num_works == -1:
            self.num_works = int(os.cpu_count() / 2)
        else:
            self.num_works = num_works
        self.print_log = print_log
        self.verbose = verbose
        self.random_state = random_state
        self.conv = None

        self.sparsity = sparsity
        self.ablation_seasonality = ablation_seasonality
        self.ablation_blocking = ablation_blocking
        self.ablation_shift = ablation_shift
        self.ignore_ho_noise = ignore_ho_noise

    def init_params(self, tensor, init="random", initialize_S_factors=False):
        k = self.k
        k_nl = self.k_nl
        I = self.I
        J = self.ranks
        kq, kl = J
        L = self.L
        M = self.M
        N = self.N

        self.Q0 = np.eye(L)
        self.Q = np.eye(L)  # Gamma
        self.R = np.eye(N)  # Sigma

        if self.ablation_seasonality:
            self.S_factors = None
            Xs = np.zeros(tensor.shape)
        else:
            if initialize_S_factors:
                self.S_factors = init_seasonal_factors(
                    tensor, self.k_seasonal, self.n_season, self.ts
                )
                self.Pcomp = None
                self.Qcomp = None
            Xs = self.predict_seasonal_tensor(len(tensor), ts=self.ts)

        Xd = tensor - Xs

        if init == "random":
            self.mu0 = np.zeros(L)
            self.A = np.eye(L)
            self.F = np.zeros((L, k_nl * kl))
            tucker_ranks = [tensor.shape[0]] + self.ranks
            core, factors = tucker_decomp(
                Xd, tucker_ranks, self.random_state, init=init
            )
            self.C = [factors[1], factors[2]]
            self.b = np.zeros(L)
            self.d = np.zeros(N)
            self.MAX_ASCENT = 5

        elif init == "safe":
            tucker_ranks = [tensor.shape[0]] + self.ranks
            core, factors = tucker_decomp(
                Xd, tucker_ranks, self.random_state, init="random"
            )
            self.C = [factors[1], factors[2]]
            state_trajectory = np.tensordot(factors[0], core, axes=(1, 0))
            self.mu0 = tensor_to_vec(state_trajectory[0], sequential=False)
            self.A = np.eye(L)
            self.F = np.zeros((L, k_nl * kl))
            self.b = np.zeros(L)
            self.d = np.zeros(N)
            self.MAX_ASCENT = 5

    def initialize(
        self,
        tensor,
        hyper_param,
        dim_poly,
        seq_len=None,
        init="random",
        initialize_S_factors=False,
    ):
        shape = tensor.shape
        kq = hyper_param[0]
        kl = hyper_param[1]
        self.ranks = [kq, kl]
        self.k = kq
        if self.sparsity:
            self.lam = hyper_param[2]
        else:
            self.lam = 0
        self.lam2 = hyper_param[2]
        self.clam = hyper_param[3]
        self.trans_offset = hyper_param[4]
        self.k_seasonal = hyper_param[5]
        self.k_nl, self.comb_list = get_k_nl(self.k, dim_poly)

        if seq_len != None:
            self.T = seq_len
        else:
            self.T = shape[0]
        self.M = len(shape) - 1
        self.N = np.prod(shape[1:])
        self.L = int(np.prod(self.ranks))
        self.I = shape[1:]

        self.dim_poly = dim_poly
        self.j_delta = np.diag(np.full(self.L, h, dtype=float))
        self.l1_lam = self.lam * self.l1_r
        self.l2_lam = self.lam2 * self.l2_r

        self.init_params(tensor, init=init, initialize_S_factors=initialize_S_factors)
        self.set_ks(dim_poly)

        return self

    def forward(self, vecX):
        (
            self.mu,
            self.mu_t,
            self.mu_o,
            self.V,
            self.P,
            self.A_nl_mu,
            self.llh,
        ) = _forward(
            vecX,
            self.T,
            self.mu0,
            self.N,
            np.array(self.ranks),
            self.L,
            self.A,
            self.F,
            self.Q,
            self.Q0,
            self.b,
            kronecker(self.C, reverse=True),
            self.R,
            self.d,
            self.j_delta,
            self.k_nl,
            self.comb_list,
        )

    def backward(self):
        (
            self.Ez,
            self.Eznl,
            self.aug_z,
            self.Ezz,
            self.Ez1z,
            self.j_m,
            self.Ws,
            self.Szznl,
            self.Sz1znl,
        ) = _backward(
            self.T,
            np.array(self.ranks),
            self.L,
            self.mu,
            self.mu_t,
            self.V,
            self.P,
            self.A_nl_mu,
            self.ks,
            self.k_d2,
            self.k_d3,
            self.k_nl,
            self.j_delta,
            self.comb_list,
            self.ignore_ho_noise,
        )

    def solve(self, vecX, Sxx, method):
        T = self.T
        kq, kl = self.ranks
        Ez = self.Ez
        Ezz = self.Ezz
        Ez1z = self.Ez1z
        Eznl = self.Eznl

        Szz = np.sum(Ezz, axis=0)
        Sz1z = np.sum(Ez1z[:-1], axis=0)
        Sxz = np.sum((np.outer(vecX[t], Ez[t]) for t in range(T)), axis=0)
        SzzT = Szz - Ezz[-1]
        y = Ez[1:] - self.b
        z = self.aug_z[:-1]

        if self.ignore_ho_noise:
            Szznl = np.einsum("tib,tjb->ijb", z, z)
            Sz1znl = np.einsum(
                "tib,tjb->ijb", Ez[1:].reshape(T - 1, kl, kq).transpose(0, 2, 1), z
            )
        else:
            Szznl = self.Szznl
            Sz1znl = self.Sz1znl

        self.mu0 = update_mu0(Ez)
        self.Q0 = update_Q0(Ez, Ezz, kq, kl, self.init_state_cov)
        self.A, self.F = update_AF(
            self.A,
            self.F,
            self.Q,
            y,
            self.b,
            method,
            z,
            Sz1z,
            SzzT,
            Szznl,
            Sz1znl,
            kq,
            kl,
            self.k_nl,
        )
        self.C = update_C(
            self.C,
            self.d,
            self.Ez,
            Szz,
            Sxz,
            self.I,
            self.R,
            kq,
            self.clam,
            self.obs_cov,
        )
        if self.trans_offset:
            self.b = update_b(Ez, Eznl, self.A, self.F)
        if self.obs_offset:
            self.d = update_d(vecX, Ez, self.C)
        self.Q = update_Q(self, Ez, Szz, Ezz, Ez1z, self.trans_cov)
        self.R = update_R(
            vecX,
            Ez,
            Sxx,
            Szz,
            Sxz,
            kronecker(self.C, reverse=True),
            self.d,
            self.obs_cov,
        )

    def check_convergence(self, oldA, oldF, oldC):
        rA = rel_norm(self.A, oldA)
        rF = rel_norm(self.F, oldF)
        rC_list = [rel_norm(cn, co) for cn, co in zip(self.C, oldC, strict=False)]
        rC = max(rC_list) if rC_list else 0.0

        param_change = max(rA, rF, rC)
        if param_change < 1e-4:
            return "pconv"
        return None

    def init_initial_state(self, mu0=None, est_vars=None, limit_val=5):
        params = lmfit.Parameters()

        # Initial state
        if mu0 is None:
            mu0 = np.zeros(self.L)

        vary = True if "mu0" in est_vars else False
        for i in range(self.L):
            params.add(
                f"mu0{i}", value=mu0[i], vary=vary, min=-limit_val, max=limit_val
            )

        self.mu0 = mu0

        return params

    def params2numpy(self, params):
        mu0 = np.zeros(self.L)
        for i in range(len(mu0)):
            mu0[i] = params[f"mu0{i}"]

        return mu0

    def residual(self, params, data, seq_len, ts):
        try:
            mu0 = self.params2numpy(params)
            pred, _, _ = self.predict(len=seq_len, ts=ts, mu0=mu0, shape=data.shape[1:])
            return (data - pred).ravel()
        except (FloatingPointError, ValueError, np.linalg.LinAlgError):
            return np.full(data.size, 1.0e10)

    def update(self, Xc, ts, seq_len, dt, print_log=False):
        # update initial state
        pre_mu0 = (
            self.mu[dt].copy()
            if hasattr(self, "mu") and len(self.mu) > dt
            else np.zeros(self.L)
        )
        # if print_log: print(pre_mu0)

        try:
            result = lmfit.minimize(
                self.residual,
                self.init_initial_state(
                    mu0=pre_mu0, est_vars="mu0"
                ),  # mu0の初期値を設定する必要があるかどうか微妙
                method="leastsq",
                args=(Xc, seq_len, ts),
                xtol=1e-7,
                ftol=1e-7,
                nan_policy="propagate",
            )

            self.mu0 = self.params2numpy(result.params)
        except Exception as e:
            print(
                f"Update optimization failed at t={ts + seq_len}: {e}. Fallback to predicted state."
            )
            self.mu0 = pre_mu0

        try:
            z, gen_vec = self.gen(T=seq_len, mu0=self.mu0, stabilize=True)
            self.mu = z
            gen = vec_to_tensor(gen_vec, Xc.shape[1:], sequential=True)

            # Online update of seasonal factors
            if not self.ablation_seasonality:
                seas_mean_tensor = compute_mean_tensor(
                    Xc - gen, self.n_season, ts, remove_temporal_mean=False
                )
                self.S_factors = self.online_update_seasonality(
                    seas_mean_tensor, self.S_factors
                )

        except FloatingPointError:
            print(f"Critical divergence in state generation at t={ts + seq_len}.")
            for i in range(len(self.mu - dt)):
                self.mu[i] = self.mu[i + dt]
            self.mu[-dt:] = np.zeros_like(self.mu[-dt:])

    def online_update_seasonality(self, tensor, seas_factors, forgetting_rate=0.1):
        for _ in range(1):
            for mode in reversed(range(len(seas_factors))):
                # Update complementary matrices
                self.Pcomp[mode] += unfolding_dot_khatri_rao(tensor, seas_factors, mode)
                self.Qcomp[mode] += (
                    seas_factors[mode].T @ seas_factors[mode]
                ) @ compute_accum(seas_factors, skip_matrix=mode)

                # Online update
                seas_factors[mode] = tl.transpose(
                    tl.solve(
                        tl.transpose(self.Qcomp[mode]), tl.transpose(self.Pcomp[mode])
                    )
                )

        return seas_factors

    def save_params(self, outdir, ts):
        model_path = outdir / f"model/ts={str(ts)}"
        model_path.mkdir(parents=True, exist_ok=True)

        kq, kl = self.ranks
        unit = 0.6

        slabels = make_feature_names(kq, self.dim_poly)

        # save the parameters for trend tensor
        np.save(model_path / "A.npy", self.A)
        np.save(model_path / "F.npy", self.F)
        for i, factor in enumerate(self.C):
            np.save(model_path / f"C{i}.npy", factor)
        np.save(model_path / "b.npy", self.b)
        np.save(model_path / "d.npy", self.d)
        np.save(model_path / "Q0.npy", self.Q0)
        np.save(model_path / "Q.npy", self.Q)
        np.save(model_path / "R.npy", self.R)
        np.save(model_path / "mu0.npy", self.mu0)

        # make figures for trend tensor
        A_diag_blocks = extract_diag_blocks(self.A, kq, kq, kl)
        for i in range(kl):
            plt.figure(figsize=((kq + 3) * unit, (kq + 3) * unit))
            sns.heatmap(
                A_diag_blocks[:, :, i],
                cmap="bwr",
                square=True,
                annot=True,
                linewidth=1,
                center=0.0,
                fmt=".4f",
                xticklabels=slabels[:kq],
                yticklabels=slabels[:kq],
            )
            plt.savefig(model_path / f"A{str(i)}.jpg")
            plt.close()

        F_diag_blocks = extract_diag_blocks(self.F, kq, self.k_nl, kl)
        for i in range(kl):
            plt.figure(figsize=(self.k_nl * unit * 1.5, (kq + 3) * unit))
            sns.heatmap(
                F_diag_blocks[:, :, i],
                cmap="bwr",
                square=True,
                annot=True,
                linewidth=0.1,
                center=0.0,
                fmt=".4f",
                xticklabels=slabels[kq:],
                yticklabels=slabels[:kq],
            )
            plt.savefig(model_path / f"F{str(i)}.jpg")
            plt.close()

        for i, factor in enumerate(self.C):
            plt.figure(figsize=((factor.shape[1] + 3) * unit, factor.shape[0] * unit))
            sns.heatmap(
                factor,
                cmap="bwr",
                square=True,
                annot=True,
                linewidth=0.1,
                center=0.0,
                fmt=".4f",
            )
            plt.savefig(model_path / f"C{str(i)}.jpg")
            plt.close()

        plt.figure(figsize=(self.Q0.shape[1] * unit, self.Q0.shape[0] * unit))
        sns.heatmap(
            self.Q0,
            cmap="bwr",
            square=True,
            annot=True,
            linewidth=0.1,
            center=0.0,
            fmt=".4f",
        )
        plt.savefig(model_path / "Q0.jpg")
        plt.close()

        plt.figure(figsize=(self.Q.shape[1] * unit, self.Q.shape[0] * unit))
        sns.heatmap(
            self.Q,
            cmap="bwr",
            square=True,
            annot=True,
            linewidth=0.1,
            center=0.0,
            fmt=".4f",
        )
        plt.savefig(model_path / "Q.jpg")
        plt.close()

        plt.figure(figsize=(self.R.shape[1] * unit / 2, self.R.shape[0] * unit / 2))
        sns.heatmap(self.R, cmap="bwr", square=True, linewidth=0.1, center=0.0)
        plt.savefig(model_path / "R.jpg")
        plt.close()

        np.save(model_path / "Ez.npy", self.Ez)

        np.save(model_path / "Ws.npy", self.Ws)

        # save parameter for seasonal tensor
        if not self.ablation_seasonality:
            for i, factor in enumerate(self.S_factors):
                np.save(model_path / f"S{str(i)}.npy", factor)
                if i == 0:
                    plt.figure()
                    plt.plot(factor)
                    plt.savefig(model_path / f"S{str(i)}.jpg")
                    plt.close()
                else:
                    plt.figure(
                        figsize=(factor.shape[1] * unit * 2, factor.shape[0] * unit)
                    )
                    sns.heatmap(
                        factor, cmap="bwr", square=True, linewidth=1, center=0.0
                    )
                    plt.savefig(model_path / f"S{str(i)}.jpg")
                    plt.close()

        with open(f"{model_path}/model.pickle", mode="wb") as f:
            pickle.dump(self, f)

    def print_params(self):
        print(f"mu0:{self.mu0}\n")
        print(f"A:{self.A}\n")
        print(f"F:{self.F}\n")
        print(f"C:{self.C}\n")
        print(f"b:{self.b}\n")
        print(f"d:{self.d}\n")
        print(f"Q:{self.Q}\n")
        print(f"R:{self.R}\n")

    def print_result(self, time=None):
        if self.print_log:
            print(f"loss: {self.loss}, rmse: {self.error}")
            print(f"lambda1: {self.lam}")
            print(f"lambda2: {self.lam2}")
            if time is not None:
                print(f"process time: {time}")
            print(f"dim: {self.dim_poly}, kq: {self.ranks[0]}, kl: {self.ranks[1]}")
            print(f"k_seasonal: {self.k_seasonal}")
            print(f"iter_num: {self.iter}")
            print(f"conv: {self.conv}")

    def set_ks(self, dim_poly):
        ks = np.ones(2) * self.k
        if dim_poly == 2:
            ks[0] = -1
            ks[1] = -1
        elif dim_poly == 3:
            ks[0] = self.k
            ks[1] = -1
        elif dim_poly == 4:
            ks[0] = self.k
            ks[1] = self.k
        self.ks = ks
        self.k_d2 = math.comb(self.k, 2)
        self.k_d3 = math.comb(self.k, 3)

    def loglikelihood(self, vecX):
        try:
            self.forward(vecX)
        except Exception as e:
            self.llh = -INF
            print("missed forwarding")
            print(e)
        return self.llh

    def err(self, data):
        try:
            _, Obs_tr = self.gen()
            Xd = vec_to_tensor(
                Obs_tr,
                self.I,
                sequential=True,
            )
            if self.ablation_seasonality:
                Xs = np.zeros(Xd.shape)
            else:
                Xs = self.predict_seasonal_tensor(self.T, ts=self.ts)
            self.Obs = Xd + Xs
            if np.any(np.isnan(self.Obs)):
                return None
            return RMSE(data, self.Obs)
        except Exception as e:
            print("exception while calcurating err")
            print(e)
            return None

    def l1_l2(self):
        F = self.F
        return l1(self.l1_lam, F) + l2(self.l2_lam, F)

    def score(self, vecX, llh=None):
        if llh is None:
            llh = self.loglikelihood(vecX)
        return -llh + self.l1_l2()

    def search_and_fit(
        self, tensor, max_iter, hyper_params, init, valid_tensor, seq_len
    ):
        num_networks = np.minimum(self.num_works, len(hyper_params))
        print(f"num of CPUs:{self.num_works}")
        print(f"num of params:{len(hyper_params)}")
        print(f"num of processes:{num_networks}")
        if self.num_works > 1:
            if self.verbose:
                results = {}
                with tqdm(total=len(hyper_params)) as pbar:
                    with ProcessPoolExecutor(
                        max_workers=num_networks,
                        mp_context=multiprocessing.get_context("spawn"),
                    ) as executor:
                        futures = {
                            executor.submit(
                                fit_each,
                                self,
                                tensor,
                                max_iter,
                                hyper_param,
                                init,
                                seq_len=seq_len,
                                initialize_S_factors=True,
                                valid_tensor=valid_tensor,
                            ): hyper_param
                            for hyper_param in hyper_params
                        }
                        for future in as_completed(futures):
                            arg = futures[future]
                            results[arg] = future.result()
                            pbar.update(1)
            else:
                with ProcessPoolExecutor(
                    max_workers=num_networks,
                    mp_context=multiprocessing.get_context("spawn"),
                ) as executor:
                    futures = {
                        executor.submit(
                            fit_each,
                            self,
                            tensor,
                            max_iter,
                            hyper_param,
                            init,
                            seq_len=seq_len,
                            initialize_S_factors=True,
                            valid_tensor=valid_tensor,
                        ): hyper_param
                        for hyper_param in hyper_params
                    }
                    results = {}
                    for future in as_completed(futures):
                        arg = futures[future]
                        results[arg] = future.result()
        else:
            results = {}
            if self.verbose:
                for hyper_param in tqdm(hyper_params):
                    results[hyper_param] = fit_each(
                        self,
                        tensor,
                        max_iter,
                        hyper_param,
                        init,
                        multi=False,
                        seq_len=seq_len,
                        initialize_S_factors=True,
                        valid_tensor=valid_tensor,
                    )
            else:
                for hyper_param in hyper_params:
                    results[hyper_param] = fit_each(
                        self,
                        tensor,
                        max_iter,
                        hyper_param,
                        init,
                        multi=False,
                        seq_len=seq_len,
                        initialize_S_factors=True,
                        valid_tensor=valid_tensor,
                    )

        if valid_tensor is None:
            metric = "err"
        else:
            metric = "vali_err"

        valid_results = [r for r in results.values() if r[metric] < INF]

        if not valid_results:
            raise RuntimeError(
                "All models failed to converge. Cannot return a valid model."
            )

        sorted_results = sorted(valid_results, key=lambda x: x[metric])

        best_model = None
        best_hyper_param = None

        for res in sorted_results:
            try:
                best_f = fit_each(
                    self,
                    tensor,
                    max_iter,
                    res["hyper_param"],
                    init,
                    dim_list=res["dim_poly"],
                    return_model=True,
                    multi=False,
                    seq_len=seq_len,
                    initialize_S_factors=True,
                )

                candidate_model = best_f["md"]
            except AttributeError:
                print("Re-training failed for this param. Trying next best...")
                continue

            if not isinstance(candidate_model, float):
                best_model = candidate_model
                best_hyper_param = res["hyper_param"]
                # print("Successfully retrieved a valid model.")
                break
            else:
                # print("Re-training failed for this param. Trying next best...")
                pass

        if best_model is None:
            raise RuntimeError(
                "Failed to reconstruct any valid model during re-training."
            )

        print(f"Best params: {best_hyper_param}")

        return best_model, best_hyper_param

    def fit(
        self,
        tensor,
        valid_tensor,
        ts,
        seq_len,
        lf,
        max_iter=50,
        init="random",
        # verbose=0,
        kq=None,
        kl=None,
        lams=[1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3],
        k_sea=None,
        l1_r=1.0,
        l2_r=0.5,
    ):
        dq, dl = tensor.shape[1:]
        self.ts = ts
        self.lf = lf
        self.l2_r = l2_r
        self.l1_r = l1_r

        if kq is None:
            kq_list = np.arange(dq, 1, -1)
        else:
            kq_list = [kq]

        if kl is None:
            kl_list = np.arange((int(dl / 3) // 2) * 2, 2, -4)
        else:
            kl_list = [kl]

        if k_sea is None:
            k_sea_list = [2, 1]
        else:
            k_sea_list = [k_sea]

        if self.ablation_blocking and kq is not None and kl is not None:
            kq_list = np.arange(np.minimum(dq, kq * kl), kq - 1, -1)
            kl_list = [1]

        self.obs_offset = True

        clams = [i for i in lams if i > 0 and i < 1e2]

        hyper_params = itertools.product(
            kq_list,
            kl_list,
            lams,
            clams,
            [True, False],
            k_sea_list,
            repeat=1,
        )
        tic1 = time.time()
        model, hyper_param = self.search_and_fit(
            tensor, max_iter, hyper_params, init, valid_tensor, seq_len
        )
        tic2 = time.time()
        err = model.err(tensor[:seq_len])
        # if self.verbose: print("rmse:", err)
        if err is None:
            model.conv = "inf"
        model.print_result(time=tic2 - tic1)
        # model.print_params()
        return model, hyper_param

    def predict(self, len, ts, mu0=None, shape=None, stabilize=False):
        if mu0 is None:
            mu0 = self.mu0
        if shape is None:
            shape = self.I
        _, Xd_vec = self.gen(T=len, mu0=mu0, stabilize=stabilize)
        Xd = vec_to_tensor(Xd_vec, shape, sequential=True)

        if self.ablation_seasonality:
            Xs = np.zeros(Xd.shape)
        else:
            Xs = self.predict_seasonal_tensor(
                n_sample=len,
                ts=ts,
            )

        pred = Xd + Xs
        pred = np.clip(pred, a_min=0.0, a_max=None)

        return pred, Xd, Xs

    def gen(self, T=None, mu0=None, stabilize=False):
        L = self.L
        kq, kl = self.ranks
        if T is None:
            T = self.T
        if mu0 is None:
            mu0 = self.mu0
        z = np.zeros((T, L))
        z[0] = mu0.copy()

        return _defunc(
            z,
            self.A,
            self.F,
            kronecker(self.C, reverse=True),
            self.b,
            self.d,
            T,
            kq,
            kl,
            self.k_nl,
            self.comb_list,
            stabilize,
        )

    def predict_seasonal_tensor(self, n_sample, ts=0):
        factors = self.S_factors
        n_season = factors[0].shape[0]
        n_fold = n_sample // n_season + 1
        pred = cp_to_tensor((None, factors))
        pred = np.tile(pred, (n_fold, *[1] * (len(factors) - 1)))
        return np.roll(pred, -np.mod(ts, n_season), axis=0)[:n_sample]

    def search(self, tensor, ts, hyper_param):
        """
        Evaluates whether the current model fits the data and determines if a model change (e.g., retraining) is necessary.

        Args:
            tensor (np.ndarray): Tensor of observation data used for evaluation.
            ts (int): Current time step.
            hyper_param (tuple): Hyperparameters to use if re-initialization is needed.

        Returns:
            tuple (bool, float):
                - True if the model needs to be changed, False otherwise.
                - The prediction loss (error) of the current model.
        """
        kq, kl = self.ranks
        k_nl = self.k_nl
        seq_len = len(tensor)
        tensor_norm = tl.norm(tensor, 2)

        # Evaluate reconstruction error.
        try:
            org_recon, _, _ = self.predict(
                len=seq_len, ts=ts, mu0=self.mu0, shape=tensor.shape[1:]
            )
            org_loss = tl.norm(tensor - org_recon, 2) / tensor_norm

            if self.ablation_shift:
                return False, org_loss

            # If the loss exceeds 1.0, we judge that the model does not fit the current situation at all.
            if org_loss > 1.0:
                print("ts=", ts)
                print(f"Reconstruction is bad. (Loss:{org_loss})")
                # Re-initialize the model with safer initial values
                self.initialize(
                    tensor,
                    hyper_param=hyper_param,
                    dim_poly=self.dim_poly,
                    seq_len=seq_len,
                    init="safe",
                    initialize_S_factors=False,
                )
                return True, org_loss

        except Exception as e:
            # Handling errors or divergence during the prediction calculation.
            print("ts=", ts)
            print(f"Prediction diverged or failed: {e}. Requesting model change.")
            if self.ablation_shift:
                return False, INF
            else:
                return True, INF

        # Baseline comparison through temporary model changes
        original_A = self.A.copy()
        original_F = self.F.copy()

        found_better_baseline = False

        try:
            for blk in range(kl):
                i_start = blk * kq
                i_end = (blk + 1) * kq

                j_start = blk * k_nl
                j_end = (blk + 1) * k_nl

                # Set the target block of A to an identity matrix and F to a zero matrix
                self.A[i_start:i_end, i_start:i_end] = np.eye(kq)
                self.F[i_start:i_end, j_start:j_end] = np.zeros((kq, k_nl))

                try:
                    recon, _, _ = self.predict(
                        len=seq_len, ts=ts, mu0=self.mu0, shape=tensor.shape[1:]
                    )
                    loss = tl.norm(tensor - recon, 2) / tensor_norm

                    # If the simplified model has a smaller error than the original complex model:
                    if loss < org_loss:
                        print("ts=", ts)
                        print(
                            f"Block {blk}: Baseline (A=I, F=0) is better (Loss: {loss} < {org_loss})."
                        )
                        # Judge that the current model is inappropriate
                        found_better_baseline = True
                        break

                except Exception:
                    pass

                self.A[i_start:i_end, i_start:i_end] = original_A[
                    i_start:i_end, i_start:i_end
                ]
                self.F[i_start:i_end, j_start:j_end] = original_F[
                    i_start:i_end, j_start:j_end
                ]

        finally:
            self.A = original_A
            self.F = original_F

        if found_better_baseline:
            print("Existing model doesn't fit. Change the model.")
            return True, org_loss

        return False, org_loss
