#!/bin/env python3
import itertools as it
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorly as tl
from numba import njit
from numba.typed import List as TypedList
from numpy import trace
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.tsa.seasonal import STL
from tensorly.decomposition import (
    non_negative_parafac,
    parafac,
    tucker,
)

INF = 1.0e20
h = 1.0e-5


def load_tensor(
    path,
    time_key,
    facets,
    values=None,
    sampling_rate="D",
    start_date=None,
    end_date=None,
    scale=True,
):
    df = pd.read_csv(path)
    tensor = df2tts(
        df,
        time_key=time_key,
        facets=facets,
        values=values,
        sampling_rate=sampling_rate,
        start_date=start_date,
        end_date=end_date,
    )

    for key in facets:
        print(sorted(list(set(df[key]))), flush=True)

    if scale == True:
        tensor = min_max_scale_tensor(tensor)

    return tensor


def min_max_scale_np(array):
    min = array.min()
    max = array.max()
    array = (array - min) / (max - min)
    return array


def min_max_scale_tensor(data):
    query_size = data.shape[1]
    geo_size = data.shape[2]
    ret = np.zeros(shape=data.shape)
    for i in range(query_size):
        for j in range(geo_size):
            ret[:, i, j] = min_max_scale_np(data[:, i, j])
    return ret


def df2tts(
    df,
    time_key,
    facets,
    values=None,
    sampling_rate="D",
    start_date=None,
    end_date=None,
):
    """Convert a DataFrame (list) to tensor time series

    df (pandas.DataFrame):
        A list of discrete events
    time_key (str):
        A column name of timestamps
    facets (list):
        A list of column names to make tensor timeseries
    values (str):
        A column name of target values (optional)
    sampling_rate (str):
        A frequancy for resampling, e.g., "7D", "12H", "H"
    """
    df[time_key] = pd.to_datetime(df[time_key])
    if start_date is not None:
        df = df[lambda x: x[time_key] >= pd.to_datetime(start_date)]
    if end_date is not None:
        df = df[lambda x: x[time_key] <= pd.to_datetime(end_date)]
    tmp = df.copy(deep=True)
    shape = tmp[facets].nunique().tolist()
    if values == None:
        values = "count"
        tmp[values] = 1
    tmp[time_key] = tmp[time_key].dt.round(sampling_rate)
    print("Tensor:")
    print(tmp.nunique()[[time_key] + facets])

    grouped = tmp.groupby([time_key] + facets).sum()[[values]]
    grouped = grouped.unstack(fill_value=0).stack()
    grouped = grouped.pivot_table(
        index=time_key,
        columns=facets,
        values=values,
        fill_value=0,
    )

    tts = grouped.values
    tts = np.reshape(tts, (-1, *shape))
    return tts


def unfolding_dot_khatri_rao(tensor, factors, mode):
    return tl.cp_tensor.unfolding_dot_khatri_rao(tensor, (None, factors), mode)


def compute_accum(factors, skip_matrix):
    accum = 1.0
    for i, factor in enumerate(factors):
        if i == skip_matrix:
            continue
        accum *= tl.dot(tl.transpose(factor), factor)

    return accum


def STL_decomp(tensor, period):
    trend = np.zeros(shape=tensor.shape)
    seasonal = np.zeros(shape=tensor.shape)
    resid = np.zeros(shape=tensor.shape)

    for i in range(tensor.shape[1]):
        for j in range(tensor.shape[2]):
            stl = STL(tensor[:, i, j], robust=True, period=period)
            stl_series = stl.fit()
            trend[:, i, j] = stl_series.trend
            seasonal[:, i, j] = stl_series.seasonal
            resid[:, i, j] = stl_series.resid

    return trend, seasonal, resid


def compute_mean_tensor(tensor, period, t=0, remove_temporal_mean=True):
    n_sample = tensor.shape[0]
    n_dims = tensor.shape[1:]
    mean_tensor = np.zeros((period, *n_dims))

    n_section = n_sample // period
    season_ids = np.arange(t, t + n_sample, 1) % period
    diff_ids = n_sample - period * n_section
    start_point = np.where(season_ids == 0)[0][0]

    rolled_tensor = np.roll(tensor, -start_point, axis=0)
    if diff_ids > 1:
        rolled_tensor = rolled_tensor[:-diff_ids]

    for w in range(n_section):
        one_period = rolled_tensor[w * period : (w + 1) * period]
        mean_tensor += one_period
        if remove_temporal_mean:
            mean_tensor -= one_period.mean(axis=0)

    mean_tensor /= n_section

    return mean_tensor


def init_seasonal_factors(
    tensor,
    rank,
    period,
    t=0,
    n_iter_max=100,
    tol=1e-8,
    non_negative=False,
    random_state=None,
):
    if period == 0:
        return None

    Xd, Xs, _ = STL_decomp(tensor, period=int(period / 2))
    mean_tensor = compute_mean_tensor(Xs, period, t)

    if non_negative:
        _, factors = non_negative_parafac(
            mean_tensor,
            rank,
            n_iter_max=n_iter_max,
            tol=tol,
            random_state=random_state,
        )
    else:
        _, factors = parafac(
            mean_tensor,
            rank,
            n_iter_max=n_iter_max,
            tol=tol,
            random_state=random_state,
        )

    return factors


def tucker_decomp(tensor, ranks, random_state, init="random"):
    time_len = tensor.shape[0]
    if time_len != ranks[0]:
        if time_len in ranks:
            ranks.remove(time_len)
            ranks.insert(0, time_len)
        else:
            raise ValueError("time length does not exist in ranks.")
    core, factors = tucker(
        tensor, ranks, n_iter_max=100, init=init, random_state=random_state
    )
    return core, [factors[0], factors[1], factors[2]]


def vec_to_tensor(vector, ranks, sequential=False):
    if sequential:
        return np.array([vec_to_tensor(vt, ranks) for vt in vector])
    return vector.reshape(ranks, order="F")


def tensor_to_vec(tensor, sequential=False):
    if sequential:
        return np.array([tensor_to_vec(tt) for tt in tensor])
    return tensor.flatten("F")


@njit(cache=True)
def vec_to_factors(vector, I, J):
    """
    Given:
    - vector: np.array
    - I: list object
    - J: list object

    Convert vector to the list of matrices,
    each of which shape is i x j in [I, J].
    """
    sizes = I * J
    index = []

    for m in range(len(I)):
        index.append(np.arange(sizes[m]))
        if m > 0:
            index[-1] += sum(sizes[:m])

    factors = TypedList()
    for m in range(len(I)):
        val = vector[index[m]].reshape((J[m], I[m])).T
        factors.append(np.ascontiguousarray(val))

    return factors


@njit(cache=True)
def factors_to_vec(factors):
    total_size = 0
    for i in range(len(factors)):
        total_size += factors[i].size

    vec = np.empty(total_size, dtype=np.float64)

    current_idx = 0
    for i in range(len(factors)):
        flat_item = factors[i].T.flatten()
        size = flat_item.size

        vec[current_idx : current_idx + size] = flat_item
        current_idx += size

    return vec


@njit(cache=True)
def mat_to_tensor(matrix, shape):
    tensor = matrix.T.flatten().reshape(shape[::-1])
    tensor = tensor.transpose(3, 2, 1, 0)
    return tensor


@njit(cache=True)
def tensor_to_mat(tensor):
    M = tensor.ndim
    I = list(tensor.shape)

    if M % 2 == 1:
        M += 1
        I.append(1)

    J = np.array(I)
    r = np.prod(J[: int(M / 2)])
    c = np.prod(J[(np.arange(M / 2) + M / 2).astype(np.int64)])

    return tensor.T.flatten().reshape((c, r)).T


@njit(cache=True)
def _kronecker(matrices, skip_matrix=-1, reverse=False):
    result = np.array([[1.0]])

    num_matrices = len(matrices)

    if reverse:
        for i in range(num_matrices - 1, -1, -1):
            if i == skip_matrix:
                continue
            result = np.kron(matrices[i], result)
    else:
        for i in range(num_matrices):
            if i == skip_matrix:
                continue
            result = np.kron(result, matrices[i])

    return result


def kronecker(matrices, skip_matrix=-1, reverse=False):
    if not isinstance(matrices, TypedList):
        typed_matrices = TypedList()
        for m in matrices:
            typed_matrices.append(np.ascontiguousarray(m))
    else:
        typed_matrices = matrices
    return _kronecker(typed_matrices, skip_matrix, reverse)


def update_multilinear_operator(B, omg, psi, phi, ups, clam, cov_type):
    if not isinstance(B, TypedList):
        B_typed = TypedList()
        for item in B:
            B_typed.append(np.ascontiguousarray(item))
        B = B_typed

    M = len(B)
    I = np.array(
        [B[m].shape[0] for m in range(M)],
    )
    J = np.array(
        [B[m].shape[1] for m in range(M)],
    )

    if cov_type == "full":
        omg = (omg + omg.T) / 2
    elif cov_type == "diag":
        omg = np.diag(omg)
    elif cov_type == "isotropic":
        omg = np.eye(np.prod(I)) * omg
    psi = (psi + psi.T) / 2

    shape = tuple(int(x) for x in (*I, *J))

    vecB = factors_to_vec(B)
    newB = descend(vecB, omg, psi, phi, ups, clam, I, J, shape)
    newB_typed = vec_to_factors(newB, I, J)
    newB = list(newB_typed)

    return newB


@njit(cache=True)
def descend(
    x0,
    omg,
    psi,
    phi,
    ups,
    clam,
    I,
    J,
    shape,
    epsilon=1.0e-10,
    maxiter=1e3,
):
    x_acc = x0.copy()
    x_p = x0.copy()
    x_iter = x0.copy()
    fx = np.inf
    fx_p = -np.inf

    l1_clam = clam
    l2_clam = 0.5 * clam
    l2_clam_step = clam
    l_k = np.sqrt(np.sum(np.power(omg, 2)) * np.sum(np.power(psi, 2))) + l2_clam_step
    l1_clam_step = l1_clam / l_k

    M = len(I)
    I_rests = (np.prod(I) / I).astype(np.int64)
    J_rests = (np.prod(J) / J).astype(np.int64)

    f_p = _f(x_p, omg, psi, phi, ups, l1_clam, l2_clam, I, J)

    for iter in range(maxiter):
        grad = _df(
            x_p,
            omg,
            psi,
            phi,
            ups,
            l2_clam_step,
            M,
            I,
            J,
            I_rests,
            J_rests,
            shape,
        )
        x_iter = soft_threshold(x_acc - grad / l_k, l1_clam_step)

        f = _f(x_iter, omg, psi, phi, ups, l1_clam, l2_clam, I, J)
        conv2 = abs(f - f_p)

        if conv2 < epsilon:
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

    return x_iter


@njit(cache=True)
def _f(z, omg, psi, phi, ups, l1_clam, l2_clam, I, J):
    matB = _kronecker(vec_to_factors(z, I, J), reverse=True)

    tmp = (phi - ups) @ matB.T
    tmp = matB @ psi @ matB.T - tmp - tmp.T

    f = trace(omg @ tmp)
    f += l1(l1_clam, z) + l2(l2_clam, z)

    return f


@njit(cache=True)
def _df(z, omg, psi, phi, ups, l2_clam_step, M, I, J, I_rests, J_rests, shape):
    IJ = I * J
    B = vec_to_factors(z, I, J)
    matB = _kronecker(B, reverse=True)

    H = psi @ matB.T - phi.T + ups.T
    H = H @ omg
    H = np.ascontiguousarray(H.T)

    g = np.zeros(np.sum(IJ))

    for m in range(M):
        F = _kronecker(
            B,
            skip_matrix=m,
            reverse=True,
        )
        G = np.zeros((I[m], J[m]))

        Hm = mat_to_tensor(H, shape)
        Hm = transpose_Hm(Hm, M, m)
        Hm = tensor_to_mat(Hm)

        Im = range(I[m])
        Jm = range(J[m])

        for i in Im:
            for j in Jm:
                r = np.arange(I_rests[m]) + I_rests[m] * i
                c = np.arange(J_rests[m]) + J_rests[m] * j
                G[i, j] = np.sum(F * Hm[r, :][:, c])

        G += l2_clam_step * B[m]
        index = np.arange(IJ[m])
        if m > 0:
            index += np.sum(IJ[:m])
        g[index] = 2 * G.T.flatten()

    return g


@njit(cache=True)
def soft_threshold(x, alpha):
    """
    Standard Soft Thresholding for L1 Regularization (Lasso).
    sign(x) * max(|x| - alpha, 0)
    """
    # alpha以下の値を0にし、それ以外をalpha分だけ原点方向に縮小する
    return np.sign(x) * np.maximum(np.abs(x) - alpha, 0.0)


@njit(cache=True)
def transpose_Hm(Hm, M, m):
    if M == 2:
        if m == 0:
            axes = (1, 0, 3, 2)
        else:  # m == 1
            axes = (0, 1, 2, 3)
    # elif M == 3:
    #     if m == 0:
    #         axes = (1, 2, 0, 4, 5, 3)
    #     elif m == 1:
    #         axes = (0, 2, 1, 3, 5, 4)
    #     else:  # m == 2
    #         axes = (0, 1, 2, 3, 4, 5)
    else:
        raise ValueError("Unsupported M")
    return np.transpose(Hm, axes)


def get_k_nl(k, dim):
    if dim == 1:
        return 0, np.zeros((1, 1), dtype="i8")
    sta = np.arange(k + 1)

    comb_iterator = it.combinations(sta, dim)

    comb_list = np.array(
        list(comb_iterator)[k:],
        dtype="i8",
    )

    return len(comb_list), comb_list


def make_feature_names(k, dim, features=None):
    if features is None:
        features = [f"s_{i}" for i in range(k)]
    poly = PolynomialFeatures(degree=dim, include_bias=False, interaction_only=True)
    poly.fit(np.zeros((1, k)))
    names = poly.get_feature_names_out(features)
    return ["$" + s + "$" for s in names]


@njit(cache=True)
def make_state_vec(sta, kq, kl, k_nl, comb_list):
    nonlinear_terms = np.zeros(k_nl * kl)
    if k_nl == 0:
        return nonlinear_terms
    for b in range(kl):
        tmp = np.append(1.0, sta[kq * b : kq * (b + 1)])
        nonlinear_terms[k_nl * b : k_nl * (b + 1)] = np.array(
            [np.prod(tmp[cmt]) for cmt in comb_list]
        )
    return nonlinear_terms


# @njit(cache=True)
def rel_norm(new, old, eps=1e-12):
    num = np.linalg.norm(new - old, ord="fro")
    den = np.linalg.norm(old, ord="fro") + eps
    return num / den


@njit(cache=True)
def l1(lam, params):
    return np.sum(np.abs(lam * params))


@njit(cache=True)
def l2(lam, params):
    return np.sum(np.power(lam * params, 2))


@njit(cache=True)
def moment(Ez, P, kq, kl, ks, k_d2, k_d3, k_nl):
    Ezz = P + np.outer(Ez, Ez)
    Eznl = np.zeros(
        k_nl * kl,
    )
    k3 = ks[0]
    k4 = ks[1]

    for b in range(kl):
        idx_offset = b * kq
        nl_offset = b * k_nl
        d0 = 0
        d1 = 0
        d2 = 0

        for i in range(kq):
            I = idx_offset + i

            for j in range(i + 1, kq):
                J = idx_offset + j
                Eznl[nl_offset + d0] = Ezz[I][J]
                d0 += 1

                for l in range(j + 1, k3):
                    L = idx_offset + l
                    Eznl[nl_offset + k_d2 + d1] = (
                        Ez[I] * Ez[J] * Ez[L]
                        + Ez[I] * P[J][L]
                        + Ez[J] * P[I][L]
                        + Ez[L] * P[I][J]
                    )
                    d1 += 1

                    for m in range(l + 1, k4):
                        M = idx_offset + m
                        Eznl[nl_offset + k_d2 + k_d3 + d2] = (
                            Ez[I]
                            * (
                                Ez[J] * (Ez[L] * Ez[M] + P[L][M])
                                + Ez[L] * P[J][M]
                                + Ez[M] * P[J][L]
                            )
                            + P[I][J] * (Ez[L] * Ez[M] + P[M][L])
                            + P[I][M] * (Ez[J] * Ez[L] + P[J][L])
                            + P[I][L] * (P[J][M] + Ez[J] * Ez[M])
                        )
                        d2 += 1

    return Ezz, Eznl


@njit(cache=True)
def exact_Szznl(mu, W, kq, k_nl, comb_list):
    L_aug = kq + k_nl
    S = np.zeros((L_aug, L_aug))

    for i in range(kq):
        for j in range(kq):
            S[i, j] = mu[i] * mu[j] + W[i, j]

    if k_nl == 0:
        return S

    for i in range(kq):
        for j_idx in range(k_nl):
            j = kq + j_idx
            c1 = comb_list[j_idx][0] - 1
            c2 = comb_list[j_idx][1] - 1

            val = (
                mu[i] * mu[c1] * mu[c2]
                + mu[i] * W[c1, c2]
                + mu[c1] * W[i, c2]
                + mu[c2] * W[i, c1]
            )
            S[i, j] = val
            S[j, i] = val

    for i_idx in range(k_nl):
        i = kq + i_idx
        a = comb_list[i_idx][0] - 1
        b = comb_list[i_idx][1] - 1

        for j_idx in range(k_nl):
            j = kq + j_idx
            c = comb_list[j_idx][0] - 1
            d = comb_list[j_idx][1] - 1

            val = (
                mu[a] * mu[b] * mu[c] * mu[d]
                + mu[a] * mu[b] * W[c, d]
                + mu[a] * mu[c] * W[b, d]
                + mu[a] * mu[d] * W[b, c]
                + mu[b] * mu[c] * W[a, d]
                + mu[b] * mu[d] * W[a, c]
                + mu[c] * mu[d] * W[a, b]
                + W[a, b] * W[c, d]
                + W[a, c] * W[b, d]
                + W[a, d] * W[b, c]
            )
            S[i, j] = val

    return S


@njit(cache=True)
def exact_Sz1znl(mu_next, mu_curr, W_curr, C_lag, kq, k_nl, comb_list):
    L_aug = kq + k_nl
    S = np.zeros((kq, L_aug))

    for k in range(kq):
        for m in range(kq):
            S[k, m] = mu_next[k] * mu_curr[m] + C_lag[k, m]

    if k_nl == 0:
        return S

    for k in range(kq):
        for m_idx in range(k_nl):
            m = kq + m_idx
            i = comb_list[m_idx][0] - 1
            j = comb_list[m_idx][1] - 1

            val = (
                mu_next[k] * mu_curr[i] * mu_curr[j]
                + mu_next[k] * W_curr[i, j]
                + mu_curr[i] * C_lag[k, j]
                + mu_curr[j] * C_lag[k, i]
            )
            S[k, m] = val

    return S


@njit(cache=True)
def extract_diag_blocks(M, k1, k2, kl):
    blocks = np.zeros((k1, k2, kl), dtype=M.dtype)

    for b in range(kl):
        blocks[:, :, b] = M[k1 * b : k1 * (b + 1), k2 * b : k2 * (b + 1)]

    return blocks


@njit(cache=True)
def diff_m(z, j_delta, kq, kl, k_nl, comb_list):
    f_delta = j_delta + z
    b_delta = z - j_delta
    L = len(z)
    jacobian = np.zeros((k_nl * kl, L))
    for i in range(L):
        jacobian[:, i] = make_state_vec(
            f_delta[i], kq, kl, k_nl, comb_list
        ) - make_state_vec(b_delta[i], kq, kl, k_nl, comb_list)

    return jacobian


@njit(cache=True)
def jacobian(z, A, F, j_delta, kq, kl, k_nl, comb_list):
    f_delta = j_delta + z
    b_delta = z - j_delta
    L = len(z)
    jacobian = np.zeros((k_nl * kl, L))
    for i in range(L):
        jacobian[:, i] = make_state_vec(
            f_delta[i], kq, kl, k_nl, comb_list
        ) - make_state_vec(b_delta[i], kq, kl, k_nl, comb_list)
    return F @ np.ascontiguousarray(jacobian) / (2 * h) + A


def make_result(model, hyper_param, dim_poly, tensor, loss):
    result = {"hyper_param": hyper_param}
    result["dim_poly"] = [dim_poly]

    if isinstance(model, float):
        result["loss"] = INF
        result["err"] = INF
        return result

    result["loss"] = loss
    result["err"] = model.err(tensor)

    if result["err"] is None:
        result["err"] = INF
    return result


def plot_result(model, data, dataset, PROJECT_ROOT, fsize=3.3, missing=None):
    dataset = dataset
    # if setting['xticklabels'] is None:
    #     xticklabels = make_feature_names(model.k, model.dim_poly)
    # else:
    #     xticklabels = setting['xticklabels']

    # if setting['yticklabels'] is None:
    #     yticklabels = xticklabels[:model.k]
    # else:
    #     yticklabels = setting['yticklabels']
    plt.rcParams["axes.xmargin"] = 0
    annot = False
    fig_type = "jpg"
    dir_path = os.path.join(PROJECT_ROOT, "result", model.fit_type, dataset)
    os.makedirs(dir_path, exist_ok=True)

    w = np.concatenate(((model.A - np.eye(model.L)), model.F), axis=1)

    size = 20
    vmin = np.min(w) - 0.1
    vmax = np.max(w) + 0.1

    plt.rcParams["font.size"] = 24
    plt.rcParams["mathtext.fontset"] = "cm"
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        gridspec_kw=dict(
            width_ratios=[1, 3],
            height_ratios=[1],
            wspace=0.1,
            hspace=0.3,
        ),
        figsize=(fsize * 4, 0.7 * 4),
    )
    sns.heatmap(
        w[:, : model.L],
        vmin=vmin,
        vmax=vmax,
        cmap="coolwarm",
        fmt="1.1e",
        center=0.0,
        cbar=None,
        ax=ax1,
        annot=annot,
    )
    ax1.tick_params(axis="x", labelrotation=30, labelsize=size)
    ax1.tick_params(axis="y", labelrotation=0, labelsize=size)
    ax1.tick_params(pad=0.5)

    sns.heatmap(
        w[:, model.L :],
        vmin=vmin,
        vmax=vmax,
        cmap="coolwarm",
        fmt="1.1e",
        center=0.0,
        ax=ax2,
        annot=annot,
    )
    ax2.tick_params(axis="x", labelrotation=30, labelsize=size)
    ax2.tick_params(axis="y", labelrotation=0, labelsize=size)
    ax2.tick_params(pad=0.5)
    cbar = ax2.collections[0].colorbar
    cbar.ax.tick_params(labelsize=13)
    fig.savefig(os.path.join(dir_path, f"st_weight_3_1.{fig_type}"))

    plt.rcParams["font.size"] = 16
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.figure(figsize=(0.7, fsize))
    sns.heatmap(model.A, cmap="coolwarm", fmt="1.1e", center=0.0, square=True)
    plt.xlabel("State")
    plt.savefig(os.path.join(dir_path, f"A.{fig_type}"), pad_inches=0.1)

    plt.rcParams["font.size"] = 16
    plt.rcParams["mathtext.fontset"] = "cm"
    for i in range(len(model.C)):
        plt.figure(figsize=(0.7, fsize))
        sns.heatmap(model.C[i], cmap="coolwarm", fmt="1.1e", center=0.0, square=True)
        plt.xlabel("State")
        plt.savefig(os.path.join(dir_path, f"C{i!s}.{fig_type}"), pad_inches=0.1)

    for i in range(6):
        plt.rcParams["font.size"] = 28
        plt.figure(figsize=(15, 3))
        # print(f"data:{np.max(data[:,:,-3])}")
        # print(f"Obs:{np.max(model.Obs[:,:,-3])}")
        plt.plot(data[:, :, i], color="lightgrey", linewidth=3)
        plt.plot(model.Obs[:, :, i], linewidth=3)
        plt.savefig(os.path.join(dir_path, f"fitting{i}.{fig_type}"), pad_inches=0.1)

    plt.rcParams["font.size"] = 10
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    for i in range(model.L):
        ax.plot(model.Ez[:, i], zorder=2, label=f"$s_{i}$")
    ax.set_xlabel("Time", fontsize=10)
    ax.set_ylabel("Value", fontsize=10)
    # fig.legend(loc='center', bbox_to_anchor=(.5, 1.1), ncol=4, fontsize=32)
    fig.savefig(
        os.path.join(dir_path, f"smoothed_latent_dynamics.{fig_type}"),
        pad_inches=0.1,
    )

    # with open(f'{dir_path}/model.pickle', mode='wb') as f:
    #     pickle.dump(model, f)
