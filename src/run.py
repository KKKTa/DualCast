import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import tensorly as tl
from frouros.detectors.concept_drift import CUSUM, CUSUMConfig

from .model import fit_each

RESULTS_DIR = Path("./_results")

INF = 1.0e20


def run(tensor, model, args):
    seq_len = args.seq_len
    pred_len = args.pred_len
    max_iter = args.max_iter
    min_lam = args.min_lam
    max_lam = args.max_lam
    lams = []
    val = min_lam
    while val <= max_lam * 1.01:
        lams.append(val)
        val = round(val * 10, 5)
    dt = 4

    folder_path = RESULTS_DIR / args.dataset / args.suffix
    folder_path = Path(str(folder_path))
    folder_path.mkdir(parents=True, exist_ok=True)
    save_path = folder_path / f"seed={str(args.exp_seed)}"
    save_path.mkdir(parents=True, exist_ok=True)

    X_train = tensor[: args.num_train]
    X_valid = tensor[args.num_train : args.num_train + args.num_val]
    test_start_point = args.num_train + args.num_val

    start = time.time()
    best_model, hyper_param = model.fit(
        X_train,
        X_valid,
        ts=0,
        seq_len=seq_len,
        lf=pred_len,
        max_iter=max_iter,
        kq=args.kq,
        kl=args.kl,
        lams=lams,
        k_sea=args.k_sea,
    )
    end = time.time()
    print(f"Grid search finished. Time: {end - start}", flush=True)

    ts = 0
    Xc = tensor[:seq_len]
    shape = tensor.shape

    model = deepcopy(best_model)
    model.save_params(save_path, ts)
    X_hat = np.zeros(shape)
    Xd_hat = np.zeros(shape)
    Xs_hat = np.zeros(shape)
    X_hat[:seq_len], Xd_hat[:seq_len], Xs_hat[:seq_len] = model.predict(
        seq_len, ts, mu0=model.mu0, shape=Xc.shape[1:], stabilize=True
    )

    cusum_config = CUSUMConfig(lambda_=0.1, delta=0.0025, min_num_instances=2)
    cusum_detector = CUSUM(config=cusum_config)

    preds, trues, ct = [], [], []
    loss_hist = []

    for t in range(seq_len + dt, len(tensor) - pred_len, dt):
        ts = t - seq_len
        if test_start_point <= t < test_start_point + dt:
            start = time.time()
            print(f"start the timer :ts={t - seq_len}, te={t}")
        # if self.verbose:
        #     print("\n")
        #     print("------------------------------------------------")
        #     print("current window:", "ts=", t - seq_len, "te=", t)

        Xc = tensor[ts:t]
        true = tensor[t : t + pred_len]

        start_one_batch = time.time()

        model.update(Xc, ts, seq_len, dt)

        flag, org_loss = model.search(Xc, ts, hyper_param)
        loss_hist.append(org_loss)
        if not flag:
            cusum_detector.update(value=org_loss)

            if cusum_detector.drift:
                print(f"ts={ts}")
                print(
                    f"CUSUM detected drift (Sum: {cusum_detector.sum_:.4f} > {cusum_detector.config.lambda_})",
                    flush=True,
                )
                flag = True

        if flag:
            current_lam = hyper_param[2]
            raw_candidates = [current_lam * 0.1, current_lam, current_lam * 10.0]
            clipped_candidates = [np.clip(c, min_lam, max_lam) for c in raw_candidates]
            candidate_lams = []
            for c in clipped_candidates:
                if not any(np.isclose(c, existing_c) for existing_c in candidate_lams):
                    candidate_lams.append(c)

            best_candidate = None
            best_loss = INF
            best_param = None

            tensor_norm = tl.norm(Xc, 2)

            for lam in candidate_lams:
                try:
                    temp_param = list(hyper_param)
                    temp_param[2] = lam
                    temp_param = tuple(temp_param)

                    result = fit_each(
                        deepcopy(model),
                        Xc,
                        max_iter=max_iter,
                        hyper_param=temp_param,
                        init="random",
                        return_model=True,
                        multi=False,
                        seq_len=seq_len,
                        ts=ts,
                    )
                    candidate = deepcopy(result["md"])

                    if isinstance(candidate, float):
                        continue

                    pred_c, _, _ = candidate.predict(
                        seq_len + pred_len, ts, mu0=candidate.mu0, shape=Xc.shape[1:]
                    )
                    recon, _ = np.split(pred_c, [seq_len])
                    loss = tl.norm(Xc - recon, 2) / tensor_norm

                    if loss < best_loss:
                        best_loss = loss
                        best_candidate = candidate
                        best_param = temp_param

                except Exception as e:
                    print(f"Failed training for lambda={lam}: {e}\n", flush=True)
                    pass

            if best_candidate is not None:
                if best_loss >= 1.0:
                    print(
                        f"Best loss is over 1.0; {best_loss}. Use existing model.\n",
                        flush=True,
                    )
                elif org_loss > best_loss:
                    model = deepcopy(best_candidate)
                    hyper_param = best_param
                    model.save_params(save_path, ts)
                    print(
                        f"Switch the model at ts={ts} (lam={hyper_param[2]}, loss={best_loss}, iter={model.iter})\n",
                        flush=True,
                    )
                else:
                    print(
                        f"New model is worse than existing model. (new:{best_loss} vs old:{org_loss})\n",
                        flush=True,
                    )
            else:
                print(f"All candidates failed training at ts={ts}\n", flush=True)
            cusum_detector.reset()

        pred, Xd, Xs = model.predict(
            seq_len + pred_len, ts, mu0=model.mu0, shape=Xc.shape[1:], stabilize=True
        )
        Xc_hat, pred = np.split(pred, [seq_len])

        end_one_batch = time.time()

        X_hat[ts:t] = Xc_hat
        Xd_hat[ts:t] = Xd[:seq_len]
        Xs_hat[ts:t] = Xs[:seq_len]

        if t >= test_start_point:
            preds.append(pred.reshape(pred.shape[0], -1))
            trues.append(true.reshape(true.shape[0], -1))
            ct.append(end_one_batch - start_one_batch)

    model.update(tensor[ts + dt : t + dt], ts + dt, seq_len, dt, print_log=True)
    X_hat[ts + dt : t + dt], Xd_hat[ts + dt : t + dt], Xs_hat[ts + dt : t + dt] = (
        model.predict(
            seq_len, ts + dt, mu0=model.mu0, shape=Xc.shape[1:], stabilize=True
        )
    )

    preds = np.array(preds)
    trues = np.array(trues)

    print("test shape:", preds.shape, trues.shape)

    np.save(save_path / "X.npy", tensor)
    np.save(save_path / "X_hat.npy", X_hat)
    np.save(save_path / "Xd_hat.npy", Xd_hat)
    np.save(save_path / "Xs_hat.npy", Xs_hat)
    np.save(save_path / "loss_hist.npy", np.array(loss_hist))
    np.save(save_path / "preds.npy", preds)
    np.save(save_path / "trues.npy", trues)

    end = time.time()
    exp_time = end - start
    print(f"total_time(s):{exp_time:.3f}")
    print("====== Experiment finished ======")

    return
