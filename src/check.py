import time
from pathlib import Path

import numpy as np

RESULTS_DIR = Path("./_results")

INF = 1.0e20


def run(tensor, model, args, ablation_seasonality=False):
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

    folder_path = RESULTS_DIR / args.dataset / args.suffix
    if ablation_seasonality:
        folder_path = f"{folder_path}_seasonality"
    folder_path = Path(str(folder_path))
    folder_path.mkdir(parents=True, exist_ok=True)
    save_path = folder_path / f"seed={str(args.exp_seed)}"
    save_path.mkdir(parents=True, exist_ok=True)

    ts = 0
    Xc = tensor[:seq_len]
    shape = tensor.shape

    X_hat = np.zeros(shape)
    Xd_hat = np.zeros(shape)
    Xs_hat = np.zeros(shape)

    start = time.time()

    for t in range(seq_len, len(tensor) - pred_len - args.num_val, seq_len):
        ts = t - seq_len
        # if self.verbose:
        #     print("\n")
        #     print("------------------------------------------------")
        #     print("current window:", "ts=", t - seq_len, "te=", t)

        Xc = tensor[ts:t]
        X_valid = tensor[t : t + args.num_val]

        start_one_batch = time.time()

        best_model, hyper_param = model.fit(
            Xc,
            X_valid,
            ts=ts,
            seq_len=seq_len,
            lf=pred_len,
            max_iter=max_iter,
            kq=args.kq,
            kl=args.kl,
            lams=lams,
            k_sea=args.k_sea,
        )
        end_one_batch = time.time()
        print(f"({ts} : {t}) Time: {end_one_batch - start_one_batch}", flush=True)

        best_model.save_params(save_path, ts)

        pred, Xd, Xs = best_model.predict(
            seq_len + pred_len,
            ts,
            mu0=best_model.mu0,
            shape=Xc.shape[1:],
            stabilize=True,
        )
        Xc_hat, pred = np.split(pred, [seq_len])

        end_one_batch = time.time()

        X_hat[ts:t] = Xc_hat
        Xd_hat[ts:t] = Xd[:seq_len]
        Xs_hat[ts:t] = Xs[:seq_len]

        recon_path = save_path / f"model/ts={str(ts)}"
        recon_path.mkdir(parents=True, exist_ok=True)

        np.save(recon_path / "Xc.npy", Xc)
        np.save(recon_path / "Xc_hat.npy", Xc_hat)
        np.save(recon_path / "Xd_hat.npy", Xd[:seq_len])
        np.save(recon_path / "res.npy", Xc - Xc_hat)

    np.save(save_path / "X.npy", tensor)
    np.save(save_path / "X_hat.npy", X_hat)
    np.save(save_path / "Xd_hat.npy", Xd_hat)
    np.save(save_path / "Xs_hat.npy", Xs_hat)
    np.save(save_path / "res.npy", tensor - X_hat)

    end = time.time()
    exp_time = end - start
    print(f"total_time(s):{exp_time:.3f}")
    print("====== Experiment finished ======")

    return
