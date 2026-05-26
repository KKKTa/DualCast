import argparse
import os

from src.model import DualCast
from src.run import run
from src.utils import load_tensor

parser = argparse.ArgumentParser()

parser.add_argument(
    "--exp_seed", type=int, default=-1, help="Random seed for the experiment"
)
parser.add_argument("--suffix", type=str)
parser.add_argument(
    "--dataset",
    type=str,
    help="the name of dataset",
    default="programming",
)
parser.add_argument("--num_works", default=-1, type=int, help="num process")
parser.add_argument(
    "--pred_len", type=int, default=39, help="Prediction sequence length"
)
parser.add_argument("--num_train", type=int, default=208)
parser.add_argument("--num_val", type=int, default=104)
parser.add_argument("--seq_len", type=int, default=104, help="Current window length")
parser.add_argument("--n_season", type=int, default=52, help="Period of seasonality")
parser.add_argument("--start_date", type=str, default="2008-01-01")
parser.add_argument("--end_date", type=str, default="2022-12-31")
parser.add_argument("--init", type=str, default="random")
parser.add_argument("--max_iter", type=int, default=50, help="Number of iterations")
parser.add_argument(
    "--initial_state_cov",
    type=str,
    default="full",
    help="Type of initial state covariance Q0 (diag, full, isotropic)",
)
parser.add_argument(
    "--transition_cov",
    type=str,
    default="full",
    help="Type of transition covariance Q",
)
parser.add_argument(
    "--observation_cov",
    type=str,
    default="full",
    help="Type of observation covariance R",
)
parser.add_argument("--min_lam", type=float, default=1e-2)
parser.add_argument("--max_lam", type=float, default=1e3)
parser.add_argument("--kq", type=int, default=None, help="Number of state dimensions")
parser.add_argument(
    "--kl", type=int, default=None, help="Number of submodels (dynamical systems)"
)
parser.add_argument("--k_sea", type=int, default=None, help="Number of seasonality")

args = parser.parse_args()

start_date = "2008-01-01"
end_date = "2022-12-31"
dataset = args.dataset
init = args.init
initial_state_cov = args.initial_state_cov
transition_cov = args.transition_cov
observation_cov = args.observation_cov

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(PROJECT_ROOT, "datasets", f"{dataset}.csv.gz")

tensor = load_tensor(
    data_path,
    time_key="date",
    facets=["query", "geo"],
    values="volume",
    start_date=start_date,
    end_date=end_date,
    scale=False,
)

model = DualCast(
    initial_state_cov=args.initial_state_cov,
    transition_cov=args.transition_cov,
    observation_cov=args.observation_cov,
    n_season=args.n_season,
    num_works=args.num_works,
    print_log=True,
    verbose=False,
    random_state=args.exp_seed,
)

run(tensor, model, args)
