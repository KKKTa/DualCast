#!/bin/bash
exp_seed=42
initial_state_cov='full'
transition_cov='full'
observation_cov='full'
min_lam=1e-1 # 1e-2
max_lam=1e1 # 1e3
kq=3 # None
kl=4 # None
ks=4 # None
n_season=52
seq_len=104 
pred_len=39
total_len=782
num_train=208
num_val=104
max_iter=50
train_rate=$(echo "scale=30; $num_train / $total_len" | bc)
test_rate=$(echo "scale=30; ($total_len - $num_train - $num_val) / $total_len" | bc)
init="random"

num_works=-1

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
PYTHON_SCRIPT="$PROJECT_ROOT/main.py"

dataset=$1

cd "$PROJECT_ROOT"

today=$(TZ=UTC-9 date '+%Y%m%d')
mkdir -p ./logs/${today}
logfn="./logs/${today}/${dataset}_${len}.out"

# nohup poetry run python "$PYTHON_SCRIPT" --dataset $dataset \
#                --exp_seed $exp_seed \
#                --seq_len $seq_len \
#                --pred_len $pred_len \
#                --num_train $num_train \
#                --num_val $num_val \
#                --num_works $num_works \
#                --initial_state_cov $initial_state_cov \
#                --transition_cov $transition_cov \
#                --observation_cov $observation_cov \
#                --min_lam $min_lam \
#                --max_lam $max_lam \
#                --kq $kq \
#                --kl $kl \
#                --k_sea $ks \
#                --n_season $n_season \
#                --init $init \
#                --max_iter $max_iter \
#                --suffix $today \
#                > $logfn 2>&1 

poetry run python "$PYTHON_SCRIPT" --dataset $dataset \
               --exp_seed $exp_seed \
               --seq_len $seq_len \
               --pred_len $pred_len \
               --num_train $num_train \
               --num_val $num_val \
               --num_works $num_works \
               --initial_state_cov $initial_state_cov \
               --transition_cov $transition_cov \
               --observation_cov $observation_cov \
               --min_lam $min_lam \
               --max_lam $max_lam \
               --kq $kq \
               --kl $kl \
               --k_sea $ks \
               --n_season $n_season \
               --init $init \
               --max_iter $max_iter \
               --suffix $today \
               > $logfn 2>&1 