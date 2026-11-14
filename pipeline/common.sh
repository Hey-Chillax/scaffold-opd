# shellcheck shell=bash disable=SC2034  # DEFAULT_* and REPO_ROOT are read by the scripts that source this file
# Copyright 2026 Yi Zhang
# SPDX-License-Identifier: Apache-2.0
#
# Shared helpers for the scripts in pipeline/. Source it; it only sets variables and defines functions.
#
# Locations every pipeline script understands (all optional; defaults in brackets):
#   MODEL_DIR    [$REPO_ROOT/models]   holds Qwen3-1.7B/ and Qwen3-4B-Instruct-2507/
#   DATA_DIR     [$REPO_ROOT/data]     train/ and bench/, written by pipeline/prepare_data.py
#   OUTPUT_ROOT  [$REPO_ROOT/outputs]  run directories: logs, rollouts, W&B runs, evaluations
#   CKPT_DIR     [the run directory]    where a training run writes its checkpoints
#   VENV         [$REPO_ROOT/.venv]    activated when it exists
#   ENV_SETUP    [unset]                a file sourced first (module loads, cache locations, proxies, ...)
#   DRY_RUN      [0]                    1 parses everything and prints each command instead of running it

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
MODEL_DIR=${MODEL_DIR:-$REPO_ROOT/models}
DATA_DIR=${DATA_DIR:-$REPO_ROOT/data}
OUTPUT_ROOT=${OUTPUT_ROOT:-$REPO_ROOT/outputs}
DEFAULT_STUDENT=$MODEL_DIR/Qwen3-1.7B
DEFAULT_TEACHER=$MODEL_DIR/Qwen3-4B-Instruct-2507
DEFAULT_TRAIN_DATA=$DATA_DIR/train/dapo_math_17k_en.parquet
DEFAULT_BENCH_DIR=$DATA_DIR/bench

log() { echo "[${LOG_TAG:-scaffold-opd}] $(date '+%F %T') $*"; }

die() {
    log "ERROR: $*" >&2
    exit 1
}

is_dry_run() { [[ "${DRY_RUN:-0}" == 1 ]]; }

# Run a command, or only print it (shell-quoted) when DRY_RUN=1.
run() {
    if is_dry_run; then
        printf '[dry-run]'
        printf ' %q' "$@"
        printf '\n'
        return 0
    fi
    "$@"
}

# Print the exported variables a launcher reads, so a dry run shows the complete configuration.
show_env() {
    is_dry_run || return 0
    local name
    for name in "$@"; do
        printf '[dry-run]   %s=%q\n' "$name" "${!name-}"
    done
}

setup_env() {
    if [[ -n "${ENV_SETUP:-}" ]]; then
        # shellcheck disable=SC1090
        source "$ENV_SETUP"
    fi
    local venv=${VENV:-$REPO_ROOT/.venv}
    if [[ -f "$venv/bin/activate" ]]; then
        # shellcheck disable=SC1091
        source "$venv/bin/activate"
    elif ! is_dry_run; then
        log "WARNING: no virtualenv at $venv; using $(command -v python3)"
    fi
    # Slurm's GPU plugin on some clusters exports AMD's ROCR/HIP device variables next to CUDA_VISIBLE_DEVICES.
    # They mean nothing on NVIDIA GPUs, but verl's workers refuse to start when both are set.
    unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
    # Training and evaluation read local copies only; nothing is downloaded mid-run.
    export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
    export PYTHONUNBUFFERED=1
}

# Stop on a missing input. A dry run only notes it, because earlier stages have not produced it yet.
require_path() {
    [[ -e "$1" ]] && return 0
    if is_dry_run; then
        log "dry-run: $1 does not exist yet"
        return 0
    fi
    die "missing $1"
}

gpu_count() { awk -F, '{print NF}' <<<"${CUDA_VISIBLE_DEVICES:-0}"; }

# The next phase needs the GPUs that the previous run's Ray and vLLM processes are still releasing.
wait_gpus_idle() {
    is_dry_run && return 0
    local used=""
    for _ in $(seq 60); do
        used=$(nvidia-smi -i "${CUDA_VISIBLE_DEVICES:-0}" --query-gpu=memory.used --format=csv,noheader,nounits \
            2>/dev/null | sort -n | tail -n 1)
        if [[ -z "$used" || "$used" -lt 2000 ]]; then
            return 0
        fi
        sleep 5
    done
    log "WARNING: a GPU still holds ${used} MiB after 5 minutes"
    nvidia-smi
}

# Evaluate one model with the benchmarks split over the first two visible GPUs (one GPU runs both halves in turn).
# Usage: eval_split <model dir> <out dir> <log prefix> <MATH-500 samples> <AIME samples>
eval_split() {
    local model=$1 out=$2 prefix=$3 n_math=$4 n_aime=$5 gpus status=0 pid_a pid_b
    IFS=, read -r -a gpus <<<"${CUDA_VISIBLE_DEVICES:-0}"
    local -a half_a=("BENCHES=math500,aime26" "BENCH_N_SAMPLES=$n_math,$n_aime")
    local -a half_b=("BENCHES=aime24,aime25" "BENCH_N_SAMPLES=$n_aime,$n_aime")
    if is_dry_run || ((${#gpus[@]} < 2)); then
        env CUDA_VISIBLE_DEVICES="${gpus[0]}" MODEL="$model" OUT_DIR="$out" "${half_a[@]}" \
            bash "$REPO_ROOT/pipeline/eval.sh" || status=1
        env CUDA_VISIBLE_DEVICES="${gpus[0]}" MODEL="$model" OUT_DIR="$out" "${half_b[@]}" \
            bash "$REPO_ROOT/pipeline/eval.sh" || status=1
        return "$status"
    fi
    env CUDA_VISIBLE_DEVICES="${gpus[0]}" MODEL="$model" OUT_DIR="$out" "${half_a[@]}" \
        bash "$REPO_ROOT/pipeline/eval.sh" >"${prefix}_a.log" 2>&1 &
    pid_a=$!
    env CUDA_VISIBLE_DEVICES="${gpus[1]}" MODEL="$model" OUT_DIR="$out" "${half_b[@]}" \
        bash "$REPO_ROOT/pipeline/eval.sh" >"${prefix}_b.log" 2>&1 &
    pid_b=$!
    wait "$pid_a" || status=1
    wait "$pid_b" || status=1
    return "$status"
}

# The HuggingFace export of a run's last saved step, for both the SFT (global_step_N/huggingface) and the
# RL (global_step_N/actor/huggingface) layouts. A dry run returns a placeholder.
last_checkpoint() {
    if is_dry_run; then
        echo "$1/global_step_<last>/actor/huggingface"
        return 0
    fi
    find "$1" -maxdepth 4 -type f -path '*global_step_*' -name config.json -printf '%h\n' 2>/dev/null |
        sort -V | tail -n 1
}
