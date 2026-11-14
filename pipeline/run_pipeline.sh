#!/usr/bin/env bash
# Copyright 2026 Yi Zhang
# SPDX-License-Identifier: Apache-2.0
#
# The full Scaffold-OPD post-training pipeline on one machine.
#
#   teacher   the teacher writes one solution per training problem                       (teacher_data.sh)
#   filter    rejection sampling: keep solutions that finished and pass the rule grader  (filter_teacher_data.py)
#   sft       SFT on the kept solutions for one epoch, then evaluation
#   scaffold  Scaffold-OPD from the SFT checkpoint, then evaluation
#   hard      problems the teacher solved but the student never solved in Scaffold-OPD  (build_hard_set.py)
#   grpo      GRPO from the Scaffold-OPD checkpoint on those problems, then evaluation
#
# A failed training stage stops the pipeline; a failed evaluation is logged and the pipeline goes on. Every
# finished stage leaves a .done_<stage> marker in the run directory, so running again with the same --run-id
# continues after the last finished stage.
#
# Usage:
#   bash pipeline/run_pipeline.sh [--run-id ID] [--dry-run]
#
#   --run-id ID   names the run directory $OUTPUT_ROOT/pipeline_<ID>; default a timestamp
#   --dry-run     check the configuration and print every stage's command without running anything
#   -h, --help    show this help
#
# Needs at least two 80 GB GPUs (the reported run used two A100s). Configuration through the environment,
# defaults in brackets (MODEL_DIR, DATA_DIR, OUTPUT_ROOT, VENV and ENV_SETUP are described in common.sh):
#   STUDENT_MODEL [$MODEL_DIR/Qwen3-1.7B]      TEACHER_MODEL [$MODEL_DIR/Qwen3-4B-Instruct-2507]
#   TRAIN_DATA [$DATA_DIR/train/dapo_math_17k_en.parquet]                 BENCH [$DATA_DIR/bench]
#   CKPT_DIR [the run directory]      SCAFFOLD_STEPS [40]      GRPO_STEPS [30, fewer if the hard set is small]
#   GRPO_BATCH [32 prompts x 8 rollouts]   RESPONSE_LEN [16384]   EVAL_N_MATH [4]   EVAL_N_AIME [16]   MAX_NEW [32768]
#
# Outputs under $OUTPUT_ROOT/pipeline_<ID>/: teacher/, sft/, scaffold/, grpo/ (one run directory per stage),
# hard_set.parquet, and eval_<stage>/<bench>.summary.json.
set -eo pipefail

usage() { sed -n '5,/^set -eo pipefail/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'; }

run_id=""
while (($#)); do
    case "$1" in
        --run-id)
            [[ $# -ge 2 && -n "$2" ]] || {
                echo "--run-id needs a value" >&2
                exit 2
            }
            run_id=$2
            shift 2
            ;;
        --run-id=*)
            run_id=${1#*=}
            shift
            ;;
        --dry-run)
            export DRY_RUN=1
            shift
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            echo "unknown argument: $1 (see --help)" >&2
            exit 2
            ;;
    esac
done

# shellcheck source=pipeline/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
LOG_TAG=pipeline
setup_env
set -u
cd "$REPO_ROOT"

run_id=${run_id:-$(date +%Y%m%d_%H%M%S)}
[[ "$run_id" =~ ^[A-Za-z0-9._-]+$ ]] || die "--run-id may only contain letters, digits, '.', '_' and '-'"
root=$OUTPUT_ROOT/pipeline_$run_id
ckpt_root=${CKPT_DIR:-$root}
export STUDENT_MODEL=${STUDENT_MODEL:-$DEFAULT_STUDENT}
export TEACHER_MODEL=${TEACHER_MODEL:-$DEFAULT_TEACHER}
train_data=${TRAIN_DATA:-$DEFAULT_TRAIN_DATA}
export BENCH=${BENCH:-$DEFAULT_BENCH_DIR}
export BENCH_DIR=$BENCH
scaffold_steps=${SCAFFOLD_STEPS:-40}
grpo_steps=${GRPO_STEPS:-30}
grpo_batch=${GRPO_BATCH:-32}
response_len=${RESPONSE_LEN:-16384}
n_math=${EVAL_N_MATH:-4}
n_aime=${EVAL_N_AIME:-16}
export MAX_NEW=${MAX_NEW:-32768}
for value in "$scaffold_steps" "$grpo_steps" "$grpo_batch" "$response_len" "$n_math" "$n_aime" "$MAX_NEW"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "step, batch, length and sample settings must be positive integers (got '$value')"
done
(($(gpu_count) >= 2)) || die "the pipeline needs at least 2 GPUs; CUDA_VISIBLE_DEVICES lists $(gpu_count)"
export MATH_GRADER_PATH=${MATH_GRADER_PATH:-$REPO_ROOT/opd/reward/grader}
python=${PYTHON:-python3}

require_path "$STUDENT_MODEL/config.json"
require_path "$TEACHER_MODEL/config.json"
require_path "$train_data"
require_path "$BENCH/aime-24_verl.parquet"
is_dry_run || mkdir -p "$root"

# Runs one stage unless its marker exists. A failed stage ends the pipeline.
stage() {
    local name=$1
    shift
    if ! is_dry_run && [[ -f "$root/.done_$name" ]]; then
        log "stage $name already done"
        return 0
    fi
    log "stage $name"
    if "$@"; then
        is_dry_run || touch "$root/.done_$name"
        log "stage $name finished"
        wait_gpus_idle
    else
        die "stage $name failed"
    fi
}

evaluate() {
    local name=$1 model=$2
    if ! is_dry_run && [[ -f "$root/.done_eval_$name" ]]; then
        return 0
    fi
    log "evaluating $name: $model"
    if eval_split "$model" "$root/eval_$name" "$root/eval_$name" "$n_math" "$n_aime"; then
        is_dry_run || touch "$root/.done_eval_$name"
        log "evaluation $name finished"
    else
        log "ERROR: evaluation $name failed; see $root/eval_${name}_[ab].log"
    fi
    wait_gpus_idle
}

log "run=$run_id root=$root checkpoints=$ckpt_root gpus=${CUDA_VISIBLE_DEVICES:-unset} dry_run=${DRY_RUN:-0}"

stage teacher env SOURCE_DATA="$train_data" OUTPUT_PARQUET="$root/teacher/teacher.parquet" MAX_NEW="$response_len" \
    bash pipeline/teacher_data.sh
stage filter run "$python" pipeline/filter_teacher_data.py --data "$root/teacher/teacher.parquet" \
    --out "$root/teacher/teacher_correct.parquet" --scores "$root/teacher/teacher_scores.parquet"

stage sft env METHOD=sft TRAIN_DATA="$root/teacher/teacher_correct.parquet" STEPS=0 BATCH=128 \
    RESPONSE_LEN="$response_len" OUTPUT_DIR="$root/sft" CKPT_DIR="$ckpt_root/sft" WANDB_TAGS=pipeline,sft \
    bash pipeline/train_stage.sh
sft_ckpt=$(last_checkpoint "$ckpt_root/sft")
[[ -n "$sft_ckpt" ]] || die "no SFT checkpoint under $ckpt_root/sft"
evaluate sft "$sft_ckpt"

stage scaffold env METHOD=scaffold STUDENT_MODEL="$sft_ckpt" TRAIN_DATA="$train_data" STEPS="$scaffold_steps" \
    BATCH=128 RESPONSE_LEN="$response_len" SAVE_FREQ="$scaffold_steps" ROLLOUT_GPU_MEMORY_UTILIZATION=0.85 \
    OUTPUT_DIR="$root/scaffold" CKPT_DIR="$ckpt_root/scaffold" WANDB_TAGS=pipeline,scaffold \
    bash pipeline/train_stage.sh
scaffold_ckpt=$(last_checkpoint "$ckpt_root/scaffold")
[[ -n "$scaffold_ckpt" ]] || die "no Scaffold-OPD checkpoint under $ckpt_root/scaffold"
evaluate scaffold "$scaffold_ckpt"

stage hard run "$python" pipeline/build_hard_set.py --source "$train_data" \
    --scores "$root/teacher/teacher_scores.parquet" --rollout_dir "$root/scaffold/rollout_dump" \
    --out "$root/hard_set.parquet"

# A run stops at the end of its data, before the last-step checkpoint, when the steps outnumber the batches.
if ! is_dry_run && [[ ! -f "$root/.done_grpo" ]]; then
    hard_rows=$("$python" -c 'import sys, pandas as pd; print(len(pd.read_parquet(sys.argv[1])))' "$root/hard_set.parquet")
    if ((hard_rows / grpo_batch < grpo_steps)); then
        grpo_steps=$((hard_rows / grpo_batch))
        ((grpo_steps > 0)) || die "the hard set has only $hard_rows problems, fewer than one GRPO batch"
        log "WARNING: the hard set has $hard_rows problems, so GRPO runs $grpo_steps steps"
    fi
fi
stage grpo env METHOD=grpo STUDENT_MODEL="$scaffold_ckpt" TRAIN_DATA="$root/hard_set.parquet" STEPS="$grpo_steps" \
    BATCH="$grpo_batch" ROLLOUT_N=8 RESPONSE_LEN="$response_len" SAVE_FREQ="$grpo_steps" \
    OUTPUT_DIR="$root/grpo" CKPT_DIR="$ckpt_root/grpo" WANDB_TAGS=pipeline,grpo bash pipeline/train_stage.sh
evaluate grpo "$(last_checkpoint "$ckpt_root/grpo")"

log "done; evaluation summaries in $root/eval_<stage>/ (collect them with tools/analysis/collect_results.py)"
