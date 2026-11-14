#!/usr/bin/env bash
# Copyright 2026 Yi Zhang
# SPDX-License-Identifier: Apache-2.0
#
# GRPO on top of an existing checkpoint: build the set of problems the teacher solved and the checkpoint's own
# training run never solved (build_hard_set.py), run GRPO on it from the checkpoint, then evaluate the last step.
#
#   START_MODEL=<run>/global_step_40/actor/huggingface ROLLOUT_DIR=<run>/rollout_dump MAX_STEP=40 \
#   SCORES=<pipeline run>/teacher/teacher_scores.parquet bash pipeline/rl_stage.sh
#
# Env (see pipeline/common.sh for paths):
#   START_MODEL     required, HuggingFace directory the stage starts from
#   ROLLOUT_DIR     required, rollout_dump/ of the run that produced START_MODEL
#   MAX_STEP        only that run's rollouts up to this step (the checkpoint's step)
#   SCORES          required, per-problem teacher verdicts written by filter_teacher_data.py
#   GRPO_STEPS / GRPO_BATCH   default 30 / 32 (8 rollouts each); fewer steps if the set is small
#   GRPO_SAVE_FREQ  checkpoint every this many steps, default only the last one. With a frequency set, a run that
#                   dies late still leaves a checkpoint to evaluate.
#   RUN_DIR         default $OUTPUT_ROOT/rl_stage_<timestamp>
#   CKPT_DIR        default $RUN_DIR/ckpt
#   EVAL_N_MATH / EVAL_N_AIME / MAX_NEW   default 4 / 16 / 32768
set -uo pipefail

# shellcheck source=pipeline/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
LOG_TAG=rl-stage
setup_env
cd "$REPO_ROOT" || exit 1

START_MODEL=${START_MODEL:?set START_MODEL}
ROLLOUT_DIR=${ROLLOUT_DIR:?set ROLLOUT_DIR}
SCORES=${SCORES:?set SCORES}
root=${RUN_DIR:-$OUTPUT_ROOT/rl_stage_$(date +%Y%m%d_%H%M%S)}
ckpt=${CKPT_DIR:-$root/ckpt}
steps=${GRPO_STEPS:-30}
batch=${GRPO_BATCH:-32}
train_data=${TRAIN_DATA:-$DEFAULT_TRAIN_DATA}
python=${PYTHON:-python3}
export MAX_NEW=${MAX_NEW:-32768}
require_path "$START_MODEL/config.json"
require_path "$ROLLOUT_DIR"
require_path "$SCORES"
is_dry_run || mkdir -p "$root"
log "start=$START_MODEL root=$root checkpoints=$ckpt gpus=${CUDA_VISIBLE_DEVICES:-unset}"

run "$python" pipeline/build_hard_set.py --source "$train_data" --scores "$SCORES" --rollout_dir "$ROLLOUT_DIR" \
    ${MAX_STEP:+--max_step "$MAX_STEP"} --out "$root/hard_set.parquet" || die "building the hard set failed"
if ! is_dry_run; then
    rows=$("$python" -c 'import sys, pandas as pd; print(len(pd.read_parquet(sys.argv[1])))' "$root/hard_set.parquet")
    if ((rows / batch < steps)); then
        steps=$((rows / batch))
        ((steps > 0)) || die "the hard set has only $rows problems, fewer than one GRPO batch"
        log "WARNING: the hard set has $rows problems, so GRPO runs $steps steps"
    fi
fi

save_freq=${GRPO_SAVE_FREQ:-$steps}
log "grpo: $steps steps x $batch problems x 8 rollouts, saving every $save_freq steps"
status=0
if ! env METHOD=grpo STUDENT_MODEL="$START_MODEL" TRAIN_DATA="$root/hard_set.parquet" STEPS="$steps" BATCH="$batch" \
    ROLLOUT_N=8 RESPONSE_LEN=16384 SAVE_FREQ="$save_freq" OUTPUT_DIR="$root/grpo" CKPT_DIR="$ckpt" \
    WANDB_TAGS=rl_stage,grpo bash pipeline/train_stage.sh; then
    log "ERROR: grpo failed"
    status=1
fi
wait_gpus_idle

model=$(last_checkpoint "$ckpt")
[[ -n "$model" ]] || die "no checkpoint under $ckpt to evaluate"
((status == 0)) || log "WARNING: evaluating the last checkpoint the failed run saved"
if eval_split "$model" "$root/eval_grpo" "$root/eval_grpo" "${EVAL_N_MATH:-4}" "${EVAL_N_AIME:-16}"; then
    log "done; summaries in $root/eval_grpo/"
else
    die "evaluation failed; see $root/eval_grpo_[ab].log"
fi
exit "$status"
