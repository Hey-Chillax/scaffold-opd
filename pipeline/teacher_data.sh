#!/usr/bin/env bash
# Copyright 2026 Yi Zhang
# SPDX-License-Identifier: Apache-2.0
#
# Generate one teacher solution per training problem (the SFT data), rendered with the student's tokenizer and
# Qwen3 non-thinking template. Each visible GPU generates one shard through opd/scripts/data/teacher_trajectories.sh;
# the shards are merged into OUTPUT_PARQUET, and finished shards are skipped when the same command is rerun.
#
#   OUTPUT_PARQUET=outputs/teacher/teacher.parquet bash pipeline/teacher_data.sh
#
# Env (see pipeline/common.sh for MODEL_DIR, DATA_DIR, OUTPUT_ROOT, VENV, ENV_SETUP, DRY_RUN):
#   STUDENT_MODEL   default $MODEL_DIR/Qwen3-1.7B; supplies the tokenizer and chat template
#   TEACHER_MODEL   default $MODEL_DIR/Qwen3-4B-Instruct-2507
#   SOURCE_DATA     default $DATA_DIR/train/dapo_math_17k_en.parquet
#   OUTPUT_PARQUET  default $OUTPUT_ROOT/teacher_data/teacher.parquet
#   MAX_NEW         max new tokens, default 16384 (temperature 1.0)
#   MAX_NUM_SEQS    concurrent sequences per GPU, default 128
#   GPU_GROUPS      ';' separates data-parallel workers, ',' joins one tensor-parallel group; default one
#                   worker per visible GPU
set -eo pipefail

# shellcheck source=pipeline/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
LOG_TAG=teacher
setup_env
set -u
cd "$REPO_ROOT"

export STUDENT_MODEL=${STUDENT_MODEL:-$DEFAULT_STUDENT}
export TEACHER_MODEL=${TEACHER_MODEL:-$DEFAULT_TEACHER}
export SOURCE_DATA=${SOURCE_DATA:-$DEFAULT_TRAIN_DATA}
export OUTPUT_PARQUET=${OUTPUT_PARQUET:-$OUTPUT_ROOT/teacher_data/teacher.parquet}
export MAX_NEW=${MAX_NEW:-16384}
export MAX_NUM_SEQS=${MAX_NUM_SEQS:-128}
visible=${CUDA_VISIBLE_DEVICES:-0}
export GPU_GROUPS=${GPU_GROUPS:-${visible//,/;}}

require_path "$STUDENT_MODEL/config.json"
require_path "$TEACHER_MODEL/config.json"
require_path "$SOURCE_DATA"
log "teacher=$TEACHER_MODEL source=$SOURCE_DATA out=$OUTPUT_PARQUET max_new=$MAX_NEW gpu_groups=$GPU_GROUPS"
show_env STUDENT_MODEL TEACHER_MODEL SOURCE_DATA OUTPUT_PARQUET MAX_NEW MAX_NUM_SEQS GPU_GROUPS
run bash opd/scripts/data/teacher_trajectories.sh
