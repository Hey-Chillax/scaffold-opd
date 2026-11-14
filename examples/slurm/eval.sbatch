#!/usr/bin/env bash
#SBATCH --job-name=scaffold-opd-eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%j.out
# Copyright 2026 Yi Zhang
# SPDX-License-Identifier: Apache-2.0
#
# Slurm wrapper around pipeline/eval.sh. Needs MODEL; one GPU evaluates all four benchmarks at the 32k budget in a few hours.
# Submit from the repository root and add your cluster's partition and account on the command line:
#   sbatch --partition=<partition> --account=<account> examples/slurm/eval.sbatch
# Every variable the wrapped script reads passes through, e.g. MODEL_DIR, DATA_DIR, OUTPUT_ROOT, CKPT_DIR, VENV and
# ENV_SETUP (a file to source first for module loads or cache locations). Compute nodes without internet are fine:
# models and data are read from local paths, HF_HUB_OFFLINE=1 and WANDB_MODE=offline are the defaults.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
exec bash pipeline/eval.sh "$@"
