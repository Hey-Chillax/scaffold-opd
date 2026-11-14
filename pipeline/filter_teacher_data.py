# Copyright 2026 Yi Zhang
# SPDX-License-Identifier: Apache-2.0
"""Rejection sampling for SFT: keep the teacher trajectories that finished and that the rule grader accepts.

CPU only. Reads the parquet written by pipeline/teacher_data.sh and writes:

    --out      the kept rows, same columns, ready for METHOD=sft in pipeline/train_stage.sh
    --scores   one row per trajectory: unique_id (the row position in the source data), correct, eos_reached,
               gen_len, which pipeline/build_hard_set.py reads

Run with the project venv and MATH_GRADER_PATH pointing at opd/reward/grader.
"""

import argparse
import importlib.util
import json
import multiprocessing
import os

import pandas as pd

_compute_score = None


def _load_grader(path: str):
    spec = importlib.util.spec_from_file_location("math_reward", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compute_score


def _init_worker(path: str) -> None:
    # The grader installs SIGALRM timers, so it has to run on a process's main thread.
    global _compute_score
    _compute_score = _load_grader(path)


def _grade(item: tuple[str, str]) -> bool:
    text, gold = item
    return _compute_score(solution_str=text, ground_truth=gold) > 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--grader", default="opd/reward/math_reward.py")
    args = parser.parse_args()

    frame = pd.read_parquet(args.data)
    items = [(str(row["messages"][-1]["content"]), str(row["gt"])) for _, row in frame.iterrows()]
    with multiprocessing.Pool(args.workers, initializer=_init_worker, initargs=(args.grader,)) as pool:
        correct = pool.map(_grade, items, chunksize=32)

    frame["correct"] = correct
    keep = frame["correct"] & frame["eos_reached"]
    kept = frame[keep].drop(columns=["correct"])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    kept.to_parquet(args.out, index=False)
    frame[["unique_id", "correct", "eos_reached", "gen_len"]].to_parquet(args.scores, index=False)

    dropped = frame[~keep]
    stats = {
        "trajectories": len(frame),
        "finished": int(frame["eos_reached"].sum()),
        "correct": int(frame["correct"].sum()),
        "kept": len(kept),
        "kept_percent": round(100 * len(kept) / max(len(frame), 1), 2),
        "kept_mean_tokens": round(float(kept["gen_len"].mean()), 1) if len(kept) else 0.0,
        "dropped_mean_tokens": round(float(dropped["gen_len"].mean()), 1) if len(dropped) else 0.0,
        "kept_total_tokens": int(kept["gen_len"].sum()),
    }
    print(json.dumps(stats))
    with open(args.out + ".stats.json", "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
