# Copyright 2026 Yi Zhang
# SPDX-License-Identifier: Apache-2.0
"""Build the problem set for the RL stage: problems the teacher solved and the student missed.

CPU only. It joins three inputs on the problem text:

    --source          the training parquet both stages drew from
    --scores          pipeline/filter_teacher_data.py's per-problem teacher verdicts
    --rollout_dir     the rollout_dump/ of the previous stage: <step>.jsonl with `input` and `score`

A relay rollout scores 0 both when the student answered wrongly and when the run was stopped after its last
takeover, so the stats report the two apart. The output has the source columns, for METHOD=grpo's TRAIN_DATA.
"""

import argparse
import glob
import json
import os

import pandas as pd


def normalize(text: str) -> str:
    return " ".join(text.split())


def problem_of_prompt(prompt) -> str:
    for message in prompt:
        if message["role"] == "user":
            return normalize(str(message["content"]))
    raise ValueError("row has no user message")


def problem_of_rollout(rendered: str) -> str:
    """Pull the user turn out of the decoded chat template: system, <text>, user, <text>, assistant."""
    body = rendered.split("\nuser\n", 1)[-1]
    return normalize(body.rsplit("\nassistant", 1)[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--rollout_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max_step", type=int, default=None, help="ignore rollout_dump/<step>.jsonl past this step")
    args = parser.parse_args()

    source = pd.read_parquet(args.source)
    position = {problem_of_prompt(prompt): i for i, prompt in enumerate(source["prompt"])}
    teacher = pd.read_parquet(args.scores)
    teacher_solved = set(teacher.loc[teacher["correct"] & teacher["eos_reached"], "unique_id"].astype(int))

    seen: dict[int, dict] = {}
    rollouts = unmatched = 0
    for path in glob.glob(os.path.join(args.rollout_dir, "*.jsonl")):
        step = os.path.splitext(os.path.basename(path))[0]
        if args.max_step is not None and step.isdigit() and int(step) > args.max_step:
            continue
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                rollouts += 1
                index = position.get(problem_of_rollout(row["input"]))
                if index is None:
                    unmatched += 1
                    continue
                entry = seen.setdefault(index, {"solved": False, "stopped": False})
                entry["solved"] |= float(row["score"]) > 0
                entry["stopped"] |= bool(row.get("trigger_stopped", False))

    missed = {i: e for i, e in seen.items() if not e["solved"]}
    hard = sorted(i for i in missed if i in teacher_solved)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    source.iloc[hard].to_parquet(args.out, index=False)

    stats = {
        "rollouts": rollouts,
        "rollouts_unmatched": unmatched,
        "problems_seen": len(seen),
        "student_missed": len(missed),
        "teacher_solved_of_all": len(teacher_solved),
        "hard_set": len(hard),
        "hard_set_stopped_by_budget": sum(missed[i]["stopped"] for i in hard),
    }
    print(json.dumps(stats))
    with open(args.out + ".stats.json", "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
