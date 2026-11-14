# Copyright 2026 Yi Zhang
# SPDX-License-Identifier: Apache-2.0
"""Compare a student alone with the same student plus concluding takeovers, at one token budget.

CPU only. Run from the repo root on a login node:

    python3 tools/analysis/conclude_probe_stats.py --student_alone <eval dir> --probe <conclude_probe dir>

The student's run must have a budget at least as large as the probe's: its responses are cut back to the probe's
budget here, and a response counts as answered only if it finished inside that budget. The last line pools the
three AIME sets and reports the change in answered rate, accuracy and accuracy among answered responses.
"""

import argparse
import json
import os


def load(directory: str, bench: str) -> list[dict]:
    with open(os.path.join(directory, f"{bench}.jsonl")) as f:
        return [json.loads(line) for line in f]


def summarize(rows: list[dict], budget: int) -> dict:
    answered = [r for r in rows if r["gen_len"] <= budget and "\\boxed" in r["gen_text"]]
    correct = sum(bool(r["correct"]) for r in answered)
    return {
        "n": len(rows),
        "answered": 100 * len(answered) / len(rows),
        "accuracy": 100 * correct / len(rows),
        "accuracy_when_answered": 100 * correct / max(len(answered), 1),
        "mean_tokens": sum(min(r["gen_len"], budget) for r in rows) / len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--student_alone", required=True)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--budget", type=int, default=16384)
    args = parser.parse_args()

    pooled = {"student": [], "probe": []}
    print(f"budget {args.budget}; 'answered' means a boxed answer inside the budget")
    print(
        f"{'bench':8s} {'run':8s} {'n':>5s} {'answered%':>10s} {'accuracy%':>10s} "
        f"{'acc|answered%':>14s} {'mean tokens':>12s}"
    )
    for bench in ("math500", "aime24", "aime25", "aime26"):
        student = load(args.student_alone, bench)
        probe = load(args.probe, bench)
        if bench != "math500":
            pooled["student"] += student
            pooled["probe"] += probe
        for name, rows in (("student", student), ("probe", probe)):
            s = summarize(rows, args.budget)
            print(
                f"{bench:8s} {name:8s} {s['n']:5d} {s['answered']:10.1f} {s['accuracy']:10.1f} "
                f"{s['accuracy_when_answered']:14.1f} {s['mean_tokens']:12.0f}"
            )

    a, b = summarize(pooled["student"], args.budget), summarize(pooled["probe"], args.budget)
    gain = b["answered"] - a["answered"]
    drop = a["accuracy_when_answered"] - b["accuracy_when_answered"]
    print(
        f"AIME pooled: answered {a['answered']:.1f} -> {b['answered']:.1f} ({gain:+.1f}), "
        f"accuracy {a['accuracy']:.1f} -> {b['accuracy']:.1f}, "
        f"acc|answered {a['accuracy_when_answered']:.1f} -> {b['accuracy_when_answered']:.1f} ({-drop:+.1f}), "
        f"tokens {a['mean_tokens']:.0f} -> {b['mean_tokens']:.0f}"
    )


if __name__ == "__main__":
    main()
