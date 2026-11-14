# Copyright 2026 Yi Zhang
# SPDX-License-Identifier: Apache-2.0
"""Per-step statistics over the rollout dumps of training runs.

Usage: python3 tools/analysis/rollout_stats.py <run_dir> [<run_dir> ...] > rollout_stats.json
Each <run_dir> holds rollout_dump/<step>.jsonl written by verl (trainer.rollout_data_dir).
Prints {run: {"steps": {step: {n, loop_rate, boxed_rate, score_mean, chars_mean}},
"teacher_first_words_steps_1_10": {...}, "teacher_first_words_last_10_steps": {...},
"segment_lengths": [...], "examples": {"early": ..., "late": ...}}}; the takeover-only fields stay empty for OPD.

A response counts as looping when one 50-character chunk (stride 10) occurs at least 5 times in its last
3000 characters; long but varied reasoning does not trip it.
"""

import glob
import json
import os
import re
import sys
from collections import Counter

LOOP_REPEATS = 5


def loop_score(text: str) -> int:
    tail = text[-3000:]
    if len(tail) < 500:
        return 1
    chunks = Counter(tail[i : i + 50] for i in range(0, len(tail) - 50, 10))
    return chunks.most_common(1)[0][1]


def teacher_segments(marked: str) -> list[str]:
    return re.findall(r"【T】(.*?)(?:【/T】|$)", marked, flags=re.S)


def first_word(segment: str) -> str:
    match = re.match(r"\s*([A-Za-z']+|\S)", segment)
    return match.group(1) if match else ""


out = {}
for run_dir in sys.argv[1:]:
    name = os.path.basename(run_dir.rstrip("/"))
    steps, words_early, words_late, seg_lens = {}, Counter(), Counter(), []
    examples = {}
    files = sorted(
        glob.glob(os.path.join(run_dir, "rollout_dump", "*.jsonl")),
        key=lambda p: int(os.path.basename(p).split(".")[0]),
    )
    last_step = int(os.path.basename(files[-1]).split(".")[0]) if files else 0
    for path in files:
        step = int(os.path.basename(path).split(".")[0])
        rows = [json.loads(line) for line in open(path)]
        loops = [loop_score(r["output"]) >= LOOP_REPEATS for r in rows]
        boxed = ["\\boxed{" in r["output"] for r in rows]
        info = {
            "n": len(rows),
            "loop_rate": sum(loops) / len(rows),
            "boxed_rate": sum(boxed) / len(rows),
            "score_mean": sum(float(r.get("score", 0.0)) for r in rows) / len(rows),
            "chars_mean": sum(len(r["output"]) for r in rows) / len(rows),
        }
        if any("relay_output_marked" in r for r in rows):
            for r in rows:
                segs = teacher_segments(r.get("relay_output_marked", ""))
                for seg in segs:
                    (words_early if step <= 10 else words_late if step > last_step - 10 else Counter()).update(
                        [first_word(seg)]
                    )
                for span in r.get("relay_teacher_spans") or []:
                    seg_lens.append(span[1] - span[0])
            # Example candidates: an early budget-exhausted response and a late natural, correct one.
            if step == 1 and "early" not in examples:
                pick = min(rows, key=lambda r: len(r["output"]))
                examples["early"] = {
                    "step": step,
                    "score": pick.get("score"),
                    "gts": pick.get("gts"),
                    "marked": pick.get("relay_output_marked", "")[:6000],
                }
            if step > last_step - 10:
                for r in rows:
                    if (
                        not r.get("trigger_stopped")
                        and float(r.get("score", 0)) > 0
                        and "【T】" in r.get("relay_output_marked", "")
                        and len(r["output"]) < examples.get("late", {}).get("chars", 10**9)
                    ):
                        examples["late"] = {
                            "step": step,
                            "score": r.get("score"),
                            "gts": r.get("gts"),
                            "chars": len(r["output"]),
                            "marked": r["relay_output_marked"][:8000],
                        }
        steps[step] = info
    out[name] = {
        "steps": steps,
        "teacher_first_words_steps_1_10": dict(words_early.most_common(15)),
        "teacher_first_words_last_10_steps": dict(words_late.most_common(15)),
        "segment_lengths": seg_lens,
        "examples": examples,
    }
json.dump(out, sys.stdout, ensure_ascii=False)
