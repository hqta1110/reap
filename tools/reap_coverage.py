#!/usr/bin/env python3
"""Coverage scan: which REAP accuracy numbers exist, for 4 models x 9 tasks.

Answers "what do we actually have?" rather than "what do the numbers say" --
one grid per calibration corpus, a value where a cell landed and "-" where it
did not. Same protocol filter and same corpus detection as reap_report.py.

Reads the published mirror AND the two live scratch trees, so a sweep still in
flight shows its cells as they land instead of looking like a hole.
"""
import json, glob, os, re, collections

import os
# Result trees to scan, colon-separated in $REAP_RESULT_ROOTS. Defaults to the
# summaries committed alongside this repo, so a fresh clone renders the tables
# with no configuration; point it at live sweep output to report in-flight runs.
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOTS = [r for r in os.environ.get("REAP_RESULT_ROOTS", "").split(":") if r] or [
    os.path.join(_HERE, "results", "codealpaca"),
    os.path.join(_HERE, "results", "tulu-personas-math"),
    os.path.join(_HERE, "results", "fineweb"),
]
CANON = {"qwen_qwen3-30b-a3b": "qwen3-30b", "qwen3-30b": "qwen3-30b",
         "qwen_qwen3.6-35b-a3b": "qwen3.6-35b", "qwen36-35b": "qwen3.6-35b",
         "google_gemma-4-26b-a4b-it": "gemma4", "gemma4": "gemma4"}
MAIN4 = ["qwen3-30b", "glm-4.7-flash", "qwen3.6-35b", "gemma4"]
# The nine the campaign runs, in the order asked for.
TASKS = ["gsm8k", "math500", "math_hard", "aime", "humaneval_plus",
         "lcb", "gpqa_diamond", "mmlu_pro_400", "livebench_mixed"]
SHORT = {"humaneval_plus": "humaneval", "gpqa_diamond": "gpqa",
         "mmlu_pro_400": "mmlu400", "livebench_mixed": "livebench"}
CORPORA = [("codealpaca", "code"), ("tulu-personas-math", "math"), ("fineweb", "fineweb")]

def canon_model(d):
    k = d.lower()
    if "glm-4.7-flash" in k or "glm4.7" in k: return "glm-4.7-flash"
    return CANON.get(k, d)

def calib(arm, model_path):
    s = f"{arm} {model_path}".lower()
    if "evol-codealpaca" in s or "codealpaca" in s or "code256" in s: return "codealpaca"
    if "fineweb" in s or "fw256" in s: return "fineweb"
    if "personas-math" in s or "math256" in s: return "tulu-personas-math"
    return "unrecorded"

have = {}   # (calib, model, ratio, task) -> value
for root in ROOTS:
    for p in glob.glob(os.path.join(root, "**", "scores.json"), recursive=True):
        rel = os.path.relpath(p, root).split(os.sep)
        if len(rel) < 3 or "reap" not in rel[1].lower(): continue
        model, arm = canon_model(rel[0]), rel[1]
        # Old server-protocol arms carry a suffix the offline campaign never uses.
        if arm.lower().endswith("_c64"): continue
        try: doc = json.load(open(p))
        except Exception: continue
        for d in (doc if isinstance(doc, list) else [doc]):
            sc = d.get("scores") or {}
            if not sc: continue
            m = d.get("meta", {})
            metric = d.get("primary_metric") or list(sc)[0]
            v = (sc.get(metric) or list(sc.values())[0]).get("value")
            if v is None: continue
            mo = re.match(r"reap(\d+)", arm.lower())
            have[(calib(arm, str(m.get("model"))), model,
                  "reap" + (mo.group(1) if mo else "?"), d.get("task"))] = v

W = 10
for key, title in CORPORA:
    print(f"\n### calibrated on {title}  ({key})")
    hdr = "model            ratio  " + "".join(SHORT.get(t, t).ljust(W) for t in TASKS)
    print(hdr); print("-" * len(hdr))
    for model in MAIN4:
        for ratio in ("reap25", "reap50"):
            cells = [have.get((key, model, ratio, t)) for t in TASKS]
            if not any(c is not None for c in cells): 
                line = f"{model:<16} {ratio:<6} " + "".join("-".ljust(W) for _ in TASKS)
            else:
                line = f"{model:<16} {ratio:<6} " + "".join(
                    (f"{c:.2f}" if c is not None else "-").ljust(W) for c in cells)
            print(line)
    n = sum(1 for k in have if k[0] == key and k[1] in MAIN4 and k[3] in TASKS)
    print(f"  {n}/{len(MAIN4)*2*len(TASKS)} cells")

other = sorted({k[0] for k in have} - {c for c, _ in CORPORA})
if other: print("\nunclassified corpora seen:", other)
