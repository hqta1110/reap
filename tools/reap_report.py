#!/usr/bin/env python3
"""Report the current REAP runs, split by calibration corpus.

ONE protocol only: 0-shot, greedy, max_tokens 4096, offline batch-pinned at B=64.
Older server-protocol runs (5-shot / 2048 / HTTP) and non-4-model results are
deliberately NOT here -- they are not comparable with these numbers and mixing
them is how a corpus effect gets read as a method result. They remain in git
history and in the results repo.

The split that IS meaningful is the calibration corpus, because it decides which
experts survive pruning. Hence two tables and a delta, not one table with a column.

Calibration corpus is not recorded in any result meta. It is recoverable only from
the checkpoint path (new runs) or the tree/arm name (old runs). Where neither says,
it prints as "unrecorded" -- which is the honest answer and also a bug report for
next time.
"""
import json, glob, os, re, collections

ROOTS = [
    # The PUBLISHED copy, not the scratch trees. /home/PC/sweep2/results and
    # /home/PC/reap-tulu/results are per-sweep scratch: sweep2's REAP arms have
    # already been cleaned off disk, which silently emptied section 1 of every
    # codealpaca row. The mirror is what actually persists, and it carries both
    # corpora, so read the report from there and let the calib column separate them.
    ("/home/PC/moe-eval-results/results/accuracy/offline_sere",        "offline", "offline B=64 (2026-09-12/14)"),
    ("/home/PC/moe-eval-unified/results",                              "server",  "results (2026-08)"),
    ("/home/PC/moe-eval-unified/results_codealpaca_reap_archive",      "server",  "codealpaca archive"),
    ("/home/PC/moe-eval-unified/results_pilot_prefinalized_20260819",  "server",  "pilot 2026-08-19"),
    ("/home/PC/moe-eval-unified/results_s43",                          "server",  "seed 43"),
    ("/home/PC/moe-eval-unified/results_s44",                          "server",  "seed 44"),
]
CANON = {"qwen_qwen3-30b-a3b": "qwen3-30b", "qwen3-30b": "qwen3-30b",
         "qwen_qwen3.6-35b-a3b": "qwen3.6-35b", "qwen36-35b": "qwen3.6-35b",
         "google_gemma-4-26b-a4b-it": "gemma4", "gemma4": "gemma4"}
MAIN4 = ["qwen3-30b", "qwen3.6-35b", "glm-4.7-flash", "gemma4"]
TASKS = ["gsm8k", "math_hard", "math500", "aime", "lcb", "humaneval_plus",
         "mmlu_pro_400", "gpqa_diamond", "livebench_mixed", "mbppplus"]

def canon_model(d):
    k = d.lower()
    if "glm-4.7-flash" in k or "glm4.7" in k: return "glm-4.7-flash"
    if "deepseek" in k: return "deepseek-v2-lite"
    if "qwen15" in k or "qwen1.5" in k: return "qwen15-moe"
    return CANON.get(k, d)

def calib(arm, model_path, root):
    """Calibration corpus, from the only places it is ever written down."""
    s = f"{arm} {model_path} {root}".lower()
    if "evol-codealpaca" in s or "codealpaca" in s or "code256" in s: return "codealpaca"
    if "fineweb" in s or "fw256" in s: return "fineweb"
    if "personas-math" in s or "math256" in s: return "tulu-personas-math"
    if "_c64" in arm.lower() or "1024" in arm.lower(): return "fineweb"   # glm early arms
    return "unrecorded"

rows = []   # (protocol, when, model, ratio, calib, arm, task, value, metric, capfrac, n, bs)
for root, proto, when in ROOTS:
    for p in glob.glob(os.path.join(root, "**", "scores.json"), recursive=True):
        rel = os.path.relpath(p, root).split(os.sep)
        if len(rel) < 3: continue
        model, arm = rel[0], rel[1]
        if "reap" not in arm.lower(): continue
        try: doc = json.load(open(p))
        except Exception: continue
        for d in (doc if isinstance(doc, list) else [doc]):
            sc = d.get("scores") or {}
            if not sc: continue
            m = d.get("meta", {})
            metric = d.get("primary_metric") or list(sc)[0]
            v = (sc.get(metric) or list(sc.values())[0]).get("value")
            if v is None: continue
            dg = d.get("diagnostics") or {}
            n = dg.get("n") or (sc.get(metric) or {}).get("n")
            cap = (100.0 * (dg.get("hit_length_cap") or 0) / n) if n else None
            mo = re.match(r"reap(\d+)", arm.lower())
            rows.append(dict(proto=proto, when=when, model=canon_model(model),
                             ratio="reap" + (mo.group(1) if mo else "?"),
                             calib=calib(arm, str(m.get("model")), root), arm=arm,
                             task=d.get("task"), v=v, metric=metric, cap=cap, n=n,
                             bs=(m.get("batching") or {}).get("requested_batch_size")))

# de-dup: same (proto,model,arm,task) can appear in both a source tree and a copy
best = {}
for r in rows:
    best[(r["proto"], r["when"], r["model"], r["arm"], r["task"])] = r
rows = list(best.values())

def table(rs, models, note=""):
    by = collections.defaultdict(dict)
    caps = collections.defaultdict(dict)
    for r in rs:
        k = (r["model"], r["ratio"], r["calib"], r["arm"], r["when"])
        by[k][r["task"]] = r["v"]
        caps[k][r["task"]] = r["cap"]
    tasks = [t for t in TASKS if any(t in v for v in by.values())]
    if not by: return
    print("| model | ratio | calib | arm | run | " + " | ".join(tasks) + " |")
    print("|" + "---|" * (len(tasks) + 5))
    for k in sorted(by, key=lambda k: (models.index(k[0]) if k[0] in models else 99, k[1], k[2])):
        cells = []
        for t in tasks:
            v = by[k].get(t)
            if v is None: cells.append("-"); continue
            c = caps[k].get(t)
            cells.append(f"{v:.2f}" + (f"<br><sub>cap {c:.0f}%</sub>" if c and c >= 20 else ""))
        print(f"| {k[0]} | {k[1]} | {k[2]} | `{k[3]}` | {k[4]} | " + " | ".join(cells) + " |")
    print()

MATH, CODE, WEB = "tulu-personas-math", "codealpaca", "fineweb"
CORPUS = {MATH: "allenai/tulu-3-sft-personas-math", CODE: "theblackcat102/evol-codealpaca-v1",
          WEB: "HuggingFaceFW/fineweb-edu"}

cur = [r for r in rows if r["proto"] == "offline" and r["model"] in MAIN4]
tasks_all = [t for t in TASKS if any(r["task"] == t for r in cur)]

def grid(calib):
    """{(model, ratio): {task: (value, cap)}} for one calibration corpus."""
    g = collections.defaultdict(dict)
    for r in cur:
        if r["calib"] == calib:
            g[(r["model"], r["ratio"])][r["task"]] = (r["v"], r["cap"])
    return g

def section(calib, title, blurb):
    g = grid(calib)
    print(f"## {title}\n")
    print(blurb + "\n")
    print("| model | ratio | " + " | ".join(tasks_all) + " |")
    print("|" + "---|" * (len(tasks_all) + 2))
    for k in sorted(g, key=lambda k: (MAIN4.index(k[0]), k[1])):
        cells = []
        for t in tasks_all:
            if t not in g[k]: cells.append("-"); continue
            v, c = g[k][t]
            cells.append(f"{v:.2f}" + (f"<br><sub>cap {c:.0f}%</sub>" if c and c >= 20 else ""))
        print(f"| {k[0]} | {k[1]} | " + " | ".join(cells) + " |")
    print()
    return g

print("# REAP accuracy — by calibration corpus\n")
print("4 models x 2 pruning ratios x 9 tasks, one protocol throughout:")
print("**0-shot, greedy (temperature 0.0), `max_tokens` 4096, offline batch-pinned at B=64.**")
print("Every number below was produced that way, so any two of them can be compared.\n")
print("The two sections differ in ONE thing: the corpus the router observations were")
print("collected on during calibration. That choice decides which experts survive pruning.\n")
print("> `cap N%` under a score = share of rows that hit the token cap and therefore")
print("> scored an automatic zero. Above ~20% the number measures termination, not accuracy.\n")

gm = section(MATH, "1. Calibrated on MATH — `" + CORPUS[MATH] + "`",
             "Run 2026-09-13/14. Arms `reap25_math256` / `reap50_math256`.")
gc = section(CODE, "2. Calibrated on CODE — `" + CORPUS[CODE] + "`",
             "Run 2026-09-12/13. Arms `reap25` / `reap50`.")
# The neutral control: general web prose, neither of the two capabilities the
# math-vs-code contrast is about. It is what says whether that contrast is
# "calibration picks a specialty" or just "some corpora prune better".
gw = section(WEB, "3. Calibrated on WEB TEXT — `" + CORPUS[WEB] + "`",
             "Run 2026-09-14. Arms `reap25_fw256` / `reap50_fw256`.")

print("## 4. Math minus code\n")
print("Same model, same ratio, same protocol -- the only difference is the corpus.")
print("Positive = math calibration scored higher.\n")
print("| model | ratio | " + " | ".join(tasks_all) + " |")
print("|" + "---|" * (len(tasks_all) + 2))
dl = collections.defaultdict(list)
for k in sorted(set(gm) & set(gc), key=lambda k: (MAIN4.index(k[0]), k[1])):
    cells = []
    for t in tasks_all:
        if t in gm[k] and t in gc[k]:
            d = gm[k][t][0] - gc[k][t][0]; dl[(k[1], t)].append(d); cells.append(f"{d:+.2f}")
        else:
            cells.append("-")
    print(f"| {k[0]} | {k[1]} | " + " | ".join(cells) + " |")
for ra in ("reap25", "reap50"):
    if not any(k[0] == ra for k in dl): continue
    print(f"| **mean** | **{ra}** | " + " | ".join(
        f"**{sum(dl[(ra,t)])/len(dl[(ra,t)]):+.2f}**" if dl.get((ra, t)) else "-"
        for t in tasks_all) + " |")
print()

print("## 5. Baseline reference — no pruning\n")
print("Un-pruned, same protocol, so sections 1 and 2 can each be read as a delta.\n")
base = []
for root in ("/home/PC/sweep2/results",
             "/home/PC/moe-eval-results/results/accuracy/moe_eval_offline",
             "/home/PC/moe-eval-results/results/accuracy/offline_sere"):
    for p2 in glob.glob(os.path.join(root, "*", "baseline", "*", "*", "scores.json")):
        try: doc = json.load(open(p2))
        except Exception: continue
        for d in (doc if isinstance(doc, list) else [doc]):
            sc = d.get("scores") or {}
            if not sc: continue
            m = d.get("meta", {})
            if m.get("mode") != "offline_batch_pinned": continue
            if (m.get("batching") or {}).get("requested_batch_size") not in (None, 64): continue
            metric = d.get("primary_metric") or list(sc)[0]
            v = (sc.get(metric) or list(sc.values())[0]).get("value")
            base.append((canon_model(os.path.relpath(p2, root).split(os.sep)[0]), d.get("task"), v))
bm = {}
for mo, t, v in base: bm[(mo, t)] = v
btasks = [t for t in TASKS if any(k[1] == t for k in bm)]
if bm:
    print("| model | " + " | ".join(btasks) + " |")
    print("|" + "---|" * (len(btasks) + 1))
    for mo in MAIN4:
        if not any(k[0] == mo for k in bm): continue
        print(f"| {mo} | " + " | ".join(f"{bm[(mo,t)]:.2f}" if (mo,t) in bm else "-" for t in btasks) + " |")
    print()


print("## Caveats\n")
print("- **Cap-hit dominates the low-ratio arms.** Where `cap%` is high the arm did not")
print("  produce scorable output; report it as that, not as an accuracy. Qwen3-30B aime")
print("  is ~95% capped at both ratios.")
print("- **Calibration corpus decides which capability survives pruning.** Section 3 is")
print("  the controlled A/B. At reap50, swapping codealpaca->math costs ~26 pts of lcb and")
print("  ~48 pts of humaneval+ on average while ADDING ~9 pts of mmlu_pro and holding math")
print("  flat; at reap25 the same swap is nearly a wash. Pruning keeps the experts the")
print("  calibration data activates, and only at 50% must the surviving set specialise.")
print("  Corroborated earlier on GLM, where prose->code calibration moved humaneval+ from")
print("  1.83 to 46.95 at the same 256 samples: sample count did not matter; domain did.")
print("- **Web text is the worst of the three corpora, not a neutral middle.** Section 3")
print("  sits at or below both task corpora almost everywhere, and collapses hardest where")
print("  a capability is narrow: code is near zero at reap25 (lcb 4.6-7.4, humaneval+ 1.2-18.3")
print("  outside gemma4) and GLM loses math outright (math_hard 2.04 at reap25, 0.30 at")
print("  reap50). Calibrating on prose does not preserve general ability -- it preserves")
print("  the experts prose happens to activate, which is a narrower set than either task")
print("  corpus selects. gemma-4 reap50 is the extreme: gsm8k 22.74 against 92.04 on math.")
print("- **Calibration BUDGET is not equalised across these rows.** The tulu-math arms used")
print("  8x32 @ L256 (256 samples); the codealpaca caches on this box are 32x8 @ L256 for")
print("  GLM (identical volume) and 4x32 @ L256 for qwen3-30b (half). Qwen3.6/gemma4")
print("  codealpaca caches are no longer on disk, so their budget is unverified. Both are")
print("  far below REAP's own default of 8x1024 @ L2048.")
print("- **aime has 4/60 rows with `\\frac` mangled** by formfeed corruption, which depresses")
print("  every absolute aime number. A/B comparisons within aime remain valid.")
print("- **Run-to-run noise is 0.5-1.1 pts** on greedy evals from batch-composition FP")
print("  effects alone. Single-run differences smaller than that are not readable.")
print("- **Calibration corpus is not recorded in result metadata.** It was recovered")
print("  from the checkpoint path in each cell's meta (`.../evol-codealpaca-v1/...` vs")
print("  `.../tulu-3-sft-personas-math/...`). Stamp it at run time so this stops being")
print("  an inference.")
