import csv, re, sys
sys.path.insert(0, "ta1"); sys.path.insert(0, "ta2")
from essay_workflow import _has_concrete_evidence, _hedge_count  # type: ignore
from essay_ta2 import writer, reviewer, editor, nums  # type: ignore

TESTS = [
    ("T1", "the impact of remote work on productivity", "guest"),
    ("T2", "why the printing press mattered", "guest"),
    ("T3", "technology", "guest"),
    ("T4", "why sleep matters for students", "ojal"),
    ("T5", "why renewable energy is growing", "guest"),
]
rows = []
for tid, topic, user in TESTS:
    s = {"topic": topic, "user": user, "revisions": 0, "hops": 0, "rounds": []}
    s.update(writer(s))
    for rnd in range(4):  # round 0 = first draft, rounds 1-3 = after each forced edit
        s.update(reviewer(s))  # scores the current draft
        d = s["draft"]
        rows.append({"test": tid, "round": rnd, "issues": s["issues"], "evidence": _has_concrete_evidence(d),
                     "hedges": _hedge_count(d), "words": len(d.split()),
                     "sents": len(re.findall(r"[.!?](?:\s|$)", d)), "numbers": len(nums(d)),
                     "cum_tokens": sum(r["tokens"] for r in s["rounds"]), "text": d.replace("\n", " ")})
        print(f"{tid} round {rnd}: issues={s['issues']} words={len(d.split())} numbers={len(nums(d))}")
        if rnd < 3: s.update(editor(s))  # edit even if issues == 0

with open("rounds_experiment.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print("saved rounds_experiment.csv")