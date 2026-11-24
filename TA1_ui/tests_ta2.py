import csv, sys, time
sys.path.insert(0, "ta1"); sys.path.insert(0, "ta2")
from essay_workflow import _has_concrete_evidence, _hedge_count  # type: ignore
from essay_ta2 import build, nums  # type: ignore

TESTS = [  # same 5 topics as TA1; T4's constraint now comes from long-term memory (user ojal)
    ("T1", "the impact of remote work on productivity", "guest"),
    ("T2", "why the printing press mattered", "guest"),
    ("T3", "technology", "guest"),
    ("T4", "why sleep matters for students", "ojal"),
    ("T5", "why renewable energy is growing", "guest"),
]
app, rows, detail = build(), [], []
for tid, topic, user in TESTS:
    t0 = time.time()
    out = app.invoke({"topic": topic, "user": user, "revisions": 0, "hops": 0, "rounds": []},
                     {"recursion_limit": 40})
    final, R = out["draft"], out["rounds"]
    rows.append({"test": tid, "topic": topic, "user": user, "revisions": out["revisions"],
                 "converged": out["issues"] == 0, "evidence": _has_concrete_evidence(final),
                 "hedges": _hedge_count(final), "words": len(final.split()),
                 "numbers_to_verify": len(nums(final)), "tokens": sum(r["tokens"] for r in R),
                 "seconds": round(time.time() - t0, 1), "hops": out["hops"],
                 "final_text": final.replace("\n", " ")})
    detail += [{"test": tid, **r} for r in R]
    print(f"{tid} done: revisions={out['revisions']} converged={out['issues'] == 0}")

for name, data in (("results", rows), ("rounds", detail)):
    with open(f"ta2/{name}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=data[0].keys()); w.writeheader(); w.writerows(data)
print("saved ta2/results.csv and ta2/rounds.csv")