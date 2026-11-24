import csv, re

def load(p): return {r["test"]: r for r in csv.DictReader(open(p, encoding="utf-8"))}
def nums(t): return len({n.strip(".,-") for n in re.findall(r"\d[\d.,-]*%?", t)})

a, b = load("ta1/results.csv"), load("ta2/results.csv")
cols = ["test", "TA1_revised", "TA1_words", "TA1_numbers", "TA1_sec",
        "TA2_revisions", "TA2_converged", "TA2_words", "TA2_numbers", "TA2_tokens", "TA2_sec"]
rows = []
for t in a:
    x, y = a[t], b.get(t, {})
    rows.append([t, x["revision_ran"], x["words"], nums(x["final_text"]), x["seconds"],
                 y.get("revisions", "-"), y.get("converged", "-"), y.get("words", "-"),
                 y.get("numbers_to_verify", "-"), y.get("tokens", "-"), y.get("seconds", "-")])

print(" | ".join(cols))
for r in rows: print(" | ".join(str(c) for c in r))
with open("comparison.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(cols); w.writerows(rows)
print("saved comparison.csv")