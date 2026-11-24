import csv, sys, time
sys.path.insert(0, "ta1")
from essay_workflow import build_graph, get_llm, _has_concrete_evidence, _hedge_count

TESTS = [
    ("T1", "the impact of remote work on productivity"),
    ("T2", "why the printing press mattered"),
    ("T3", "technology"),
    ("T4", "why sleep matters for students, in a formal tone under 80 words"),
    ("T5", "why renewable energy is growing"),
]

llm = get_llm()
if llm is None:
    sys.exit("Start Ollama first.")
app = build_graph(llm)

rows = []
for tid, topic in TESTS:
    t0 = time.time()
    s = app.invoke({"topic": topic, "style": "quick", "messages": []})
    final = s.get("revision", s["draft"])
    rows.append({
        "test": tid, "topic": topic,
        "issues_found": s["issues_found"],
        "revision_ran": "revision" in s,
        "evidence": _has_concrete_evidence(final),
        "hedges": _hedge_count(final),
        "words": len(final.split()),
        "seconds": round(time.time() - t0, 1),
        "final_text": final.replace("\n", " "),
    })

with open("ta1/results.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)
print("saved ta1/results.csv")