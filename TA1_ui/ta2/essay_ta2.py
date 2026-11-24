import os, re, sqlite3, sys, time
from typing import TypedDict
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ta1"))
from essay_workflow import _has_concrete_evidence, _hedge_count  # type: ignore

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_REV, MAX_HOPS = 3, 10
llm = ChatOllama(model="qwen2.5:3b", temperature=0.3)
RULES = [l.strip() for l in open(os.path.join(HERE, "..", "rubric.txt"), encoding="utf-8") if l.strip()]
DB = Chroma.from_texts(RULES, OllamaEmbeddings(model="nomic-embed-text"), collection_name="rubric")  # RAG
conn = lambda f: sqlite3.connect(os.path.join(HERE, f), check_same_thread=False)
MEM, CP = conn("memory.db"), SqliteSaver(conn("ckpt.db"))  # long-term memory, checkpointer
MEM.execute("CREATE TABLE IF NOT EXISTS prefs (user TEXT, k TEXT, v TEXT, PRIMARY KEY (user, k))")

def prefs(u): return dict(MEM.execute("SELECT k, v FROM prefs WHERE user=?", (u,)).fetchall())

def set_pref(u, k, v):
    MEM.execute("INSERT OR REPLACE INTO prefs VALUES (?,?,?)", (u, k, v)); MEM.commit()

def style(u):
    p = prefs(u)
    lim = f" HARD LIMIT: at most {p['max_words']} words." if "max_words" in p else ""
    return "Student style preferences: " + "; ".join(f"{k}={v}" for k, v in p.items()) + "." + lim if p else ""

def rules_for(text, k=5):
    r = "\n".join("- " + d.page_content for d in DB.similarity_search(text, k=k))
    print("[RAG rules]\n" + r)
    return r

def nums(t): return sorted({n.strip(".,-") for n in re.findall(r"\d[\d.,-]*%?", t)})

class S(TypedDict, total=False):
    topic: str; user: str; draft: str; critique: str; issues: int
    revisions: int; hops: int; last: str; next: str; rounds: list

def ask(prompt, para=False):
    t0 = time.time()
    r = llm.invoke([HumanMessage(content=prompt)])
    t = r.content.strip()
    if para: t = max(t.split("\n\n"), key=len).replace("*", "").strip()  # drop commentary
    return t, (r.usage_metadata or {}).get("total_tokens", 0), round(time.time() - t0, 1)

def step(s, node, rev, t, tok, sec, **kw):  # one per-round log row + state bookkeeping
    row = dict(node=node, rev=rev, issues=kw.get("issues"), tokens=tok, sec=sec, words=len(t.split()),
               sents=len(re.findall(r"[.!?](?:\s|$)", t)))
    return {"last": node, "hops": s["hops"] + 1, "rounds": s["rounds"] + [row], **kw}

def writer(s):
    t, tok, sec = ask(f"Write one paragraph (4-6 sentences) on: {s['topic']}. {style(s['user'])} "
                      "Use only well-known facts. Never cite a study, journal, report or statistic unless it is famous and you are certain it is real. "
                       "Return only the paragraph.", True)
    return step(s, "writer", 0, t, tok, sec, draft=t)

def reviewer(s):
    d, lim = s["draft"], prefs(s["user"]).get("max_words", "")
    no_ev, vague = not _has_concrete_evidence(d), _hedge_count(d) >= 3
    long = lim.isdigit() and len(d.split()) > int(lim)
    cites = bool(re.search(r"\b(a study (by|from|at|published|found|showed)|studies (show|found|have)|journal of|according to|published in|researchers? (found|at)|(university|institute|school) (study|found))\b", d, re.I))
    sents = len(re.findall(r"[.!?](?:\s|$)", d))
    bad_len = not 4 <= sents <= 6
    n = vague + long + cites + bad_len   # missing evidence is no longer a hard failure
    if n == 0:
        return step(s, "reviewer", s["revisions"], d, 0, 0, critique="No issues found.", issues=0)
    q = " ".join(m for m, f in [("missing specific date, number or named example.", no_ev),
                                ("vague hedging words.", vague),
                                ("over the word limit, style preferences.", long),
                                ("cites a study or journal that may be invented; do not invent sources.", cites),
                                ("must be between 4 and 6 sentences.", bad_len)] if f)
    t, tok, sec = ask("Rubric rules:\n" + rules_for(q or d) + f"\n{style(s['user'])}\n\nProblems found: {q}\n\n"
                      "List 2 short bullets naming which rubric rules this paragraph breaks and why. "
                      "Bullets only.\n\nParagraph:\n" + d)
    return step(s, "reviewer", s["revisions"], d, tok, sec, critique=t, issues=n)

def editor(s):
    t, tok, sec = ask("Rewrite the paragraph to fix the critique. Follow these rubric rules:\n" + rules_for(s["critique"])
                      + f"\n{style(s['user'])}\n\nParagraph:\n" + s["draft"] + "\n\nCritique:\n" + s["critique"]
                      + "\n\nDo not invent studies, journals, sources or statistics. If you are not certain a number is real, "
  "use a clear example or a general claim instead. Return only the paragraph.", True)
    return step(s, "editor", s["revisions"] + 1, t, tok, sec, draft=t, revisions=s["revisions"] + 1)

def supervisor(s):
    last = s.get("last")
    if s["hops"] >= MAX_HOPS: nxt = "end"
    elif last is None: nxt = "writer"
    elif last in ("writer", "editor"): nxt = "reviewer"
    else: nxt = "editor" if s["issues"] and s["revisions"] < MAX_REV else "end"
    return {"next": nxt}

def build(cp=None):  # with a checkpointer the graph pauses for human approval before every edit
    g = StateGraph(S)
    for f in (supervisor, writer, reviewer, editor): g.add_node(f.__name__, f)
    g.set_entry_point("supervisor")
    g.add_conditional_edges("supervisor", lambda s: s["next"],
                            {"writer": "writer", "reviewer": "reviewer", "editor": "editor", "end": END})
    for n in ("writer", "reviewer", "editor"): g.add_edge(n, "supervisor")
    return g.compile(checkpointer=cp, interrupt_before=["editor"] if cp else [])

def show(app, cfg):
    st = app.get_state(cfg); v = st.values
    print("\n== " + ("PAUSED before " + st.next[0] + " (approval needed)" if st.next else "DONE"))
    print(v["draft"], "\ncritique:", v.get("critique"), "\nnumbers to verify:", nums(v["draft"]))
    for r in v["rounds"]: print(r)

if __name__ == "__main__":
    a = sys.argv[1:] + [None] * 4
    cfg = lambda t: {"configurable": {"thread_id": t}, "recursion_limit": 40}
    if a[0] == "set": set_pref(a[1], a[2], a[3]); print("saved", a[1:4])
    elif a[0] == "resume":  # approve the pending edit and continue from the saved checkpoint
        app = build(CP); app.invoke(None, cfg(a[1])); show(app, cfg(a[1]))
    else:  # essay_ta2.py "<topic>" [user] [thread]: giving a thread id turns on pause/resume
        topic, user, tid = a[0] or "the impact of remote work on productivity", a[1] or "guest", a[2]
        init = {"topic": topic, "user": user, "revisions": 0, "hops": 0, "rounds": []}
        if tid:
            app = build(CP); app.invoke(init, cfg(tid)); show(app, cfg(tid))
        else:
            out = build().invoke(init, {"recursion_limit": 40})
            print(out["draft"], "\nnumbers to verify:", nums(out["draft"]))
            for r in out["rounds"]: print(r)