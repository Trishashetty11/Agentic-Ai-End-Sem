"""
Essay Assistant UI  --  run:  python app.py   then open http://127.0.0.1:5000

TA1  : Draft -> Critique -> Revise (LangGraph, conditional edge)
TA2  : Writer/Reviewer/Editor loop with memory + RAG rubric (needs Ollama + nomic-embed-text)

If Ollama is not running, TA1 falls back to the built-in MockLLM so the UI still works.
"""
import os, sys, io, contextlib
from flask import Flask, jsonify, render_template, request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "ta1"))
sys.path.insert(0, os.path.join(HERE, "ta2"))

import essay_workflow as ta1
app = Flask(__name__)

# ---------- TA1 ----------
_llm_cache = {}
def ta1_llm():
    if "llm" not in _llm_cache:
        with contextlib.redirect_stdout(io.StringIO()):
            _llm_cache["llm"] = ta1.get_llm()          # None -> Ollama down
    return _llm_cache["llm"]

@app.get("/")
def index():
    return render_template("index.html")

@app.get("/api/status")
def status():
    real = ta1_llm() is not None
    return jsonify(ollama=real, mode="Ollama qwen2.5:3b" if real else "Mock model (Ollama not running)")

@app.post("/api/ta1")
def run_ta1():
    d = request.get_json(force=True)
    topic, style = d.get("topic", "").strip(), d.get("style", "quick")
    if not topic:
        return jsonify(error="Enter a topic"), 400
    llm = ta1_llm()
    mock = llm is None
    if mock:  # deterministic demo: pick flawed vs clean mock by style
        llm = ta1.MockLLM(kind="clean" if style == "polished" else "flawed")
    graph = ta1.build_graph(llm)
    steps, final = [], {}
    with contextlib.redirect_stdout(io.StringIO()):
        for upd in graph.stream({"topic": topic, "style": style, "messages": []}, stream_mode="updates"):
            for node, val in upd.items():
                val = {k: v for k, v in val.items() if k != "messages"}
                steps.append({"node": node, **val}); final.update(val)
    return jsonify(mock=mock, steps=steps,
                   final=final.get("revision", final.get("draft", "")),
                   revised="revision" in final)

# ---------- TA2 ----------
_ta2 = {}
def ta2_mod():
    if "m" not in _ta2:
        import essay_ta2                                  # connects to Ollama at import
        _ta2["m"] = essay_ta2
    return _ta2["m"]

@app.post("/api/ta2")
def run_ta2():
    d = request.get_json(force=True)
    topic, user = d.get("topic", "").strip(), (d.get("user") or "guest").strip()
    if not topic:
        return jsonify(error="Enter a topic"), 400
    try:
        m = ta2_mod()
    except Exception as e:
        return jsonify(error=f"TA2 needs Ollama running with qwen2.5:3b and nomic-embed-text "
                             f"({type(e).__name__}). Start `ollama serve` and pull both models."), 503
    init = {"topic": topic, "user": user, "revisions": 0, "hops": 0, "rounds": []}
    with contextlib.redirect_stdout(io.StringIO()):
        out = m.build().invoke(init, {"recursion_limit": 40})
    return jsonify(draft=out["draft"], critique=out.get("critique", ""), issues=out.get("issues"),
                   revisions=out["revisions"], rounds=out["rounds"], numbers=m.nums(out["draft"]),
                   prefs=m.prefs(user))

@app.post("/api/pref")
def pref():
    d = request.get_json(force=True)
    try:
        ta2_mod().set_pref(d["user"], d["k"], d["v"])
    except Exception as e:
        return jsonify(error=str(e)), 503
    return jsonify(ok=True)

if __name__ == "__main__":
    app.run(debug=False, port=5000)
