# Essay Assistant UI

## Run
1. Install Ollama, then: `ollama pull qwen2.5:3b` and `ollama pull nomic-embed-text` (TA2 needs the second one)
2. `ollama serve`
3. `pip install -r requirements.txt`
4. `python app.py` -> open http://127.0.0.1:5000

Without Ollama the TA1 tab still works using the built-in MockLLM; the TA2 tab shows an error.

## Change made to your code
ta1/essay_workflow.py: added `style: str` to `EssayState`. Without it LangGraph silently dropped the
"style" input, so the "polished" draft prompt was never used.
