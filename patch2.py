p = "app.py"
lines = open(p, encoding="utf-8").read().splitlines()
out = []
for l in lines:
    if l.startswith('"""UI:'):
        l = "# UI: streamlit run app.py"
    if 'st.session_state.setdefault("key"' in l or 'st.text_input("LLM API key"' in l or 'os.environ["LLM_API_KEY"] = key' in l:
        continue
    out.append(l)
open(p, "w", encoding="utf-8").write("\n".join(out) + "\n")

p = "src/agent.py"
s = open(p, encoding="utf-8").read()
if "LLM_REASONING" not in s:
    s = s.replace("def _create(**kw):", 'def _create(**kw):\n    if os.environ.get("LLM_REASONING"):\n        kw["extra_body"] = {"reasoning_effort": os.environ["LLM_REASONING"]}', 1)
open(p, "w", encoding="utf-8").write(s)
print("done")