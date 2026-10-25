p = "src/agent.py"
s = open(p, encoding="utf-8").read()
if "NEVER escalate" not in s:
    s = s.replace("5. If KB has no answer", "4b. NEVER escalate or create a ticket unless an ESC-1 trigger is clearly true. Do not assume the customer asked for a human; they must say so explicitly. A normal policy question is just answered.\n5. If KB has no answer", 1)
if "LLM_REASONING" not in s:
    s = s.replace("def _create(**kw):", 'def _create(**kw):\n    if os.environ.get("LLM_REASONING"):\n        kw["extra_body"] = {"reasoning_effort": os.environ["LLM_REASONING"]}', 1)
open(p, "w", encoding="utf-8").write(s)
print("done")