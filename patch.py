import re
p = "src/agent.py"
s = open(p, encoding="utf-8").read()
s = s.replace("r = cl.chat.completions.create(model=MODEL(), temperature=0,", "r = _create(temperature=0,")
s = s.replace("r = _client().chat.completions.create(model=MODEL(), temperature=0,", "r = _create(temperature=0,")
if "def _create" not in s:
    s = s.replace("class Session:", '''import time
def _create(**kw):
    models = [MODEL()] + [m for m in os.environ.get("LLM_FALLBACK", "").split(",") if m]
    for attempt in range(8):
        for m in models:
            try:
                return _client().chat.completions.create(model=m, **kw)
            except Exception as e:
                last = e
                print("retry:", str(e)[:80])
        time.sleep(5 * (attempt + 1))
    raise last

class Session:''', 1)
open(p, "w", encoding="utf-8").write(s)
e = open(".env", encoding="utf-8").read()
if "LLM_FALLBACK" not in e:
    open(".env", "a", encoding="utf-8").write("\nLLM_FALLBACK=gemini-3.5-flash-lite,gemini-3.7-flash\n")
print("patched")