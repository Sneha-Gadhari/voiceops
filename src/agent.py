"""ZentroMart support agent: RAG + tool calling over any OpenAI-compatible API (OpenAI, Gemini, Groq, OpenRouter...).
Env: LLM_API_KEY, LLM_BASE_URL (optional), LLM_MODEL."""
import os, sys, re, json, csv
from dotenv import load_dotenv
load_dotenv()
from datetime import date, datetime, timedelta
from openai import OpenAI
from src.retriever import Retriever, render, DATASET

sys.path.insert(0, DATASET)
from tools import tools as mock_tools  # lookup_order, create_ticket (mock, identity-gated)

TODAY = date(2026, 10, 20)
CORE_PREFIX = ("KB-00", "ESC-", "IDV-", "STY-")

def _read(name):
    with open(os.path.join(DATASET, "data", name), encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))
PRODUCTS = {r["sku"]: r for r in _read("products.csv")}
PINS = {r["pincode"]: r for r in _read("pincodes.csv")}
HOLIDAYS = {date.fromisoformat(r["date"]) for r in _read("holidays.csv")}

RET = Retriever()
CORE = [c for c in RET.chunks if c["id"].startswith(CORE_PREFIX)]

def is_bday(d): return d.weekday() != 6 and d not in HOLIDAYS
def add_business_days(start, n):
    d = date.fromisoformat(start)
    while n > 0:
        d += timedelta(days=1)
        if is_bday(d): n -= 1
    return d.isoformat()
def count_business_days(start, end):
    """Business days elapsed after `start` up to and including `end`."""
    a, b, n = date.fromisoformat(start), date.fromisoformat(end), 0
    while a < b:
        a += timedelta(days=1)
        n += is_bday(a)
    return n

SYSTEM = f"""You are ZentroMart's AI support assistant (fictional Indian online store). Today is {TODAY.isoformat()} (use this, not the system clock).
ANSWER ONLY from the knowledge base (KB) excerpts, catalog/order tool results. Never invent policy, numbers, or promises.
Precedence when documents disagree: current policy > agent guide > help > FAQ > marketing. Superseded docs apply only where their successor says so. Check effective_from vs order/delivery dates.
Workflow:
1. General policy question: answer, no verification needed. Use search_kb (write the query in English, even if customer writes Hindi/Hinglish) and get_product when a SKU/product is named (it gives return_category; then search the return rule for that category).
2. Anything about a specific order: ask for order ID, then ask ONE verification value (last 4 digits of registered phone OR delivery pincode). Call lookup_order(order_id) with no proof first, then with verification_value only after customer gives it. Allow one retry only. Never reveal expected values, full phone, email, address.
3. Dates/timelines: use add_business_days / count_business_days tools (Mon-Sat, skip holidays). Return windows are calendar days.
4. Escalate (create_ticket FIRST, then tell customer a colleague will contact them; never promise outcome/amount/time beyond policy) per ESC-1: >Rs10,000 damaged/wrong item, unauthorised activity, legal threat, 3rd+ contact in 7 days (contacts_last_7_days>=2), refund >3 business days past max timeline, customer asks for human, safety risk. Only say ticket created if tool returned created=True. Ticket summary per ESC-3.
4b. NEVER escalate or create a ticket unless an ESC-1 trigger is clearly true. Do not assume the customer asked for a human; they must say so explicitly. A normal policy question is just answered.
5. If KB has no answer: say you don't have that information and offer a human colleague (decision "escalate" or "answer" with that offer). Do not guess.
6. Style: first reply introduces you as AI assistant. Chat: 2-3 short sentences. Reply in customer's language (Hindi/Hinglish/English etc). One question at a time. Dates with month name. Voice channel (STY-3): no markdown/lists/emoji; 2-3 short sentences; always include the year in dates; amounts in words ("forty-nine rupees"); order IDs in groups of four digits.
Final output: ONLY a JSON object {{"decision": "answer|clarify|escalate|refuse", "reply": "<text to customer>", "evidence": ["SECTION-ID", "products.csv:SKU", ...]}} listing every section/catalog row you relied on.

CORE RULES:
""" + "\n\n".join(render(c) for c in CORE)

TOOLS = [
 {"type": "function", "function": {"name": "search_kb", "description": "Search the knowledge base. Query in English. Returns sections with IDs, status, effective_from.",
  "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
 {"type": "function", "function": {"name": "get_product", "description": "Catalog row by SKU (return_category, price, seller, assured, warranty_months).",
  "parameters": {"type": "object", "properties": {"sku": {"type": "string"}}, "required": ["sku"]}}},
 {"type": "function", "function": {"name": "find_product", "description": "Find SKUs by product name text.",
  "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}}},
 {"type": "function", "function": {"name": "check_pincode", "description": "Serviceability, COD and Express availability for a pincode.",
  "parameters": {"type": "object", "properties": {"pincode": {"type": "string"}}, "required": ["pincode"]}}},
 {"type": "function", "function": {"name": "lookup_order", "description": "Order lookup. Without verification_value returns verification prompt only. verification_value = customer-supplied 4-digit phone suffix OR 6-digit pincode.",
  "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "verification_value": {"type": "string"}}, "required": ["order_id"]}}},
 {"type": "function", "function": {"name": "create_ticket", "description": "Human handover ticket. Summary must follow ESC-3.",
  "parameters": {"type": "object", "properties": {"summary": {"type": "string"}, "order_id": {"type": "string"}, "reason": {"type": "string"}, "intent": {"type": "string"}}, "required": ["summary"]}}},
 {"type": "function", "function": {"name": "add_business_days", "description": "Date after n business days (Mon-Sat, no holidays) from start YYYY-MM-DD.",
  "parameters": {"type": "object", "properties": {"start": {"type": "string"}, "n": {"type": "integer"}}, "required": ["start", "n"]}}},
 {"type": "function", "function": {"name": "count_business_days", "description": "Business days elapsed from start to end (YYYY-MM-DD).",
  "parameters": {"type": "object", "properties": {"start": {"type": "string"}, "end": {"type": "string"}}, "required": ["start", "end"]}}},
]

def _client():
    return OpenAI(api_key=os.environ["LLM_API_KEY"], base_url=os.environ.get("LLM_BASE_URL") or None, max_retries=8, timeout=60)
MODEL = lambda: os.environ.get("LLM_MODEL", "gpt-4o-mini")

import time
def _create(**kw):
    if os.environ.get("LLM_REASONING"):
        kw["extra_body"] = {"reasoning_effort": os.environ["LLM_REASONING"]}
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

class Session:
    def __init__(self, channel="chat"):
        self.channel, self.msgs, self.fail = channel, [], 0
        self.evidence, self.calls, self.verified = set(), [], {}

    def _run_tool(self, name, a):
        self.calls.append(f"{name}({', '.join(f'{k}={v}' for k, v in a.items() if k != 'summary')})")
        if name == "search_kb":
            hits = RET.search(a["query"], 6)
            return "\n\n".join(render(c) for c in hits)
        if name == "get_product":
            p = PRODUCTS.get(a["sku"].upper())
            if not p: return "SKU not found"
            self.evidence.add(f"products.csv:{p['sku']}")
            return json.dumps(p, ensure_ascii=False)
        if name == "find_product":
            n = a["name"].lower()
            return json.dumps([{"sku": s, "title": p["title"]} for s, p in PRODUCTS.items() if n in p["title"].lower()][:5])
        if name == "check_pincode": return json.dumps(PINS.get(a["pincode"], "not in service list"))
        if name == "add_business_days": return add_business_days(a["start"], a["n"])
        if name == "count_business_days": return str(count_business_days(a["start"], a["end"]))
        if name == "create_ticket": return json.dumps(mock_tools.create_ticket(**a))
        if name == "lookup_order":
            oid = re.sub(r"\D", "", a["order_id"])
            if a.get("verification_value"):
                if self.fail >= 2:
                    return json.dumps({"verified": False, "message": "Verification locked: do not share details. Offer general policy help; suggest contacting from registered phone."})
                r = mock_tools.lookup_order(oid, a["verification_value"])
                if r.get("verified"): self.verified[oid] = True
                else: self.fail += 1
                return json.dumps(r, default=str, ensure_ascii=False)
            return json.dumps(mock_tools.lookup_order(oid))
        return "unknown tool"

    def chat(self, user_msg, max_steps=8):
        self.calls = []
        pre = RET.search(user_msg, 4)  # cheap first-pass retrieval on raw text
        for c in pre: self.evidence.add(c["id"])
        ctx = "Auto-retrieved KB (may be incomplete; call search_kb for more):\n" + "\n\n".join(render(c) for c in pre)
        self.msgs.append({"role": "user", "content": f"[channel={self.channel}]\n{user_msg}\n\n{ctx}"})
        cl = _client()
        for _ in range(max_steps):
            r = _create(temperature=0,
                messages=[{"role": "system", "content": SYSTEM}] + self.msgs, tools=TOOLS)
            m = r.choices[0].message
            self.msgs.append(m.model_dump(exclude_none=True))
            if not m.tool_calls: break
            for tc in m.tool_calls:
                try: out = self._run_tool(tc.function.name, json.loads(tc.function.arguments or "{}"))
                except Exception as e: out = f"tool error: {e}"
                for sid in re.findall(r"^\[([A-Za-z0-9.\-]+)\]", out, re.M): self.evidence.add(sid)
                self.msgs.append({"role": "tool", "tool_call_id": tc.id, "content": out})
        txt = m.content or ""
        j = re.search(r"\{.*\}", txt, re.S)
        try: res = json.loads(j.group(0))
        except Exception: res = {"decision": "answer", "reply": txt, "evidence": []}
        res["evidence"] = sorted(set(res.get("evidence", [])))
        res["tool_calls"] = list(self.calls)
        # keep history compact: store only the final reply for next turns' context
        self.msgs = [x for x in self.msgs if x["role"] in ("user",) or (x["role"] == "assistant" and not x.get("tool_calls"))]
        return res

BASELINE_SYSTEM = ("You are ZentroMart's AI customer support assistant for an Indian online store. Introduce yourself as an AI assistant "
  "in your first reply. Answer the customer's question helpfully and briefly. Reply in the customer's language. "
  'Output ONLY JSON: {"decision":"answer|clarify|escalate|refuse","reply":"...","evidence":[]}')

def baseline(question):
    """Step 1: LLM only, no documents, no tools."""
    r = _create(temperature=0,
        messages=[{"role": "system", "content": BASELINE_SYSTEM}, {"role": "user", "content": question}])
    txt = r.choices[0].message.content or ""
    j = re.search(r"\{.*\}", txt, re.S)
    try: res = json.loads(j.group(0))
    except Exception: res = {"decision": "answer", "reply": txt}
    res.update(evidence=[], tool_calls=[]); return res
