"""Run practice sets. Usage: python eval.py [questions|conversations|both] [--asr] [--limit N]"""
import sys, json, pandas as pd
from src.agent import Session, DATASET, baseline

def facts_ok(reply, must):
    r = reply.lower()
    return all(any(alt.strip().lower() in r for alt in f.split(" / ")) for f in must.split(" ; ") if f.strip())
def bad_hit(reply, mn):
    r = reply.lower()
    return [p for p in mn.split(" ; ") if p.strip() and p.strip().lower() in r]
def ev_recall(got, gold):
    g = {x.strip() for x in gold.split(";") if x.strip()}
    return (len(g & set(got)) / len(g)) if g else None

def run_questions(asr=False, limit=None, mode="rag"):
    q = pd.read_csv(f"{DATASET}/practice/questions.csv", dtype=str).fillna("")
    rows = []
    for _, r in q.head(limit or len(q)).iterrows():
        s = Session()
        qtxt = r.asr_text if asr else r.question
        out = baseline(qtxt) if mode == "baseline" else s.chat(qtxt)
        acc = {d.strip() for d in r.acceptable_decisions.split(";")} | {r.expected_decision}
        rows.append(dict(id=r.id, lang=r.language, decision_ok=out["decision"] in acc,
            facts_ok=facts_ok(out["reply"], r.must_include), safe=not bad_hit(out["reply"], r.must_not_include),
            ev_recall=ev_recall(out["evidence"] + sorted(s.evidence), r.gold_evidence) if mode == "rag" else 0.0, reply=out["reply"]))
    d = pd.DataFrame(rows); d["passed"] = d.decision_ok & d.facts_ok & d.safe
    return d

def run_conversations(limit=None):
    c = pd.read_csv(f"{DATASET}/practice/conversations.csv", dtype=str).fillna("")
    rows = []
    for cid, g in list(c.groupby("conversation_id"))[:limit]:
        s = Session()
        for _, r in g.sort_values("turn", key=lambda x: x.astype(int)).iterrows():
            out = s.chat(r.customer_message)
            exp_tools = [t.split("(")[0] for t in r.expected_tool_calls.split(") ") if t.strip()] if r.expected_tool_calls else []
            rows.append(dict(conv=cid, turn=r.turn, decision_ok=out["decision"] == r.expected_decision,
                facts_ok=facts_ok(out["reply"], r.must_include), safe=not bad_hit(out["reply"], r.must_not_include),
                tools_ok=all(any(t in c_ for c_ in out["tool_calls"]) for t in exp_tools), reply=out["reply"]))
    return pd.DataFrame(rows)

if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "both"
    lim = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    if what in ("questions", "both"):
        for mode in ("baseline", "rag"):   # Step 1 vs Step 2
            d = run_questions("--asr" in sys.argv, lim, mode); d.to_csv(f"results_questions_{mode}.csv", index=False)
            print(f"QUESTIONS [{mode}]", d[["passed", "decision_ok", "facts_ok", "safe", "ev_recall"]].mean(numeric_only=True).round(3).to_dict())
    if what in ("conversations", "both"):
        d = run_conversations(lim); d.to_csv("results_conversations.csv", index=False)
        print("CONVERSATIONS", d[["decision_ok", "facts_ok", "safe", "tools_ok"]].mean().round(3).to_dict())
