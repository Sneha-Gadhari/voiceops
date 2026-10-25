"""Batch run for organizers.  python run.py --audio_dir ./audio --out sample_outputs
Per audio file: STT -> agent (fresh session) -> TTS. Saves <name>.json (heard, reply, decision, retrieved chunks, tool calls) + <name>.wav."""
import argparse, json, os, glob
from src.agent import Session
from src import voice

ap = argparse.ArgumentParser()
ap.add_argument("--audio_dir", required=True)
ap.add_argument("--out", default="sample_outputs")
ap.add_argument("--lang", default=None, help="STT language code, e.g. en-IN, hi-IN (default env STT_LANG)")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
files = sorted(f for ext in ("wav", "mp3", "m4a", "flac", "ogg") for f in glob.glob(os.path.join(a.audio_dir, f"*.{ext}")))
rows = []
for f in files:
    name = os.path.splitext(os.path.basename(f))[0]
    try:
        heard, out = voice.voice_turn(Session("voice"), f, os.path.join(a.out, name + ".wav"), a.lang)
    except Exception as e:  # never crash the batch
        heard, out = "", {"decision": "clarify", "reply": voice.RETRY, "evidence": [], "tool_calls": [], "error": str(e)}
        try: voice.synthesize(out["reply"], os.path.join(a.out, name + ".wav"))
        except Exception: pass
    rec = {"file": os.path.basename(f), "transcript": heard, "reply_text": out["reply"], "decision": out["decision"],
           "retrieved_sources": out["evidence"], "tool_calls": out["tool_calls"], "reply_audio": name + ".wav"}
    json.dump(rec, open(os.path.join(a.out, name + ".json"), "w"), ensure_ascii=False, indent=2)
    rows.append(rec); print(name, "->", out["decision"], out["evidence"])
json.dump(rows, open(os.path.join(a.out, "all_results.json"), "w"), ensure_ascii=False, indent=2)
print(f"done: {len(rows)} files -> {a.out}")
