"""Step 4: Gnani STT -> agent -> Gnani TTS (official SDK: pip install gnani-vachana). Key in env GNANI_API_KEY."""
import os, re
from dotenv import load_dotenv
load_dotenv()

STT_LANG = os.environ.get("STT_LANG", "en-IN")     # en-IN also accepts English+Hindi mix (Latin script)
TTS_VOICE = os.environ.get("TTS_VOICE", "Kaveri")
TTS_MODEL = os.environ.get("TTS_MODEL", "timbre-v2.5")  # always name the model
_stt = _tts = None

def _clients():
    global _stt, _tts
    if _stt is None:
        from gnani.stt import GnaniSTTClient
        from gnani.tts import GnaniTTSClient
        _stt, _tts = GnaniSTTClient(), GnaniTTSClient()
    return _stt, _tts

DIGITS = {"zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
          "six": "6", "seven": "7", "eight": "8", "nine": "9"}
def normalize_asr(t):
    """'order eight nine seven eight' -> 'order 8978'. Joins runs of spoken/typed single digits."""
    words = [DIGITS.get(w.lower().strip(",."), w) for w in t.split()]
    out, run = [], []
    for w in words + [""]:
        if len(w) == 1 and w.isdigit(): run.append(w)
        else:
            if run: out.append("".join(run)); run = []
            if w: out.append(w)
    return " ".join(out)

def spoken(reply):
    """Write for the ear: group 8-digit order IDs, amounts as rupees."""
    reply = re.sub(r"\b(\d{4})(\d{4})\b", r"\1 \2", reply)
    reply = re.sub(r"[*_`#>]", "", reply)
    reply = re.sub(r"(?:₹|Rs\.?\s?)\s?(\d[\d,]*)", lambda m: m.group(1).replace(",", "") + " rupees", reply)
    return reply

def tts_lang(reply):
    return "hi-IN" if re.search(r"[\u0900-\u097F]", reply) else "en-IN"

def transcribe(path, lang=None):
    stt, _ = _clients()
    try: return normalize_asr(stt.transcribe(path, language_code=lang or STT_LANG).get("transcript", ""))
    except Exception as e:
        print("STT error:", e); return ""

def synthesize(text, out_file, lang=None):
    _, tts = _clients()
    tts.synthesize(spoken(text), voice=TTS_VOICE, model=TTS_MODEL, language=lang or tts_lang(text), output_file=out_file)
    return out_file

RETRY = "Sorry, I didn't catch that. Could you say it again?"
def voice_turn(session, audio_path, out_wav="reply.wav", lang=None):
    heard = transcribe(audio_path, lang)
    if not heard.strip():
        out = {"decision": "clarify", "reply": RETRY, "evidence": [], "tool_calls": []}
    else:
        out = session.chat(heard)
    synthesize(out["reply"], out_wav)
    return heard, out
