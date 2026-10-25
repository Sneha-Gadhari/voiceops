from dotenv import load_dotenv
load_dotenv()
import streamlit as st
from src.agent import Session

st.set_page_config(page_title="ZentroMart Support", page_icon="🛒")
st.title("🛒 ZentroMart AI Support")

with st.sidebar:
    channel = st.selectbox("Channel", ["chat", "voice"])
    show = False
    if st.button("New conversation"):
        st.session_state.clear()
        st.rerun()

if "s" not in st.session_state:
    st.session_state.s, st.session_state.log = Session(channel), []
st.session_state.s.channel = channel

for m in st.session_state.log:
    with st.chat_message(m["role"]):
        st.write(m["text"])
        if show and m.get("meta"):
            st.caption(m["meta"])
        if m.get("audio"):
            st.audio(m["audio"])

def reply(msg, audio_heard=False):
    st.session_state.log.append({"role": "user", "text": msg})
    with st.chat_message("user"):
        st.write(msg)
    with st.spinner("Thinking..."):
        out = st.session_state.s.chat(msg)
    meta = f"decision={out['decision']} | evidence={out['evidence']} | tools={out['tool_calls']}"
    return out, meta

if channel == "voice":
    aud = st.audio_input("🎤 Speak")
    if aud is not None and st.session_state.get("last_aud") != aud.size:
        st.session_state.last_aud = aud.size
        from src import voice
        open("_in.wav", "wb").write(aud.getvalue())
        heard = voice.transcribe("_in.wav")
        if not heard.strip():
            heard, text, meta = "(unclear audio)", voice.RETRY, "decision=clarify"
        else:
            out, meta = reply(heard)
            text = out["reply"]
        voice.synthesize(text, "_reply.wav")
        st.session_state.log.append({"role": "assistant", "text": text, "meta": meta, "audio": open("_reply.wav", "rb").read()})
        st.rerun()

if msg := st.chat_input("Type your message (English / हिन्दी / Hinglish)"):
    out, meta = reply(msg)
    st.session_state.log.append({"role": "assistant", "text": out["reply"], "meta": meta})
    st.rerun()