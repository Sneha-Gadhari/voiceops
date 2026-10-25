# VoiceOps 2026 – ZentroMart Support Agent
Public repo: https://github.com/Sneha-Gadhari/voiceops

## Install
```
pip install -r requirements.txt
cp .env.example .env      # fill LLM_API_KEY, GNANI_API_KEY
```
Extract `Capstone_Project_Dataset_FTTT_2026/` next to `run.py` (or set `DATASET_DIR`). Dataset is not included here.

## Run
- Batch (organizers): `python run.py --audio_dir ./audio --out sample_outputs`  (or `sh run.sh ./audio`)
  Saves per file: transcript, reply text, reply .wav, retrieved source chunks, tool calls.
- UI chat + mic: `streamlit run app.py`  (Channel = voice for mic)
- Evaluation Step 1 vs Step 2: `python -m src.eval both` (`--limit 10` for quick run)
- Notebook: `capstone.ipynb`

## Design
Step 1 prompt-only baseline (`baseline()`). Step 2 section-level TF-IDF retrieval keeping [SECTION-ID]/status/effective_from + core rules always in prompt.
Step 3 tool-calling agent: search_kb, get_product, check_pincode, lookup_order, create_ticket, business-day calculators; verification retry limit enforced in code.
Step 4 Gnani STT (gnani-vachana SDK) -> agent -> Gnani TTS; empty transcript -> "didn't catch that"; replies normalized for the ear.
