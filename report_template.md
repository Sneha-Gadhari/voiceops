# Capstone Report – ZentroMart Support Agent
1. Problem & dataset (ZentroMart, 52 Qs, 23 conversations, KB 21 docs, ref date 20 Oct 2026)
2. Architecture (diagram: user -> language/ASR -> agent LLM -> tools [KB search, catalog, order lookup, ticket, business-days] -> reply -> TTS)
3. Retrieval: chunk by [SECTION-ID], TF-IDF word+char, core rules always included, precedence KB-00.1, superseded docs (returns v3 vs v4)
4. Tools & safety: identity verification (IDV-1..4, one retry), escalation ESC-1 triggers, never claim ticket without created=True
5. Multilingual (Hindi/Hinglish): LLM rewrites query to English for search_kb; reply in customer language
6. Voice (Gnani STT/TTS over WebSocket): flow, latency, ASR-noise results
7. Evaluation: table of decision accuracy / must_include / must_not_include / evidence recall (from eval.py), per language
8. Error analysis: 3-5 failures + fixes
9. Limitations & future work
10. Links: GitHub repo, demo video/screenshots
