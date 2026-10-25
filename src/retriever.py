"""Section-level KB retriever (TF-IDF word + char n-grams). Keeps [SECTION-ID], status, effective_from."""
import os, re, glob
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def _find_dataset():
    env = os.environ.get("DATASET_DIR")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for c in [env, os.path.join(root, "Capstone_Project_Dataset_FTTT_2026"), os.path.join(root, "dataset"),
              os.path.join(root, "..", "Capstone_Project_Dataset_FTTT_2026")]:
        if c and os.path.isdir(os.path.join(c, "kb")): return os.path.abspath(c)
    raise SystemExit("Dataset not found. Extract Capstone_Project_Dataset_FTTT_2026 next to run.py or set DATASET_DIR in .env")
DATASET = _find_dataset()
PRECEDENCE = {"policy": 1, "agent_guide": 2, "help": 3, "faq": 4, "marketing": 5, "meta": 2}

def load_chunks():
    chunks = []
    for path in sorted(glob.glob(os.path.join(DATASET, "kb", "*.md"))):
        raw = open(path, encoding="utf-8").read()
        m = re.match(r"---\n(.*?)\n---\n(.*)", raw, re.S)
        meta = dict(l.split(": ", 1) for l in m.group(1).splitlines() if ": " in l)
        body = m.group(2)
        parts = re.split(r"(?m)^## \[([^\]]+)\] ?(.*)$", body)
        for i in range(1, len(parts), 3):
            sid, title, text = parts[i], parts[i + 1], parts[i + 2].strip()
            chunks.append({
                "id": sid, "title": title, "text": text, "doc_id": meta.get("doc_id"),
                "doc_title": meta.get("title"), "doc_type": meta.get("doc_type"),
                "status": meta.get("status"), "effective_from": meta.get("effective_from"),
                "language": meta.get("language"), "file": os.path.basename(path),
            })
    return chunks

def render(c):
    return (f"[{c['id']}] {c['title']}  (doc={c['doc_title']}; type={c['doc_type']}; "
            f"status={c['status']}; effective_from={c['effective_from']})\n{c['text']}")

class Retriever:
    def __init__(self):
        self.chunks = load_chunks()
        docs = [f"{c['doc_title']} {c['title']} {c['text']}" for c in self.chunks]
        self.w = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, lowercase=True).fit(docs)
        self.c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit(docs)
        self.Xw, self.Xc = self.w.transform(docs), self.c.transform(docs)

    def search(self, query, k=6):
        sw = (self.Xw @ self.w.transform([query]).T).toarray().ravel()
        sc = (self.Xc @ self.c.transform([query]).T).toarray().ravel()
        score = 0.5 * sw + 0.5 * sc
        order = np.argsort(-score)[:k]
        return [dict(self.chunks[i], score=float(score[i])) for i in order]

    def get(self, ids):
        return [c for c in self.chunks if c["id"] in ids]
