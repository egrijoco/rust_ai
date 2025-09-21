# rag.py – E:\Rust_Ai\data-ból épített index használata
import os
import json
import re
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# FAISS vagy sklearn fallback
try:
    import faiss  # type: ignore
    HAVE_FAISS = True
except Exception:
    HAVE_FAISS = False
    from sklearn.neighbors import NearestNeighbors  # type: ignore

# BOW
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import load_npz
from sklearn.preprocessing import normalize

# Fuzzy
from rapidfuzz import process as rf_process, fuzz as rf_fuzz

load_dotenv()

BASE = Path(__file__).parent.resolve()
INDEX_DIR = BASE / "index"
META_FILE = INDEX_DIR / "meta.json"
FAISS_FILE = INDEX_DIR / "faiss.index"
SK_VECS_FILE = INDEX_DIR / "sk_vectors.npy"
BOW_MATRIX_FILE = INDEX_DIR / "tfidf_matrix.npz"
BOW_VOCAB_FILE = INDEX_DIR / "tfidf_vocab.json"
LEXICON_FILE = INDEX_DIR / "lexicon.json"

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
RAG_SHOW_SOURCES = os.getenv("RAG_SHOW_SOURCES", "0").strip().lower() in ("1","true","yes")

SYSTEM_PROMPT_PATH = BASE / "prompts" / "system_hu.txt"
SYSTEM_PROMPT = (
    SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    if SYSTEM_PROMPT_PATH.exists()
    else (
        "Magyarul válaszolj. Légy tömör és pontos. Ha a forrás hiányos vagy bizonytalan, mondd ki. "
        "Ne találj ki forrásokat"
    )
)

def _norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\(.*?\)", "", s)
    s = re.sub(r"[^a-z0-9áéíóöőúüű \-]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

# --- Kétnyelvű lekérdezés-bővítés (HU→EN alias + opcionálisan Argos Translate) ---
HU_EN_ALIASES: Dict[str, List[str]] = {
    "akkumulátor": ["battery", "small battery", "large battery", "akku"],
    "napelem": ["solar panel", "large solar panel", "solar"],
    "szélturbina": ["wind turbine", "turbine"],
    "elosztó": ["power splitter", "splitter"],
    "ágaztató": ["electrical branch", "branch"],
    "blokkoló": ["power blocker", "blocker"],
    "okos kapcsoló": ["smart switch"],
    "kapcsoló": ["switch"],
    "időzítő": ["timer"],
    "kombinátor": ["root combiner", "combiner"],
    "újrahasznosítás": ["recycle", "recycling"],
    "kutatás": ["research", "research cost", "scrap"],
    "kemence": ["furnace", "smelting"],
    "kén": ["sulfur"],
    "robbanótöltet": ["c4", "explosive", "satchel", "explosive 5.56"],
    "ajtó": ["door"],
    "páncélozott": ["armored"],
    # kapacitás témához
    "kapacitás": ["capacity", "charge capacity", "storage capacity"],
    "töltési ráta": ["charge rate", "charging rate"],
    "kimenet": ["output", "max output", "power output"],
}

def expand_query_bilingual(q: str) -> str:
    """
    1) kézi HU→EN aliasok hozzáadása
    2) ha elérhető az Argos Translate HU→EN, a fordítást is hozzácsapja
    Visszaad: kibővített lekérdezés egyetlen stringben.
    """
    qn = _norm(q)
    terms = [q]  # eredeti kérdés mindig marad
    # kézi aliasok
    for hu, eng_list in HU_EN_ALIASES.items():
        if hu in qn:
            terms += eng_list
    # opcionális offline fordító (Argos Translate)
    try:
        from argostranslate import translate  # type: ignore
        installed = translate.get_installed_languages()
        hu_lang = next((l for l in installed if (l.code or "").startswith("hu")), None)
        en_lang = next((l for l in installed if (l.code or "").startswith("en")), None)
        if hu_lang and en_lang:
            tr = translate.Translation(hu_lang, en_lang)
            en_q = tr.translate(q)
            if en_q and en_q.lower() not in {t.lower() for t in terms}:
                terms.append(en_q)
    except Exception:
        pass
    # egyedi, tisztított
    compact = " ".join(sorted(set(t.strip() for t in terms if t.strip())))
    return compact

class Retriever:
    def __init__(self):
        if not META_FILE.exists():
            raise FileNotFoundError("Hiányzik az index/meta.json. Futtasd: python ingest.py")
        meta = json.loads(META_FILE.read_text(encoding="utf-8"))
        self.docs = meta["docs"]
        self.settings = meta.get("settings", {})
        self.alpha = float(self.settings.get("hybrid_alpha", 0.6))
        self.dense_k = int(self.settings.get("dense_k", 8))
        self.lex_k = int(self.settings.get("lexical_k", 10))

        self.model = SentenceTransformer(meta["model"])

        if meta.get("faiss", True) and HAVE_FAISS and FAISS_FILE.exists():
            self.index = faiss.read_index(str(FAISS_FILE))
            self._is_faiss = True
        else:
            vec = np.load(SK_VECS_FILE)
            self.nn = NearestNeighbors(n_neighbors=self.dense_k, metric="cosine").fit(vec)
            self._vec = vec
            self._is_faiss = False

        self.bow_X = load_npz(BOW_MATRIX_FILE)
        vocab = json.loads(BOW_VOCAB_FILE.read_text(encoding="utf-8"))
        self.vectorizer = CountVectorizer(decode_error="ignore", vocabulary=vocab)
        self.lexicon: Dict[str, List[int]] = json.loads(LEXICON_FILE.read_text(encoding="utf-8"))

    def _dense_search(self, query: str) -> List[Tuple[int, float]]:
        q = self.model.encode(["query: " + query], normalize_embeddings=True).astype(np.float32)
        if self._is_faiss:
            D, I = self.index.search(q, self.dense_k)
            return [(int(i), float(d)) for i, d in zip(I[0], D[0])]
        else:
            D, I = self.nn.kneighbors(q, n_neighbors=self.dense_k)
            scores = [1 - float(d) for d in D[0]]
            idxs = [int(i) for i in I[0]]
            return list(zip(idxs, scores))

    def _lex_search(self, query: str) -> List[Tuple[int, float]]:
        q_vec = self.vectorizer.transform([_norm(query)])
        q_vec = normalize(q_vec, norm="l2", copy=False)
        sims = (self.bow_X @ q_vec.T).toarray().ravel()
        top = np.argsort(-sims)[: self.lex_k]
        return [(int(i), float(sims[i])) for i in top if sims[i] > 0.0]

    def _alias_match(self, query: str) -> List[int]:
        qn = _norm(query)
        hits = set()
        if qn in self.lexicon:
            hits.update(self.lexicon[qn])
        keys = list(self.lexicon.keys())
        res = rf_process.extract(qn, keys, scorer=rf_fuzz.WRatio, limit=5)
        for key, score, _ in res:
            if score >= 85:
                hits.update(self.lexicon.get(key, []))
        return list(hits)

    def search(self, query: str, k: int = 5) -> List[Tuple[int, float]]:
        # >>> KÉTNYELVŰ BŐVÍTÉS BEKAPCSOLVA <<<
        query = expand_query_bilingual(query)

        dense = self._dense_search(query)
        lex = self._lex_search(query)

        combined: Dict[int, float] = {}
        for i, s in dense:
            combined[i] = max(combined.get(i, 0.0), self.alpha * s)
        for i, s in lex:
            combined[i] = max(combined.get(i, 0.0), (1 - self.alpha) * s)

        if not combined:
            for i in self._alias_match(query):
                combined[i] = 0.3

        # --- Tartalomalapú apró boost kapacitás-kérdéseknél ---
        q_low = query.lower()
        wants_capacity = any(x in q_low for x in ["kapacit", "capacity", "rwm"])
        if wants_capacity:
            for i in list(combined.keys()):
                doc = self.docs[i]
                txt = f"{doc.get('title','')} {doc.get('content','')}".lower()
                if ("kapacit" in txt) or ("capacity" in txt) or (" rwm" in txt):
                    combined[i] *= 1.10  # +10%

        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:k]
        return ranked

def build_prompt(question: str, contexts: List[dict]) -> str:
    ctx_blocks = []
    for c in contexts:
        snippet = c["content"]
        if len(snippet) > 1200:
            snippet = snippet[:1200] + "…"
        title = c.get("title") or c.get("id") or "ismeretlen"
        ctx_blocks.append(f"### Forrás: {title}\n{snippet}")
    ctx_text = "\n\n".join(ctx_blocks)
    user = (
        "Kérdés: " + question + "\n\n"
        "Használd a fenti forrásokat. Ha nincs adat, mondd ki. "
        "A végén sorold fel a forrásokat így: [Forrás: <cím>]. "
        "Magyar nyelven válaszolj, tömören és pontosan."
    )
    return f"<s>[SYSTEM]\n{SYSTEM_PROMPT}\n\n[CONTEXT]\n{ctx_text}\n\n[USER]\n{user}\n"

def ollama_generate(prompt: str, temperature: float = 0.2, max_tokens: int = 512) -> str:
    url = f"{OLLAMA_URL}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "num_ctx": 8192
        }
    }
    with requests.post(url, json=payload, timeout=120, stream=True) as r:
        r.raise_for_status()
        text = ""
        for line in r.iter_lines():
            if not line:
                continue
            try:
                obj = json.loads(line.decode("utf-8"))
            except Exception:
                continue
            text += obj.get("response", "")
            if obj.get("done"):
                break
    return text.strip()

def answer(question: str, k: int = 5) -> str:
    retriever = Retriever()
    hits = retriever.search(question, k=k)
    if not hits:
        return "Nincs elég releváns adat. Pontosítsd a kifejezéseket."

    ctxs = [retriever.docs[i] for i, _ in hits]
    prompt = build_prompt(question, ctxs)
    out = ollama_generate(prompt)

    # Forrástagok biztos hozzáfűzése (ha a modell nem tette meg)
    srcs = []
    for c in ctxs:
        tag = c.get("title") or c.get("id")
        if tag:
            srcs.append(f"[Forrás: {tag}]")
    if srcs and srcs[-1] not in out:
        out += "\n\n" + " ".join(srcs)
    return out

# Egyszerű chat (RAG nélkül)
def chat(message: str, temperature: float = 0.5, max_tokens: int = 512) -> str:
    system = (
        "Magyar nyelvű, tömör és tárgyszerű asszisztens vagy. "
        "Ha nem tudsz valamit, mondd ki."
    )
    prompt = f"<s>[SYSTEM]\n{system}\n\n[USER]\n{message}\n"
    return ollama_generate(prompt, temperature=temperature, max_tokens=max_tokens)

if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "Mennyi C4 kell egy armored doorhoz?"
    print(answer(q))
