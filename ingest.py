# ingest.py
# HYBRID+LEXICON indexépítő Rust HU RAG bothoz
# - E:\Rust_Ai\data alól (rekurzívan) minden *.json beolvasása
# - HU→EN alias-lexikon
# - Embedding: intfloat/multilingual-e5-base (env-ből felülírható)
# - FAISS (dense) + BOW (CountVectorizer) index
# - Meta: docs, vocab, lexicon, beállítások

import os
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# a többi import fölé/alá mehet
try:
    from docx import Document
    HAVE_DOCX = True
except Exception:
    HAVE_DOCX = False


# FAISS vagy sklearn fallback
try:
    import faiss  # type: ignore
    HAVE_FAISS = True
except Exception:
    HAVE_FAISS = False
    from sklearn.neighbors import NearestNeighbors  # type: ignore

# BOW
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from scipy.sparse import save_npz

load_dotenv()

# ----- Fix forrásmappa: E:\Rust_Ai\data (env-ből felülírható) -----
DATA_DIR = Path(os.getenv("RAG_DATA_DIR", r"E:\Rust_Ai\data")).resolve()

BASE = Path(__file__).parent.resolve()
INDEX_DIR = (BASE / "index")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

FAISS_FILE = INDEX_DIR / "faiss.index"
SK_VECS_FILE = INDEX_DIR / "sk_vectors.npy"
META_FILE = INDEX_DIR / "meta.json"
TFIDF_MATRIX_FILE = INDEX_DIR / "tfidf_matrix.npz"
TFIDF_VOCAB_FILE = INDEX_DIR / "tfidf_vocab.json"
LEXICON_FILE = INDEX_DIR / "lexicon.json"

# Modell és hibrid súlyok env-ből állíthatók
MODEL_NAME = os.getenv("RAG_EMB_MODEL", "intfloat/multilingual-e5-base")
HYBRID_ALPHA = float(os.getenv("RAG_HYBRID_ALPHA", "0.6"))
DENSE_K = int(os.getenv("RAG_DENSE_K", "8"))
LEXICAL_K = int(os.getenv("RAG_LEXICAL_K", "10"))

# Kézi HU alias magok (bővíthető)
HU_ALIASES_SEED: Dict[str, List[str]] = {
    "Large Solar Panel": ["napelem", "nagy napelem", "solar panel", "nagy solar"],
    "Small Battery": ["kis akku", "kis akkumulátor", "small battery"],
    "Large Battery": ["nagy akku", "nagy akkumulátor", "large battery"],
    "Root Combiner": ["root combiner", "összegző", "forrásösszegző", "kombinátor"],
    "Electrical Branch": ["branch", "ágaztató", "elosztó ág", "elektromos branch"],
    "Power Splitter": ["splitter", "elosztó", "power splitter"],
    "Power Blocker": ["blocker", "blokkoló"],
    "Smart Switch": ["okos kapcsoló", "smart switch"],
    "Switch": ["kapcsoló", "switch"],
    "Timer": ["időzítő", "timer"],
    "Wind Turbine": ["szélkerék", "szélturbina", "wind turbine"],
    "HBHF Sensor": ["mozgásérzékelő", "hbhf", "sensor"],
    
}

def _norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\(.*?\)", "", s)
    s = re.sub(r"[^a-z0-9áéíóöőúüű \-]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

def _norm_key(row: Dict[str, Any], fallback: str = "") -> str:
    def first(*keys):
        for k in keys:
            v = row.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip().lower()
        return ""
    key = first("short_name", "shortName", "display_name", "displayName")
    return key or (fallback.strip().lower() if fallback else "")

def _merge_base_fields(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
    mapping = {
        "display_name": ["display_name", "displayName"],
        "short_name":   ["short_name", "shortName"],
        "item_id":      ["item_id", "itemId", "id"],
        "category":     ["category"],
        "desc_short":   ["desc_short", "description"],
        "image":        ["image", "icon", "iconUrl"],
    }
    for dst_key, candidates in mapping.items():
        if not dst.get(dst_key):
            for ck in candidates:
                val = src.get(ck)
                if val is not None and val != "":
                    dst[dst_key] = val
                    break

def load_all_items() -> Dict[str, Dict[str, Any]]:
    files = sorted(DATA_DIR.rglob("*.json"))  # rekurzív
    if not files:
        raise FileNotFoundError(f"Nincs .json a forrásmappában: {DATA_DIR}")
    merged: Dict[str, Dict[str, Any]] = {}
    print(f"[LOAD] {len(files)} JSON fájl → {DATA_DIR}")
    for p in files:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] {p.name} kihagyva (JSON hiba): {e}")
            continue
        if isinstance(data, dict):
            iterable = [(k, v) for k, v in data.items() if isinstance(v, dict)]
        elif isinstance(data, list):
            iterable = [(str(i), v) for i, v in enumerate(data) if isinstance(v, dict)]
        else:
            continue
        before = len(merged)
        for k, row in iterable:
            key = _norm_key(row, k)
            if not key:
                continue
            dst = merged.setdefault(key, {})
            _merge_base_fields(dst, row)
            if isinstance(row.get("desc_long"), str) and row["desc_long"].strip():
                if len(row["desc_long"]) > len(dst.get("desc_long", "") or ""):
                    dst["desc_long"] = row["desc_long"]
            if isinstance(row.get("mechanics"), dict):
                dst.setdefault("mechanics", {}).update(row["mechanics"])
            dst.setdefault("sources", {})[p.stem] = True
        print(f"[LOAD] {p.name}: +{len(merged)-before} új/egyesített")
    print(f"[LOAD] Összesen {len(merged)} tétel")
    return merged

def _aliases_for_item(display: str, short: str) -> List[str]:
    al = set()
    if display:
        al.add(display); al.add(_norm(display)); al.add(display.lower())
    if short:
        al.add(short); al.add(_norm(short)); al.add(short.lower())
    for tok in (display or "").replace("-", " ").split():
        if len(tok) > 2:
            al.add(tok.lower())
    return sorted(a for a in al if a)

def build_documents_and_lexicon(items: Dict[str, Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, List[int]]]:
    docs: List[Dict[str, Any]] = []
    lexicon: Dict[str, List[int]] = {}

    fields = [
    ("io", "I/O"),
    ("power_output", "Power Output"),
    ("power_consumption", "Power Consumption"),
    ("active_usage", "Active Usage"),
    ("capacity", "Kapacitás"),            # ÚJ
    ("charge_rate", "Töltési ráta"),      # ÚJ
    ("max_output", "Max kimenet"),        # ÚJ
    ("crafting", "Crafting"),
    ("recycles_into", "Recycle"),
    ("stack_size", "Stack Size"),
    ("workbench_required", "Workbench"),
    ("workbench_required_level", "WB Level"),
    ("research_cost", "Research Scrap"),
    ("fuel_consumption", "Fuel Consumption"),
    ]


    print("[BUILD] Dokumentumok + lexikon…")
    for k, it in tqdm(items.items()):
        name = it.get("display_name") or it.get("short_name") or k
        short = it.get("short_name") or ""
        mech = it.get("mechanics", {}) or {}
        rustrician_url = mech.get("rustrician_url")

        parts: List[str] = [f"Név: {name}"]
        if it.get("category"): parts.append(f"Kategória: {it['category']}")
        if it.get("desc_long"): parts.append(f"Leírás: {it['desc_long']}")
        elif it.get("desc_short"): parts.append(f"Leírás: {it['desc_short']}")
        for fld, label in fields:
            val = mech.get(fld)
            if val: parts.append(f"{label}: {val}")

        content = "\n".join(parts)
        doc = {
            "id": k,
            "title": name,
            "content": content,
            "url": rustrician_url,
            "sources": list((it.get("sources") or {}).keys())
        }
        docs.append(doc)
        idx = len(docs) - 1

        als = _aliases_for_item(name, short)
        if name in HU_ALIASES_SEED: als.extend(HU_ALIASES_SEED[name])

        for a in set(als):
            for key in {a, _norm(a)}:
                if not key: continue
                lexicon.setdefault(key, [])
                if idx not in lexicon[key]:
                    lexicon[key].append(idx)

    return docs, lexicon

def build_bow(docs: List[Dict[str, Any]], lexicon: Dict[str, List[int]]):
    corpus: List[str] = []
    for i, d in enumerate(docs):
        aliases_here = [k for k, arr in lexicon.items() if i in arr]
        text = d["content"] + "\naliases: " + " ".join(sorted(set(aliases_here)))
        corpus.append(text)

    print("[BOW] Vectorizer tanítása + mátrix…")
    vec = CountVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1)
    X = vec.fit_transform(corpus)
    X = normalize(X, norm="l2", copy=False)
    save_npz(TFIDF_MATRIX_FILE, X)
    TFIDF_VOCAB_FILE.write_text(json.dumps(vec.vocabulary_, ensure_ascii=False), encoding="utf-8")
    print(f"[BOW] Kész: {X.shape[0]} doksi, {X.shape[1]} szó")

def main():
    print(f"[SRC] Forrásmappa: {DATA_DIR}")
    items = load_all_items()
    docs, lexicon = build_documents_and_lexicon(items)

    print(f"[EMBED] Modell: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    texts = ["passage: " + d["content"] for d in docs]
    emb = model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True).astype(np.float32)
    dim = emb.shape[1]

    if HAVE_FAISS:
        print("[INDEX] FAISS (IP a normalizált vektorokon)…")
        index = faiss.IndexFlatIP(dim)
        index.add(emb)
        faiss.write_index(index, str(FAISS_FILE))
    else:
        print("[INDEX] sklearn NN fallback (cosine)…")
        # Csak a vektorokat mentjük; a NN-t a lekérdező tölti be
        np.save(SK_VECS_FILE, emb)

    build_bow(docs, lexicon)

    META_FILE.write_text(json.dumps({
        "model": MODEL_NAME,
        "faiss": HAVE_FAISS,
        "dim": int(dim),
        "docs": docs,
        "settings": {
        "hybrid_alpha": HYBRID_ALPHA,
        "dense_k": DENSE_K,
        "lexical_k": LEXICAL_K
        }
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    LEXICON_FILE.write_text(json.dumps(lexicon, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[KÉSZ] Index + lexikon kész:", INDEX_DIR)

if __name__ == "__main__":
    main()
