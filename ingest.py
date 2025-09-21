# ingest.py – E:\Rust_Ai\data beolvasás, egyesítés, index építés
import os
import re
import json
import glob
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np

# --- BoW / Embedding / FAISS ---
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix, save_npz
from dotenv import load_dotenv
from pathlib import Path

try:
    import faiss  # type: ignore
    HAVE_FAISS = True
except Exception:
    HAVE_FAISS = False
    from sklearn.neighbors import NearestNeighbors  # type: ignore

from sentence_transformers import SentenceTransformer

# DOCX (opcionális)
try:
    from docx import Document  # type: ignore
    HAVE_DOCX = True
except Exception:
    HAVE_DOCX = False


BASE = Path(__file__).parent.resolve()
load_dotenv(dotenv_path=BASE / ".env")
DATA_DIR = Path(os.getenv("RAG_DATA_DIR", str(BASE / "data")))
INDEX_DIR = BASE / "index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# ENV beállítások
EMB_MODEL = os.getenv("RAG_EMB_MODEL", "intfloat/multilingual-e5-base")
HYBRID_ALPHA = float(os.getenv("RAG_HYBRID_ALPHA", "0.6"))
DENSE_K = int(os.getenv("RAG_DENSE_K", "8"))
LEX_K = int(os.getenv("RAG_LEXICAL_K", "10"))

# Kimeneti fájlok
META_FILE = INDEX_DIR / "meta.json"
FAISS_FILE = INDEX_DIR / "faiss.index"
SK_VECS_FILE = INDEX_DIR / "sk_vectors.npy"
BOW_MATRIX_FILE = INDEX_DIR / "tfidf_matrix.npz"
BOW_VOCAB_FILE = INDEX_DIR / "tfidf_vocab.json"
LEXICON_FILE = INDEX_DIR / "lexicon.json"
CATALOG_FILE = INDEX_DIR / "catalog.json"

# ---------- segédek ----------
def _norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\(.*?\)", "", s)
    s = re.sub(r"[^a-z0-9áéíóöőúüű \-]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        try:
            return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            return None

def _as_lines(d: Dict[str, Any], keys_top: List[str]) -> str:
    """Kiválasztott kulcsok rendezett megjelenítése; fallback: rövid pretty JSON."""
    lines = []
    for k in keys_top:
        v = d.get(k)
        if v is not None and v != "":
            if isinstance(v, (dict, list)):
                try:
                    v = json.dumps(v, ensure_ascii=False)
                except Exception:
                    v = str(v)
            lines.append(f"{k}: {v}")
    if not lines:
        # rövid fallback
        try:
            js = json.dumps(d, ensure_ascii=False)
            return js[:3000]
        except Exception:
            return str(d)[:3000]
    return "\n".join(lines)

# ---------- Forrás-parszolók ----------
def parse_rusthelp_items(data: Any) -> List[Dict[str, Any]]:
    """RustHelp admin-item-list-public.json / item-list-public.json"""
    out = []
    rows = []
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        rows = list(data.values())
    for r in rows:
        dn = r.get("displayName") or r.get("display_name") or r.get("name")
        sn = r.get("shortName") or r.get("short_name")
        iid = r.get("itemId") or r.get("id")
        cat = r.get("category") or r.get("Category") or r.get("type")
        desc = r.get("description") or r.get("desc") or ""
        if dn or sn:
            out.append({
                "display_name": dn or sn,
                "short_name": sn,
                "item_id": iid,
                "category": cat,
                "description": desc,
                "sources": {"rusthelp": True},
            })
    return out

def parse_rustplusplus_items(data: Any) -> List[Dict[str, Any]]:
    """
    RustPlusPlus staticFiles/items.json – tipikusan id->objektum mapping,
    mezők: name, shortname, description, (néha) category.
    """
    out = []
    if isinstance(data, dict):
        values = list(data.values())
    elif isinstance(data, list):
        values = data
    else:
        values = []
    for r in values:
        dn = r.get("name") or r.get("display_name")
        sn = r.get("shortname") or r.get("short_name")
        iid = r.get("id") or r.get("itemId")
        cat = r.get("category")
        desc = r.get("description") or ""
        if dn or sn:
            out.append({
                "display_name": dn or sn,
                "short_name": sn,
                "item_id": iid,
                "category": cat,
                "description": desc,
                "sources": {"rustplusplus": True},
            })
    return out

def parse_rustlabs_data(files: Dict[str, Path]) -> Dict[str, Dict[str, Any]]:
    """
    Visszaad: short_name -> mechanics kiegészítések (stack, recycle, smelt, despawn, durability, upkeep, decay…)
    Elfogad bármelyik meglévő rustlabs*Data*.json fájlt (ha nincs, kihagyja).
    """
    mech: Dict[str, Dict[str, Any]] = {}
    def up(sn: str, key: str, val: Any):
        if not sn:
            return
        node = mech.setdefault(sn, {})
        if isinstance(val, (dict, list)):
            node[key] = val
        elif val is not None:
            node[key] = val

    # Stack
    p = files.get("stack")
    if p:
        data = _read_json(p) or {}
        # várható: { shortname: stacksize, ... }
        for sn, stack in data.items():
            up(sn, "stack_size", stack)

    # Recycle
    p = files.get("recycle")
    if p:
        data = _read_json(p) or {}
        # formátum: { shortname: [{item,amount,prob}, ...] }
        for sn, arr in data.items():
            up(sn, "recycles_into", arr)

    # Smelting
    p = files.get("smelting")
    if p:
        data = _read_json(p) or {}
        for sn, arr in data.items():
            up(sn, "smelting", arr)

    # Despawn
    p = files.get("despawn")
    if p:
        data = _read_json(p) or {}
        for sn, secs in data.items():
            up(sn, "despawn_seconds", secs)

    # Durability
    p = files.get("durability")
    if p:
        data = _read_json(p) or {}
        for sn, info in data.items():
            up(sn, "durability", info)

    # Decay & Upkeep
    p = files.get("decay")
    if p:
        data = _read_json(p) or {}
        up("__global__", "decay", data)
    p = files.get("upkeep")
    if p:
        data = _read_json(p) or {}
        up("__global__", "upkeep", data)

    # Craft
    p = files.get("craft")
    if p:
        data = _read_json(p) or {}
        for sn, info in data.items():
            up(sn, "crafting", info)

    return mech

# ---------- Adatbetöltés ----------
def collect_items_and_docs(data_dir: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Visszaad:
      items: kanonikus item rekordok listája (display_name, short_name, item_id, category, description, mechanics)
      docs:  kereshető dokumentumok listája (id, title, content, category)
    """
    items: Dict[str, Dict[str, Any]] = {}
    docs: List[Dict[str, Any]] = []

    # 1) RustHelp listák
    for fp in glob.glob(str(data_dir / "*item*list*.json")) + glob.glob(str(data_dir / "*admin*item*list*.json")):
        data = _read_json(Path(fp))
        for it in parse_rusthelp_items(data):
            key = _norm(it.get("short_name") or it.get("display_name") or "")
            if not key:
                continue
            base = items.setdefault(key, {"mechanics": {}, "sources": {}})
            for k in ("display_name","short_name","item_id","category","description"):
                if it.get(k) and not base.get(k):
                    base[k] = it[k]
            base["sources"]["rusthelp"] = True

    # 2) RustPlusPlus staticFiles (ha letöltötted ide: data/rustplusplus/*.json)
    rpp_dir = data_dir / "rustplusplus"
    if rpp_dir.exists():
        # items.json
        for fp in glob.glob(str(rpp_dir / "items*.json")):
            data = _read_json(Path(fp))
            for it in parse_rustplusplus_items(data):
                key = _norm(it.get("short_name") or it.get("display_name") or "")
                if not key:
                    continue
                base = items.setdefault(key, {"mechanics": {}, "sources": {}})
                for k in ("display_name","short_name","item_id","category","description"):
                    if it.get(k) and not base.get(k):
                        base[k] = it[k]
                base["sources"]["rustplusplus"] = True

    # 3) RustLabs *Data.json – mechanics
    rl_files = {
        "stack": next((Path(p) for p in glob.glob(str(data_dir / "rustlabsStackData*.json"))), None),
        "recycle": next((Path(p) for p in glob.glob(str(data_dir / "rustlabsRecycleData*.json"))), None),
        "smelting": next((Path(p) for p in glob.glob(str(data_dir / "rustlabsSmeltingData*.json"))), None),
        "despawn": next((Path(p) for p in glob.glob(str(data_dir / "rustlabsDespawnData*.json"))), None),
        "durability": next((Path(p) for p in glob.glob(str(data_dir / "rustlabsDurabilityData*.json"))), None),
        "decay": next((Path(p) for p in glob.glob(str(data_dir / "rustlabsDecayData*.json"))), None),
        "upkeep": next((Path(p) for p in glob.glob(str(data_dir / "rustlabsUpkeepData*.json"))), None),
        "craft": next((Path(p) for p in glob.glob(str(data_dir / "rustlabsCraftData*.json"))), None),
    }
    rl_files = {k:v for k,v in rl_files.items() if v}
    if rl_files:
        mech_map = parse_rustlabs_data(rl_files)  # short_name -> mechanics
        for sn, mech in mech_map.items():
            if sn == "__global__":
                # globális info (decay/upkeep) – berakjuk külön docnak is
                docs.append({
                    "id": f"rustlabs_global_{list(mech.keys())}",
                    "title": "RustLabs – Global Base Mechanics (decay/upkeep)",
                    "content": _as_lines(mech, list(mech.keys())),
                    "category": "mechanics",
                })
                continue
            key = _norm(sn)
            base = items.setdefault(key, {"mechanics": {}, "sources": {}})
            base["mechanics"].update(mech)
            base["sources"]["rustlabs"] = True

    # 4) DOCX/TXT/MD – sima dokumentumok
    for fp in glob.glob(str(data_dir / "**/*.docx"), recursive=True) if HAVE_DOCX else []:
        try:
            doc = Document(fp)
            text = "\n".join(p.text for p in doc.paragraphs)
            title = Path(fp).stem
            if text.strip():
                docs.append({"id": fp, "title": title, "content": text[:120000], "category": "doc"})
        except Exception:
            pass

    for pattern in ("**/*.txt", "**/*.md"):
        for fp in glob.glob(str(data_dir / pattern), recursive=True):
            try:
                text = Path(fp).read_text(encoding="utf-8", errors="ignore")
                title = Path(fp).stem
                if text.strip():
                    docs.append({"id": fp, "title": title, "content": text[:120000], "category": "doc"})
            except Exception:
                pass

    # 5) Item rekordokból generált „doc” (kereshető kivonat)
    for key, it in items.items():
        title = it.get("display_name") or it.get("short_name") or key
        content = _as_lines({
            "short_name": it.get("short_name"),
            "item_id": it.get("item_id"),
            "category": it.get("category"),
            "description": it.get("description"),
            "mechanics": it.get("mechanics"),
        }, ["short_name","item_id","category","description","mechanics"])
        docs.append({
            "id": f"item::{key}",
            "title": title,
            "content": content,
            "category": it.get("category") or "item"
        })

    # 6) Maradék JSON-ok mint „nyers” doc (ha még nincs dokumentumban)
    for fp in glob.glob(str(data_dir / "**/*.json"), recursive=True):
        p = Path(fp)
        # kihagyjuk a már feldolgozott tipikus fájlokat
        name = p.name.lower()
        if any(x in name for x in ["item-list", "admin-item-list", "items.json",
                                   "rustlabs", "blueprint", "stack", "recycle",
                                   "smelting", "despawn", "durability", "decay", "upkeep", "craft"]):
            continue
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
            if txt.strip():
                docs.append({"id": fp, "title": p.stem, "content": txt[:120000], "category": "raw"})
        except Exception:
            pass

    # Katalógus
    catalog = []
    for key, it in items.items():
        catalog.append({
            "display_name": it.get("display_name") or it.get("short_name") or key,
            "short_name": it.get("short_name"),
            "item_id": it.get("item_id"),
            "category": it.get("category") or "",
        })

    return catalog, docs

# ---------- Index építés ----------
def build_bow(docs: List[Dict[str, Any]]) -> Tuple[CountVectorizer, csr_matrix, Dict[str,int]]:
    texts = [ _norm(d.get("title","") + " " + d.get("content","")) for d in docs ]
    vectorizer = CountVectorizer(ngram_range=(1,2), min_df=1, max_features=80000)
    X = vectorizer.fit_transform(texts)
    X = normalize(X, norm="l2", copy=False)
    vocab = vectorizer.vocabulary_
    return vectorizer, X, vocab

def build_dense(docs: List[Dict[str, Any]]) -> np.ndarray:
    model = SentenceTransformer(EMB_MODEL)
    texts = [ "passage: " + (d.get("title","") + "\n" + d.get("content",""))[:4000] for d in docs ]
    emb = model.encode(texts, show_progress_bar=True, normalize_embeddings=True).astype(np.float32)
    return emb

def build_faiss(emb: np.ndarray):
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    faiss.write_index(index, str(FAISS_FILE))

def save_meta(docs: List[Dict[str, Any]]):
    meta = {
        "model": EMB_MODEL,
        "faiss": HAVE_FAISS,
        "docs": docs,
        "settings": {
            "hybrid_alpha": HYBRID_ALPHA,
            "dense_k": DENSE_K,
            "lexical_k": LEX_K
        }
    }
    META_FILE.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

def build_lexicon(docs: List[Dict[str, Any]]) -> Dict[str, List[int]]:
    lex: Dict[str, List[int]] = {}
    def add(k: str, idx: int):
        k = _norm(k)
        if not k:
            return
        lex.setdefault(k, []).append(idx)

    for i, d in enumerate(docs):
        t = d.get("title") or ""
        add(t, i)
        # item nevek szétbontva
        for token in re.split(r"[/\-–:,]", t):
            add(token, i)
    return lex

def main():
    print(f"[INGEST] DATA_DIR: {DATA_DIR}")
    catalog, docs = collect_items_and_docs(DATA_DIR)
    print(f"[INGEST] items a katalógusban: {len(catalog)} | dokumentumok: {len(docs)}")

    # Katalógus mentése
    CATALOG_FILE.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INGEST] catalog.json mentve ({CATALOG_FILE})")

    # BoW
    vectorizer, X, vocab = build_bow(docs)
    save_npz(str(BOW_MATRIX_FILE), X)
    safe_vocab = {k: int(v) for k, v in vocab.items()}
    BOW_VOCAB_FILE.write_text(json.dumps(safe_vocab, ensure_ascii=False), encoding="utf-8")

    print(f"[INGEST] BoW kész: {X.shape}")

    # Dense
    emb = build_dense(docs)
    np.save(str(SK_VECS_FILE), emb)
    print(f"[INGEST] Dense vektorok: {emb.shape}")

    if HAVE_FAISS:
        build_faiss(emb)
        print(f"[INGEST] FAISS index mentve: {FAISS_FILE}")
    else:
        print("[INGEST] FAISS nincs, sklearn fallback marad.")

    # Meta & Lexikon
    save_meta(docs)
    lex = build_lexicon(docs)
    LEXICON_FILE.write_text(json.dumps(lex, ensure_ascii=False), encoding="utf-8")
    print(f"[INGEST] meta.json + lexicon.json kész.")

if __name__ == "__main__":
    main()
