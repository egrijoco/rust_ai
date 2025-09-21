import os
import json
import time
import re
import logging
from urllib.parse import urljoin
from collections import defaultdict
from dotenv import load_dotenv
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import glob

# Környezeti változók betöltése
load_dotenv()

# Naplózás beállítása
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rust_data_collection.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

UA = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/91.0.4472.124 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1"
}

# Steam API kulcs (ha később kellene)
STEAM_API_KEY = os.getenv("STEAM_API_KEY")

# ----- Stabil hálózati kliens (retry + backoff) -----
SESSION = requests.Session()
retry = Retry(
    total=5,
    connect=5,
    read=5,
    backoff_factor=0.8,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=frozenset(["GET", "POST"])
)
SESSION.mount("https://", HTTPAdapter(max_retries=retry))
SESSION.mount("http://", HTTPAdapter(max_retries=retry))


def http_get(url, *, params=None, headers=None, timeout=30):
    h = dict(UA)
    if headers:
        h.update(headers)
    try:
        r = SESSION.get(url, params=params, headers=h, timeout=timeout)
        r.raise_for_status()
        return r
    except requests.exceptions.RequestException as e:
        logger.error(f"Hiba a következő URL lekérésekor: {url}: {e}")
        return None


# ========== 0) Helyi JSON fájlok összegyűjtése ==========
def load_local_json_files(directory="data"):
    """Összes helyi JSON fájl betöltése és egyesítése"""
    logger.info(f"[HELYI] Helyi JSON fájlok betöltése a '{directory}' könyvtárból…")

    items = {}
    json_files = glob.glob(os.path.join(directory, "*.json"))

    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

                if isinstance(data, dict):
                    for key, item_data in data.items():
                        key_norm = str(key).lower().strip()
                        if key_norm not in items:
                            items[key_norm] = dict(item_data)
                            items[key_norm].setdefault("sources", {})["local"] = [os.path.basename(file_path)]
                        else:
                            items[key_norm].update(item_data)
                            items[key_norm].setdefault("sources", {}).setdefault("local", []).append(
                                os.path.basename(file_path)
                            )

                elif isinstance(data, list):
                    for item_data in data:
                        key = item_data.get("item_id") or item_data.get("display_name") or str(
                            hash(json.dumps(item_data, sort_keys=True))
                        )
                        key_norm = str(key).lower().strip()

                        if key_norm not in items:
                            items[key_norm] = dict(item_data)
                            items[key_norm].setdefault("sources", {})["local"] = [os.path.basename(file_path)]
                        else:
                            items[key_norm].update(item_data)
                            items[key_norm].setdefault("sources", {}).setdefault("local", []).append(
                                os.path.basename(file_path)
                            )

                logger.info(f"[HELYI] {os.path.basename(file_path)}: "
                            f"{len(data) if isinstance(data, (dict, list)) else 1} elem")

        except Exception as e:
            logger.error(f"[HELYI] Hiba a {file_path} fájl feldolgozásakor: {e}")

    logger.info(f"[HELYI] Összesen {len(items)} egyedi elem a helyi fájlokból")
    return items


# ========== 1) Alap lista – RustHelp Export ==========
RUSTHELP_JSON_URL = "https://rusthelp.com/downloads/admin-item-list-public.json"


def fetch_rusthelp_items():
    logger.info("[RUSTHELP] letöltés…")
    r = http_get(RUSTHELP_JSON_URL)
    if not r:
        logger.error("[RUSTHELP] Nem sikerült letölteni az adatokat")
        return {}

    try:
        data = r.json()
    except Exception as e:
        logger.error(f"[RUSTHELP] JSON parse hiba: {e}")
        return {}

    items = {}
    for row in data:
        key = (row.get("shortName") or row.get("displayName", "")).strip().lower()
        if not key:
            continue

        items[key] = {
            "display_name": row.get("displayName"),
            "short_name": row.get("shortName"),
            "item_id": row.get("itemId") or row.get("id"),
            "category": row.get("category"),
            "desc_short": row.get("description"),
            "image": row.get("image"),
            "sources": {"rusthelp": True},
            "mechanics": {}
        }
    logger.info(f"[RUSTHELP] {len(items)} tétel beolvasva.")
    return items


# ========== 2) Rustrician – wiki bejárás + mezők ==========
RUSTRICIAN_BASE = "https://www.rustrician.io"
RUSTRICIAN_INDEX = f"{RUSTRICIAN_BASE}/wiki/"

FIELD_ALIASES = {
    "Item ID": "item_id",
    "Description": "desc_rustrician",
    "Crafting Recipe": "crafting",
    "Recycles Into": "recycles_into",
    "Stack Size": "stack_size",
    "Workbench Required": "workbench_required",
    "Workbench Required Level": "workbench_required_level",
    "Research Table Cost": "research_cost",
    "Inputs/Outputs": "io",
    "Inputs / Outputs": "io",
    "Power Output": "power_output",
    "Power Consumption": "power_consumption",
    "Active Usage": "active_usage",
    "Fuel Consumption": "fuel_consumption",
}


def fetch_rustrician_pages():
    logger.info("[RUSTRICIAN] Wiki index letöltése…")
    r = http_get(RUSTRICIAN_INDEX)
    if not r:
        logger.error("[RUSTRICIAN] Nem sikerült letölteni az index oldalt")
        return []

    soup = BeautifulSoup(r.text, "html.parser")
    pages = []

    for a in soup.select("a[href]"):
        href = a["href"]
        if href.startswith("/wiki/") and href.endswith(".html"):
            full_url = urljoin(RUSTRICIAN_BASE, href)
            pages.append(full_url)

    pages = sorted(set(pages))
    logger.info(f"[RUSTRICIAN] {len(pages)} oldal talált.")
    return pages


def parse_rustrician_page(url):
    logger.debug(f"[RUSTRICIAN] Oldal feldolgozása: {url}")
    r = http_get(url)
    if not r:
        return None

    soup = BeautifulSoup(r.text, "html.parser")

    # Oldal címének kinyerése
    title_el = soup.find("h1") or soup.find("h2") or soup.find("title")
    title = title_el.get_text(strip=True) if title_el else ""

    # Tartalom kinyerése (JAVÍTVA)
    content = (
        soup.find("div", class_="wiki-content")
        or soup.find("main")
        or soup.find("article")
        or soup
    )

    # Táblázatos adatok kinyerése
    block = {}
    tables = content.find_all("table")

    for table in tables:
        rows = table.find_all("tr")
        for row in rows:
            cells = row.find_all(["th", "td"])
            if len(cells) == 2:
                key = cells[0].get_text(strip=True)
                value = cells[1].get_text(strip=True)

                if key in FIELD_ALIASES and value:
                    field_name = FIELD_ALIASES[key]
                    block[field_name] = value

    # Szöveges tartalom kinyerése
    text_content = content.get_text("\n", strip=True)
    lines = text_content.splitlines()

    for line in lines:
        if ":" in line:
            parts = line.split(":", 1)
            if len(parts) == 2:
                k, v = parts[0].strip(), parts[1].strip()
                if k in FIELD_ALIASES and v:
                    field_name = FIELD_ALIASES[k]
                    if field_name not in block:
                        block[field_name] = v

    if block or title:
        result = {"name_guess": title, "rustrician_url": url}
        result.update(block)
        return result

    return None


def _norm(s: str) -> str:
    if not s:
        return ""
    s = s.lower().strip()
    s = re.sub(r"\(.*?\)", "", s)   # zárójeles részek ki
    s = re.sub(r"[^a-z0-9áéíóöőúüű\s\-]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def enrich_with_rustrician(items):
    logger.info("[RUSTRICIAN] dúsítás indul…")
    pages = fetch_rustrician_pages()
    total = len(pages)
    if total == 0:
        logger.warning("[RUSTRICIAN] nincs oldal.")
        return items

    # Index a normalizált nevekről
    name_index = defaultdict(list)
    for k, it in items.items():
        dn = it.get("display_name") or ""
        sn = it.get("short_name") or ""
        if dn:
            name_index[_norm(dn)].append(k)
        if sn and sn != dn:
            name_index[_norm(sn)].append(k)

    successful = 0
    for i, url in enumerate(pages, 1):
        pr = parse_rustrician_page(url)
        if pr:
            name = _norm(pr.get("name_guess") or "")

            matched_keys = set()
            if name in name_index:
                matched_keys.update(name_index[name])

            if not matched_keys:
                for norm_name, keys in name_index.items():
                    if norm_name and (norm_name in name or name in norm_name):
                        matched_keys.update(keys)

            for key in matched_keys:
                items[key].setdefault("mechanics", {})

                for field, value in pr.items():
                    if field in ["name_guess", "rustrician_url"]:
                        continue
                    if not items[key]["mechanics"].get(field):
                        items[key]["mechanics"][field] = value

                items[key].setdefault("sources", {})["rustrician"] = True
                successful += 1

        if i % 5 == 0 or i == total:
            pct = i / total * 100
            logger.info(f"[RUSTRICIAN] {i}/{total} ({pct:.1f}%), {successful} sikeres egyezés")

        time.sleep(0.2)  # szerverkímélés

    logger.info(f"[RUSTRICIAN] kész: {total} oldal feldolgozva, {successful} sikeres egyezés.")
    return items


# ========== 3) Facepunch Wiki támogatás ==========
FACEPUNCH_WIKI_BASE = "https://wiki.facepunch.com"
FACEPUNCH_WIKI_API = f"{FACEPUNCH_WIKI_BASE}/api.php"


def fetch_facepunch_wiki_data():
    """Facepunch Wiki adatainak letöltése API-n keresztül"""
    logger.info("[FACEPUNCH WIKI] Adatok letöltése…")

    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": "Category:Rust",
        "cmlimit": "max",
        "format": "json"
    }

    r = http_get(FACEPUNCH_WIKI_API, params=params)
    if not r:
        logger.error("[FACEPUNCH WIKI] Nem sikerült letölteni az adatokat")
        return {}

    data = r.json()
    pages = data.get("query", {}).get("categorymembers", [])

    items = {}
    for page in pages:
        page_id = page.get("pageid")
        page_title = page.get("title")

        content_params = {
            "action": "query",
            "prop": "revisions|images",
            "rvprop": "content",
            "rvslots": "main",
            "pageids": page_id,
            "format": "json"
        }

        content_r = http_get(FACEPUNCH_WIKI_API, params=content_params)
        if content_r:
            content_data = content_r.json()
            pages_data = content_data.get("query", {}).get("pages", {})
            if str(page_id) in pages_data:
                page_data = pages_data[str(page_id)]
                revisions = page_data.get("revisions", [])
                if revisions:
                    content = revisions[0].get("slots", {}).get("main", {}).get("content", "")
                    items[page_title] = parse_facepunch_content(content, page_title)

    logger.info(f"[FACEPUNCH WIKI] {len(items)} tétel beolvasva.")
    return items


def parse_facepunch_content(content, title):
    """Facepunch Wiki oldal tartalmának feldolgozása"""
    result = {
        "display_name": title,
        "sources": {"facepunch_wiki": True},
        "mechanics": {}
    }

    patterns = {
        "description": r"\|description=(.*?)(?=\n\||\Z)",
        "crafting": r"\|crafting=(.*?)(?=\n\||\Z)",
        "stack_size": r"\|stack[ _]size=(.*?)(?=\n\||\Z)",
        "workbench": r"\|workbench=(.*?)(?=\n\||\Z)",
        "research": r"\|research[ _]cost=(.*?)(?=\n\||\Z)"
    }

    for field, pattern in patterns.items():
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            if field == "description":
                result["desc_facepunch"] = value
            elif field == "research":
                result["mechanics"]["research_cost"] = value
            else:
                result["mechanics"][field] = value

    return result


def enrich_with_facepunch_wiki(items):
    """Facepunch Wiki adatok hozzáadása a meglévő itemekhez"""
    logger.info("[FACEPUNCH WIKI] Adatok hozzáadása…")
    facepunch_data = fetch_facepunch_wiki_data()

    for key, item in items.items():
        display_name = item.get("display_name", "") or ""
        if not display_name:
            continue
        item.setdefault("mechanics", {})

        for facepunch_title, facepunch_item in facepunch_data.items():
            t = facepunch_title.lower()
            dn = display_name.lower()
            if dn in t or t in dn:
                if "desc_facepunch" in facepunch_item and not item.get("desc_long"):
                    item["desc_long"] = facepunch_item["desc_facepunch"]

                for mech_key, mech_value in facepunch_item.get("mechanics", {}).items():
                    if not item["mechanics"].get(mech_key):
                        item["mechanics"][mech_key] = mech_value

                item.setdefault("sources", {})["facepunch_wiki"] = True
                break

    return items


# ========== 4) RustClash Wiki támogatás ==========
RUSTCLASH_BASE = "https://wiki.rustclash.com"
RUSTCLASH_API = f"{RUSTCLASH_BASE}/api.php"


def fetch_rustclash_data():
    """RustClash Wiki adatainak letöltése API-n keresztül"""
    logger.info("[RUSTCLASH] Adatok letöltése…")

    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": "Category:Items",
        "cmlimit": "max",
        "format": "json"
    }

    r = http_get(RUSTCLASH_API, params=params)
    if not r:
        logger.error("[RUSTCLASH] Nem sikerült letölteni az adatokat")
        return {}

    data = r.json()
    pages = data.get("query", {}).get("categorymembers", [])

    items = {}
    for page in pages:
        page_id = page.get("pageid")
        page_title = page.get("title")

        content_params = {
            "action": "query",
            "prop": "revisions|images",
            "rvprop": "content",
            "rvslots": "main",
            "pageids": page_id,
            "format": "json"
        }

        content_r = http_get(RUSTCLASH_API, params=content_params)
        if content_r:
            content_data = content_r.json()
            pages_data = content_data.get("query", {}).get("pages", {})
            if str(page_id) in pages_data:
                page_data = pages_data[str(page_id)]
                revisions = page_data.get("revisions", [])
                if revisions:
                    content = revisions[0].get("slots", {}).get("main", {}).get("content", "")
                    items[page_title] = parse_rustclash_content(content, page_title)

    logger.info(f"[RUSTCLASH] {len(items)} tétel beolvasva.")
    return items


def parse_rustclash_content(content: str, title: str):
    """RustClash Wiki oldal tartalmának feldolgozása"""
    result = {
        "display_name": title,
        "sources": {"rustclash": True},
        "mechanics": {}
    }

    patterns = {
        "description": r"\|description=(.*?)(?=\n\||\Z)",
        "crafting":   r"\|crafting=\((.*?)\)",
        "stack_size": r"\|stack[ _]size=(.*?)(?=\n\||\Z)",
        "workbench":  r"\|workbench=(.*?)(?=\n\||\Z)",
        "research":   r"\|research[ _]cost=(.*?)(?=\n\||\Z)"
    }

    for field, pattern in patterns.items():
        m = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        if not m:
            continue
        value = m.group(1).strip()
        if not value:
            continue

        if field == "description":
            result["desc_rustclash"] = value
        elif field == "research":
            result["mechanics"]["research_cost"] = value
        else:
            result["mechanics"][field] = value

    return result


def enrich_with_rustclash(items: dict) -> dict:
    """RustClash Wiki adatok hozzáadása a meglévő itemekhez (név szerinti puha egyezéssel)"""
    logger.info("[RUSTCLASH] Adatok hozzáadása…")
    rc_data = fetch_rustclash_data()
    if not rc_data:
        return items

    for key, item in items.items():
        dn = (item.get("display_name") or "").lower()
        if not dn:
            continue
        item.setdefault("mechanics", {})

        for title, rc_item in rc_data.items():
            t = title.lower()
            if dn in t or t in dn:
                if "desc_rustclash" in rc_item and not item.get("desc_long"):
                    item["desc_long"] = rc_item["desc_rustclash"]

                for mk, mv in rc_item.get("mechanics", {}).items():
                    if not item["mechanics"].get(mk):
                        item["mechanics"][mk] = mv

                item.setdefault("sources", {})["rustclash"] = True
                break

    return items


# ========== Segédek: összefésülés és mentés ==========
def merge_items(*dicts):
    """Alap egyesítés kulcs szerint. Ütközésnél a már létező mezőket nem írjuk felül kivéve, ha üresek."""
    merged = {}
    for d in dicts:
        for k, v in d.items():
            if k not in merged:
                merged[k] = dict(v)
                merged[k].setdefault("mechanics", {})
                merged[k].setdefault("sources", {})
            else:
                # egyszerű mezők
                for fld, val in v.items():
                    if fld in ("mechanics", "sources"):
                        continue
                    if not merged[k].get(fld) and val:
                        merged[k][fld] = val

                # mechanics
                merged[k].setdefault("mechanics", {})
                for mf, mv in v.get("mechanics", {}).items():
                    if not merged[k]["mechanics"].get(mf) and mv:
                        merged[k]["mechanics"][mf] = mv

                # sources
                merged[k].setdefault("sources", {})
                for sf, sv in v.get("sources", {}).items():
                    if isinstance(sv, list):
                        merged[k]["sources"].setdefault(sf, [])
                        for entry in sv:
                            if entry not in merged[k]["sources"][sf]:
                                merged[k]["sources"][sf].append(entry)
                    else:
                        merged[k]["sources"][sf] = sv
    return merged


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"[MENTÉS] {path} kész.")


# ========== Fő folyamat ==========
def main():
    # 0) Helyi
    local_items = load_local_json_files(directory="data")

    # 1) RustHelp
    rusthelp_items = fetch_rusthelp_items()

    # Alap merge
    items = merge_items(rusthelp_items, local_items)

    # 2) Rustrician dúsítás
    items = enrich_with_rustrician(items)

    # 3) Facepunch wiki
    items = enrich_with_facepunch_wiki(items)

    # 4) RustClash wiki
    items = enrich_with_rustclash(items)

    # Mentés
    save_json(items, os.path.join("items_merged.json"))


if __name__ == "__main__":
    main()
