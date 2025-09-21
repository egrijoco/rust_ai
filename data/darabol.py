import json
import sys
from pathlib import Path

def json_size_bytes(obj) -> int:
    # tömör JSON méret byte-ban (UTF-8)
    return len(json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))

def balanced_chunks_for_list(lst, parts):
    # Elemenkénti súly (méret), majd greedy bepakolás a legkisebb összegű vödörbe
    weights = [json_size_bytes(e) for e in lst]
    idxs = list(range(len(lst)))
    # nagyoktól kicsik felé
    idxs.sort(key=lambda i: weights[i], reverse=True)
    buckets = [[] for _ in range(parts)]
    sums = [0] * parts
    for i in idxs:
        j = min(range(parts), key=lambda x: sums[x])
        buckets[j].append(lst[i])
        sums[j] += weights[i]
    # visszarendezés a stabilitás kedvéért (nem kötelező)
    for b in buckets:
        b.sort(key=lambda e: json.dumps(e, ensure_ascii=False, separators=(",", ":")))
    return buckets, sums

def balanced_chunks_for_dict(dct, parts):
    items = list(dct.items())
    # Kulcsok stabil sorrendje (opcionális): ABC szerint
    items.sort(key=lambda kv: str(kv[0]))
    # Méret súly a {k:v} párokra
    weights = [json_size_bytes({k: v}) for k, v in items]
    idxs = list(range(len(items)))
    idxs.sort(key=lambda i: weights[i], reverse=True)
    buckets = [dict() for _ in range(parts)]
    sums = [0] * parts
    for i in idxs:
        k, v = items[i]
        j = min(range(parts), key=lambda x: sums[x])
        buckets[j][k] = v
        sums[j] += weights[i]
    return buckets, sums

def main():
    if len(sys.argv) < 2:
        print("Használat: python split_json_balanced.py <input.json> [parts=4]")
        sys.exit(1)
    in_path = Path(sys.argv[1])
    parts = int(sys.argv[2]) if len(sys.argv) >= 3 else 4
    if not in_path.exists():
        print(f"Nem található: {in_path}")
        sys.exit(1)
    if parts < 2:
        print("A 'parts' legalább 2 legyen.")
        sys.exit(1)

    data = json.loads(in_path.read_text(encoding="utf-8"))
    stem, suffix = in_path.stem, in_path.suffix
    out_dir = in_path.parent

    if isinstance(data, list):
        buckets, sums = balanced_chunks_for_list(data, parts)
        for i, b in enumerate(buckets, start=1):
            out = out_dir / f"{stem}_part{i}{suffix}"
            out.write_text(json.dumps(b, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
            print(f"Írva: {out}  elemek: {len(b):5d}   becsült méret: {sums[i-1]:,} B")
    elif isinstance(data, dict):
        buckets, sums = balanced_chunks_for_dict(data, parts)
        for i, b in enumerate(buckets, start=1):
            out = out_dir / f"{stem}_part{i}{suffix}"
            out.write_text(json.dumps(b, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
            print(f"Írva: {out}  kulcsok: {len(b):5d}   becsült méret: {sums[i-1]:,} B")
    else:
        print("A felső szintű JSON nem lista és nem objektum — ezt nem támogatja.")
        sys.exit(1)

if __name__ == "__main__":
    main()
