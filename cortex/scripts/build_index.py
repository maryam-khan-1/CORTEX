"""Build offline Chroma index from post-Jan-2025 KEV + recent CVE/OSV records.

Run once (network allowed at build time). Demo runtime is offline.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PERSIST_DIR = ROOT / "data" / "chroma"
META_PATH = PERSIST_DIR / "structured_lookup.json"
COLLECTION = "cortex_cve"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CUTOFF = date(2025, 1, 1)
MAX_RECENT = 250

KEV_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
OSV_QUERY = "https://api.osv.dev/v1/query"


def fetch_kev(client: httpx.Client) -> list[dict]:
    r = client.get(KEV_URL, timeout=60.0)
    r.raise_for_status()
    data = r.json()
    out = []
    for v in data.get("vulnerabilities", []):
        added = v.get("dateAdded") or ""
        try:
            d = date.fromisoformat(added)
        except ValueError:
            continue
        if d >= CUTOFF:
            cve = v.get("cveID") or v.get("cveId") or ""
            doc_id = f"kev:{cve}" if cve else f"kev:{v.get('vulnerabilityName', 'unknown')}"
            text = (
                f"{cve} KEV {v.get('vulnerabilityName', '')}. "
                f"Vendor: {v.get('vendorProject', '')}. Product: {v.get('product', '')}. "
                f"Description: {v.get('shortDescription', '')}. "
                f"Required action: {v.get('requiredAction', '')}. "
                f"dateAdded={added}"
            )
            out.append(
                {
                    "id": doc_id,
                    "text": text,
                    "source": "kev",
                    "cve_id": cve,
                    "date": added,
                    "packages": [],
                }
            )
    return out


def fetch_osv_recent(client: httpx.Client) -> list[dict]:
    """Pull a few hundred 2025–2026 OSV entries via ecosystem package queries."""
    # Representative packages commonly seen in audits; not exhaustive.
    packages = [
        ("PyPI", "requests"),
        ("PyPI", "urllib3"),
        ("PyPI", "django"),
        ("PyPI", "flask"),
        ("PyPI", "pillow"),
        ("PyPI", "cryptography"),
        ("npm", "lodash"),
        ("npm", "axios"),
        ("npm", "next"),
        ("Go", "golang.org/x/net"),
        ("Maven", "org.apache.logging.log4j:log4j-core"),
        ("crates.io", "openssl"),
    ]
    seen: set[str] = set()
    out: list[dict] = []
    for eco, name in packages:
        if len(out) >= MAX_RECENT:
            break
        try:
            r = client.post(
                OSV_QUERY,
                json={"package": {"name": name, "ecosystem": eco}},
                timeout=60.0,
            )
            if r.status_code != 200:
                continue
            vulns = r.json().get("vulns") or []
        except Exception:
            continue
        for v in vulns:
            if len(out) >= MAX_RECENT:
                break
            vid = v.get("id") or ""
            if not vid or vid in seen:
                continue
            published = (v.get("published") or "")[:10]
            try:
                d = date.fromisoformat(published) if published else None
            except ValueError:
                d = None
            if d is None or d < CUTOFF:
                # Also accept aliases CVE-2025/2026 even if published parse fails
                aliases = v.get("aliases") or []
                if not any(a.startswith("CVE-2025") or a.startswith("CVE-2026") for a in aliases + [vid]):
                    continue
            seen.add(vid)
            cves = [a for a in (v.get("aliases") or []) if a.startswith("CVE-")]
            if vid.startswith("CVE-"):
                cves = [vid] + cves
            primary_cve = cves[0] if cves else vid
            summary = v.get("summary") or (v.get("details") or "")[:500]
            affected_pkgs = []
            for aff in v.get("affected") or []:
                pkg = aff.get("package") or {}
                pname = pkg.get("name")
                if pname:
                    versions = []
                    for rge in aff.get("ranges") or []:
                        for ev in rge.get("events") or []:
                            if "introduced" in ev:
                                versions.append(f"introduced:{ev['introduced']}")
                            if "fixed" in ev:
                                versions.append(f"fixed:{ev['fixed']}")
                    for ver in aff.get("versions") or []:
                        versions.append(ver)
                    affected_pkgs.append(
                        {
                            "ecosystem": pkg.get("ecosystem"),
                            "name": pname,
                            "versions": versions[:20],
                        }
                    )
            text = (
                f"{primary_cve} {vid}. Package:{name} ecosystem:{eco}. "
                f"Summary: {summary}. Published: {published}."
            )
            out.append(
                {
                    "id": f"osv:{vid}",
                    "text": text,
                    "source": "osv",
                    "cve_id": primary_cve if primary_cve.startswith("CVE-") else "",
                    "date": published,
                    "packages": affected_pkgs,
                }
            )
    return out


def seed_fallback_docs() -> list[dict]:
    """Minimal offline seed if network fails — still post-cutoff themed demos."""
    return [
        {
            "id": "demo:CVE-2025-32711",
            "text": "CVE-2025-32711 Microsoft M365 Copilot AI command injection via prompt injection. KEV-listed 2025.",
            "source": "seed",
            "cve_id": "CVE-2025-32711",
            "date": "2025-06-01",
            "packages": [],
        },
        {
            "id": "demo:CVE-2025-24813",
            "text": "CVE-2025-24813 Apache Tomcat path equivalence remote code execution. Critical. dateAdded 2025.",
            "source": "seed",
            "cve_id": "CVE-2025-24813",
            "date": "2025-03-10",
            "packages": [{"ecosystem": "Maven", "name": "org.apache.tomcat:tomcat", "versions": ["introduced:0", "fixed:9.0.99"]}],
        },
        {
            "id": "demo:CVE-2025-29927",
            "text": "CVE-2025-29927 Next.js middleware authorization bypass. npm next package. 2025.",
            "source": "seed",
            "cve_id": "CVE-2025-29927",
            "date": "2025-03-21",
            "packages": [{"ecosystem": "npm", "name": "next", "versions": ["introduced:0", "fixed:14.2.25"]}],
        },
    ]


def build_structured(docs: list[dict]) -> dict:
    by_cve: dict[str, list[str]] = {}
    by_package: dict[str, list[str]] = {}
    kev_ids: set[str] = set()
    for d in docs:
        cve = d.get("cve_id") or ""
        if cve:
            by_cve.setdefault(cve.upper(), []).append(d["id"])
        if d.get("source") == "kev" and cve:
            kev_ids.add(cve.upper())
        for p in d.get("packages") or []:
            key = f"{p.get('ecosystem', '')}:{p.get('name', '')}".lower()
            by_package.setdefault(key, []).append(d["id"])
            by_package.setdefault(p.get("name", "").lower(), []).append(d["id"])
    return {
        "by_cve": {k: sorted(set(v)) for k, v in by_cve.items()},
        "by_package": {k: sorted(set(v)) for k, v in by_package.items()},
        "kev_cves": sorted(kev_ids),
        "doc_ids": [d["id"] for d in docs],
    }


def persist_chroma(docs: list[dict]) -> None:
    import chromadb
    from chromadb.utils import embedding_functions

    PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    from core.offline import enable_hf_offline, resolve_embedding_model

    model_name = resolve_embedding_model(EMBED_MODEL)
    enable_hf_offline()
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model_name,
        local_files_only=True,
    )
    client = chromadb.PersistentClient(path=str(PERSIST_DIR))
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = client.create_collection(name=COLLECTION, embedding_function=ef)
    ids = [d["id"] for d in docs]
    documents = [d["text"] for d in docs]
    metadatas = [
        {
            "source": d.get("source", ""),
            "cve_id": d.get("cve_id") or "",
            "date": d.get("date") or "",
        }
        for d in docs
    ]
    # Chroma batch limit
    batch = 100
    for i in range(0, len(ids), batch):
        col.add(
            ids=ids[i : i + batch],
            documents=documents[i : i + batch],
            metadatas=metadatas[i : i + batch],
        )
    META_PATH.write_text(json.dumps(build_structured(docs), indent=2))
    print(f"Indexed {len(docs)} docs -> {PERSIST_DIR}")


def main() -> None:
    docs: list[dict] = []
    try:
        with httpx.Client(follow_redirects=True) as client:
            print("Fetching CISA KEV (post-2025-01-01)...")
            kev = fetch_kev(client)
            print(f"  KEV kept: {len(kev)}")
            print("Fetching OSV recent vulns...")
            osv = fetch_osv_recent(client)
            print(f"  OSV kept: {len(osv)}")
            docs = kev + osv
    except Exception as e:
        print(f"Network fetch failed ({e}); using seed docs.")
        docs = []

    if len(docs) < 10:
        docs = docs + seed_fallback_docs()

    # Dedupe by id
    by_id = {d["id"]: d for d in docs}
    docs = list(by_id.values())
    persist_chroma(docs)


if __name__ == "__main__":
    main()
