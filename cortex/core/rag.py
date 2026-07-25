"""ChromaDB retrieve + structured CVE/KEV/package lookup."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PERSIST = ROOT / "data" / "chroma"
META_PATH = DEFAULT_PERSIST / "structured_lookup.json"
COLLECTION = "cortex_cve"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CVE_RE = re.compile(r"\bCVE-\d{4}-\d{4,}\b", re.IGNORECASE)


@dataclass
class RetrievedDoc:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    distance: Optional[float] = None


class RAG:
    """Local embeddings + Chroma. Semantic for vuln-class explain; structured for package/CVE."""

    def __init__(
        self,
        persist_dir: Optional[Path] = None,
        collection: str = COLLECTION,
        embedding_model: str = EMBED_MODEL,
    ):
        import chromadb
        from chromadb.utils import embedding_functions

        self.persist_dir = Path(persist_dir or DEFAULT_PERSIST)
        self.collection_name = collection
        self.embedding_model = embedding_model
        self.meta = self._load_meta()
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.embedding_model
        )
        self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        self._col = self._client.get_collection(
            name=self.collection_name, embedding_function=ef
        )

    def _load_meta(self) -> dict[str, Any]:
        path = self.persist_dir / "structured_lookup.json"
        if path.exists():
            return json.loads(path.read_text())
        return {"by_cve": {}, "by_package": {}, "kev_cves": [], "doc_ids": []}

    def semantic_search(self, query: str, top_k: int = 5) -> list[RetrievedDoc]:
        res = self._col.query(query_texts=[query], n_results=top_k)
        docs: list[RetrievedDoc] = []
        ids = (res.get("ids") or [[]])[0]
        documents = (res.get("documents") or [[]])[0]
        metadatas = (res.get("metadatas") or [[]])[0]
        distances = (res.get("distances") or [[]])[0]
        for i, doc_id in enumerate(ids):
            docs.append(
                RetrievedDoc(
                    id=doc_id,
                    text=documents[i] or "",
                    metadata=metadatas[i] or {},
                    distance=distances[i] if i < len(distances) else None,
                )
            )
        return docs

    def lookup_cve(self, cve_id: str) -> list[RetrievedDoc]:
        cve = cve_id.upper().strip()
        ids = list(self.meta.get("by_cve", {}).get(cve, []))
        return self._get_by_ids(ids)

    def lookup_kev(self, cve_id: str) -> list[RetrievedDoc]:
        cve = cve_id.upper().strip()
        if cve not in set(self.meta.get("kev_cves") or []):
            # Still try kev: prefix id
            return self._get_by_ids([f"kev:{cve}"])
        return self.lookup_cve(cve)

    def lookup_package(self, name: str, ecosystem: str = "") -> list[RetrievedDoc]:
        key = f"{ecosystem}:{name}".lower() if ecosystem else name.lower()
        ids = list(self.meta.get("by_package", {}).get(key, []))
        if not ids:
            ids = list(self.meta.get("by_package", {}).get(name.lower(), []))
        return self._get_by_ids(ids)

    def _get_by_ids(self, ids: list[str]) -> list[RetrievedDoc]:
        if not ids:
            return []
        # Filter to known ids
        known = set(self.meta.get("doc_ids") or [])
        ids = [i for i in ids if not known or i in known]
        if not ids:
            return []
        try:
            res = self._col.get(ids=ids, include=["documents", "metadatas"])
        except Exception:
            return []
        out: list[RetrievedDoc] = []
        for i, doc_id in enumerate(res.get("ids") or []):
            docs = res.get("documents") or []
            metas = res.get("metadatas") or []
            out.append(
                RetrievedDoc(
                    id=doc_id,
                    text=(docs[i] if i < len(docs) else "") or "",
                    metadata=(metas[i] if i < len(metas) else {}) or {},
                )
            )
        return out

    def retrieve_for_query(self, query: str, top_k: int = 5) -> list[RetrievedDoc]:
        """Hybrid: structured CVE/package hits first, then semantic fill."""
        found: dict[str, RetrievedDoc] = {}
        for cve in CVE_RE.findall(query):
            for d in self.lookup_cve(cve):
                found[d.id] = d
        # crude package tokens from import/require-ish strings
        for token in re.findall(r"[A-Za-z0-9_.-]{3,}", query):
            if token.lower() in {"import", "from", "require", "the", "and", "for"}:
                continue
            for d in self.lookup_package(token):
                found[d.id] = d
                if len(found) >= top_k:
                    break
            if len(found) >= top_k:
                break
        if len(found) < top_k:
            for d in self.semantic_search(query, top_k=top_k):
                found.setdefault(d.id, d)
                if len(found) >= top_k:
                    break
        return list(found.values())[:top_k]

    def format_evidence(self, docs: list[RetrievedDoc]) -> str:
        lines = []
        for d in docs:
            lines.append(f"[{d.id}] {d.text}")
        return "\n".join(lines)

    @property
    def all_doc_ids(self) -> set[str]:
        return set(self.meta.get("doc_ids") or [])
