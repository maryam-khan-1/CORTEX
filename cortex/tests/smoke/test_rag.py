from pathlib import Path
assert Path("data/chroma/structured_lookup.json").exists() or Path("data/chroma").exists()
from core.rag import RAG
r = RAG(); docs = r.retrieve_for_query("Apache Tomcat remote code execution", top_k=3)
assert docs and all(d.id for d in docs); print("rag ok", [d.id for d in docs])
