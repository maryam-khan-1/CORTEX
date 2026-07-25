from core.offline import resolve_embedding_model, enable_hf_offline
enable_hf_offline()
p = resolve_embedding_model("data/models/all-MiniLM-L6-v2")
assert "all-MiniLM" in p and __import__("pathlib").Path(p).is_dir()
print("offline ok", p)
