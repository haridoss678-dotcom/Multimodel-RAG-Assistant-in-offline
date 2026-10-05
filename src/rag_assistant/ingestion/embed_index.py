from rag_assistant.ingestion.models import Chunk
from rag_assistant.storage.vector_store import upsert_chunks

_model = None


def _get_model():
    global _model
    if _model is None:
        # Imported lazily: BGE-M3 is only needed once embedding actually runs,
        # not at module import time (keeps startup light, RAM matters here - §29).
        from FlagEmbedding import BGEM3FlagModel

        _model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False, devices=["cpu"])
    return _model


def _to_sparse_dict(lexical_weights: dict) -> dict:
    """BGE-M3's lexical_weights is {token_id: weight}; Qdrant sparse vectors
    want parallel indices/values lists (CLAUDE.md §17)."""
    indices = [int(k) for k in lexical_weights]
    values = [float(v) for v in lexical_weights.values()]
    return {"indices": indices, "values": values}


def embed_and_index(chunks: list[Chunk]) -> None:
    """Hybrid dense+sparse embedding (CLAUDE.md §17/§32 V2 - one BGE-M3 call
    produces both, which is exactly why BGE-M3 was chosen over a dense-only
    model, §13)."""
    if not chunks:
        return
    model = _get_model()
    texts = [c.text for c in chunks]
    output = model.encode(texts, return_dense=True, return_sparse=True, return_colbert_vecs=False)
    dense_vectors = [v.tolist() for v in output["dense_vecs"]]
    sparse_vectors = [_to_sparse_dict(w) for w in output["lexical_weights"]]
    payloads = [c.model_dump() for c in chunks]
    upsert_chunks(payloads, dense_vectors, sparse_vectors)


def embed_query(text: str) -> tuple[list[float], dict]:
    model = _get_model()
    output = model.encode([text], return_dense=True, return_sparse=True, return_colbert_vecs=False)
    return output["dense_vecs"][0].tolist(), _to_sparse_dict(output["lexical_weights"][0])
