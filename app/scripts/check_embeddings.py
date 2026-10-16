"""Check that the ONNX (fastembed) query vectors match the sentence-transformers vectors your index was built with.

    pip install fastembed          # in the venv that already has sentence-transformers
    python -m app.scripts.check_embeddings

Prints the cosine similarity per query. Expect >= 0.99. Much lower means the index and the deployed embedder
disagree and retrieval quality would drop: then host with the default backend instead.
"""
from __future__ import annotations

import numpy as np

from app.core.config import get_settings
from app.core.embeddings import Embedder, FastEmbedEmbedder

QUERIES = [
    "Describe the anatomy of the human heart, including its four chambers and valves",
    "What are the layers of the pericardium?",
    "Which nerve supplies the diaphragm?",
    "treatment of nephrotic syndrome",
]

s = get_settings()
a = Embedder(s.embedding_model, dim=s.embedding_dim)
b = FastEmbedEmbedder(s.embedding_model, dim=s.embedding_dim)
worst = 1.0
for q in QUERIES:
    cos = float(np.dot(a.embed_query(q), b.embed_query(q)))
    worst = min(worst, cos)
    print(f"{cos:.4f}  {q}")
print("PASS" if worst >= 0.99 else "FAIL", f"(worst {worst:.4f})")
