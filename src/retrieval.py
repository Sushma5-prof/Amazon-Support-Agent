import os
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


class HistoricalRetriever:
    """Retrieves similar historical AmazonHelp conversations using sentence-transformers embeddings.

    Pre-computes embeddings locally (avoids Chroma ONNX model download).
    """

    def __init__(
        self,
        persist_directory: str = "data/processed/chroma_db",
        collection_name: str = "amazonhelp_history",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.persist_directory = str(Path(persist_directory))
        self.embedding_model = SentenceTransformer(embedding_model)
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_examples(self, customer_texts, historical_replies, ids):
        documents, metadatas = [], []
        for text, replies in zip(customer_texts, historical_replies):
            reply_text = "\n".join(str(r) for r in replies) if isinstance(replies, list) else str(replies)
            documents.append(str(text))
            metadatas.append({"historical_replies": reply_text})
        embeddings = self.embedding_model.encode(documents, normalize_embeddings=True).tolist()
        self.collection.upsert(
            ids=[str(x) for x in ids],
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def retrieve(self, query: str, top_k: int = 5):
        query_embedding = self.embedding_model.encode([query], normalize_embeddings=True).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        evidence = []
        docs  = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]
        for doc, meta, dist in zip(docs, metas, dists):
            evidence.append({
                "customer_message": doc,
                "historical_replies": meta.get("historical_replies", ""),
                "distance": float(dist),
                "similarity": float(1 - dist),
            })
        return evidence
