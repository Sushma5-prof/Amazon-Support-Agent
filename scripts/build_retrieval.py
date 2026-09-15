import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import load_raw_data, build_candidate_pool

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np

DB_PATH  = str(PROJECT_ROOT / "data" / "processed" / "chroma_db")
RAW_PATH = str(PROJECT_ROOT / "data" / "raw" / "tweets.csv")
BATCH_SIZE = 512

def main():
    print("Loading raw data ...")
    raw_df = load_raw_data(RAW_PATH)
    print("Building candidate pool ...")
    candidates = build_candidate_pool(raw_df)
    print(f"Candidates: {len(candidates)}")

    print("Loading embedding model ...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    os.makedirs(DB_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=DB_PATH,
                                       settings=Settings(anonymized_telemetry=False))

    # Use embedding_function=None so Chroma does NOT try to download ONNX
    coll = client.get_or_create_collection(
        name="amazonhelp_history",
        metadata={"hnsw:space": "cosine"},
    )

    ids_all, docs_all, metas_all = [], [], []
    for _, row in candidates.iterrows():
        replies = row.get("historical_replies", [])
        replies_txt = "\n".join(str(r) for r in replies) if isinstance(replies, list) else str(replies)
        ids_all.append(str(row["customer_tweet_id"]))
        docs_all.append(str(row["customer_text"]))
        metas_all.append({
            "conversation_root": str(row["conversation_root"]),
            "historical_replies": replies_txt,
        })

    total = len(ids_all)
    print(f"Embedding and indexing {total} documents in batches of {BATCH_SIZE} ...")
    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        batch_ids   = ids_all[start:end]
        batch_docs  = docs_all[start:end]
        batch_metas = metas_all[start:end]

        embeddings = model.encode(batch_docs, normalize_embeddings=True, show_progress_bar=False).tolist()
        coll.upsert(ids=batch_ids, documents=batch_docs, embeddings=embeddings, metadatas=batch_metas)

        if (start // BATCH_SIZE) % 10 == 0:
            print(f"  Indexed {end}/{total} ...")

    print(f"\nChroma index built at {DB_PATH}  ({total} documents)")

if __name__ == "__main__":
    main()
