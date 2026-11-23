import os
import numpy as np
import pandas as pd

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "arxiv_subset.parquet"
EMBEDDINGS_FILE = ROOT / "embeddings" / "embeddings.npy"

INDEX_NAME = "arxiv-papers"
VECTOR_DIM = 768
BATCH_SIZE = 100


def main():
    load_dotenv(ROOT / ".env")

    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY is missing")

    # Load and validate prepared data
    df = pd.read_parquet(DATA_FILE).reset_index(drop=True)
    embeddings = np.load(EMBEDDINGS_FILE)

    if embeddings.shape != (len(df), VECTOR_DIM):
        raise ValueError(
            f"Invalid embedding shape: {embeddings.shape}"
        )

    if df["id"].duplicated().any():
        raise ValueError("Duplicate paper IDs detected")

    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain invalid values")

    print(f"Papers: {len(df)}")
    print(f"Embeddings: {embeddings.shape}")

    # Connect to Pinecone
    pc = Pinecone(api_key=api_key)

    if INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating index: {INDEX_NAME}")

        pc.create_index(
            name=INDEX_NAME,
            dimension=VECTOR_DIM,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )
    else:
        print(f"Index already exists: {INDEX_NAME}")

    # Validate existing or newly created index
    index_info = pc.describe_index(INDEX_NAME)

    if index_info.dimension != VECTOR_DIM:
        raise ValueError("Index dimension mismatch")

    if index_info.metric != "cosine":
        raise ValueError("Index metric mismatch")

    index = pc.Index(INDEX_NAME)

    # Upload vectors in batches
    for start in tqdm(
        range(0, len(df), BATCH_SIZE),
        desc="Uploading to Pinecone",
    ):
        end = min(start + BATCH_SIZE, len(df))
        vectors = []

        for i in range(start, end):
            paper = df.iloc[i]

            metadata = {
                "arxiv_id": str(paper["id"]),
                "title": str(paper["title"]),
                "abstract": str(paper["abstract"])[:500],
                "authors": str(paper["authors"])[:200],
                "year": int(paper["year"]),
                "category": str(paper["category"]),
            }

            vectors.append({
                "id": f"paper_{i}",
                "values": embeddings[i].tolist(),
                "metadata": metadata,
            })

        response = index.upsert(vectors=vectors)

        if response.upserted_count != len(vectors):
            raise RuntimeError(
                f"Batch {start}:{end} was not fully uploaded"
            )

    # Index statistics
    stats = index.describe_index_stats()

    print(f"\nIndex: {INDEX_NAME}")
    print(f"Dimension: {stats.dimension}")
    print(f"Vectors reported: {stats.total_vector_count}")
    print(f"Expected vectors: {len(df)}")
    print("Upload requests completed.")


if __name__ == "__main__":
    main()
