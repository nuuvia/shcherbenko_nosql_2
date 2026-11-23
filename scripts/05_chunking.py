import os
import re
import time
import numpy as np
import pandas as pd

from pathlib import Path
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = "allenai/specter2_base"

FIXED_INDEX = "arxiv-chunks-fixed"
SEMANTIC_INDEX = "arxiv-chunks-semantic"

DIMENSION = 768
MAX_WORDS = 100
OVERLAP = 20
BATCH_SIZE = 32
UPSERT_BATCH = 100
TOP_K = 5


def fixed_chunks(text):
    """Split into fixed word windows with overlap."""
    words = text.split()
    chunks = []
    step = MAX_WORDS - OVERLAP

    for start in range(0, len(words), step):
        chunk = words[start:start + MAX_WORDS]
        if not chunk:
            break

        chunks.append(" ".join(chunk))

        if start + MAX_WORDS >= len(words):
            break

    return chunks


def semantic_chunks(text):
    """Group complete sentences up to MAX_WORDS."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks = []
    current = []
    current_length = 0

    for sentence in sentences:
        words = sentence.split()
        if not words:
            continue

        # A single very long sentence must be split.
        if len(words) > MAX_WORDS:
            if current:
                chunks.append(" ".join(current))
                current = []
                current_length = 0

            for start in range(0, len(words), MAX_WORDS):
                chunks.append(
                    " ".join(words[start:start + MAX_WORDS])
                )
            continue

        if current and current_length + len(words) > MAX_WORDS:
            chunks.append(" ".join(current))
            current = []
            current_length = 0

        current.append(sentence)
        current_length += len(words)

    if current:
        chunks.append(" ".join(current))

    return chunks


def prepare_chunks(df, strategy, split_function):
    """Build chunk records with paper metadata."""
    records = []

    for paper_number, paper in df.iterrows():
        pieces = split_function(str(paper["abstract"]))

        for chunk_number, chunk in enumerate(pieces):
            records.append({
                "id": f"{strategy}_{paper_number}_{chunk_number}",
                "arxiv_id": str(paper["id"]),
                "title": str(paper["title"]),
                "chunk_text": chunk,
                "chunk_number": chunk_number,
                "year": int(paper["year"]),
                "category": str(paper["category"]),
            })

    return records


def create_index(pc, name):
    """Create a Pinecone dense index if necessary."""
    if name not in pc.list_indexes().names():
        print(f"Creating index: {name}")

        pc.create_index(
            name=name,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )

    # Wait until the index is ready.
    for _ in range(60):
        info = pc.describe_index(name)

        if info.dimension != DIMENSION or info.metric != "cosine":
            raise ValueError(f"Invalid index configuration: {name}")

        if info.status["ready"]:
            return pc.Index(name)

        time.sleep(2)

    raise TimeoutError(f"Index not ready: {name}")


def upload_chunks(index, model, records):
    """Generate embeddings and upsert chunk batches."""
    texts = [
        record["title"] + " [SEP] " + record["chunk_text"]
        for record in records
    ]

    vectors = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    assert vectors.shape == (len(records), DIMENSION)

    for start in tqdm(
        range(0, len(records), UPSERT_BATCH),
        desc="Uploading chunks",
    ):
        end = min(start + UPSERT_BATCH, len(records))
        batch = []

        for i in range(start, end):
            record = records[i]

            batch.append({
                "id": record["id"],
                "values": vectors[i].tolist(),
                "metadata": {
                    "arxiv_id": record["arxiv_id"],
                    "title": record["title"][:300],
                    "chunk_text": record["chunk_text"][:1000],
                    "chunk_number": record["chunk_number"],
                    "year": record["year"],
                    "category": record["category"],
                },
            })

        result = index.upsert(vectors=batch)

        if result.upserted_count != len(batch):
            raise RuntimeError("Incomplete Pinecone upsert")

    return vectors


def search_chunks(index, model, query, strategy):
    """Search the corresponding chunk index."""
    query_vector = model.encode(
        query,
        normalize_embeddings=True,
    )

    response = index.query(
        vector=query_vector.tolist(),
        top_k=TOP_K,
        include_metadata=True,
    )

    print(f"\n--- {strategy}: {query} ---")

    for rank, match in enumerate(response.matches, 1):
        metadata = match.metadata or {}

        print(
            f"{rank}. {metadata.get('title', 'Unknown')}\n"
            f"   Score: {match.score:.4f}\n"
            f"   Chunk: {metadata.get('chunk_text', '')[:180]}...\n"
        )


def main():
    load_dotenv(ROOT / ".env")

    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("Missing PINECONE_API_KEY")

    df = pd.read_parquet(
        ROOT / "data" / "arxiv_subset.parquet"
    )

    # Select the 30 papers with the longest abstracts.
    df = (
        df.assign(
            word_count=df["abstract"].str.split().str.len()
        )
        .nlargest(30, "word_count")
        .reset_index(drop=True)
    )

    print(f"Selected papers: {len(df)}")
    print(f"Longest abstract: {df['word_count'].max()} words")
    print(f"Shortest selected: {df['word_count'].min()} words")

    fixed = prepare_chunks(df, "fixed", fixed_chunks)
    semantic = prepare_chunks(df, "semantic", semantic_chunks)

    print(f"\nFixed chunks: {len(fixed)}")
    print(f"Semantic chunks: {len(semantic)}")

    for name, records in [
        ("Fixed", fixed),
        ("Semantic", semantic),
    ]:
        lengths = [len(r["chunk_text"].split()) for r in records]
        print(
            f"{name}: avg={np.mean(lengths):.1f} words, "
            f"max={max(lengths)} words"
        )

    model = SentenceTransformer(MODEL_NAME)
    pc = Pinecone(api_key=api_key)

    fixed_index = create_index(pc, FIXED_INDEX)
    semantic_index = create_index(pc, SEMANTIC_INDEX)

    print("\n=== Encoding fixed-size chunks ===")
    upload_chunks(fixed_index, model, fixed)

    print("\n=== Encoding semantic chunks ===")
    upload_chunks(semantic_index, model, semantic)

    # Use one query based on the selected corpus
    # and two general scientific questions.
    queries = [
        "machine learning methods for scientific research",
        "neural networks and image recognition",
        str(df.iloc[0]["title"]),
    ]

    for query in queries:
        search_chunks(fixed_index, model, query, "FIXED")
        search_chunks(semantic_index, model, query, "SEMANTIC")

    print("\n=== Index Statistics ===")
    for name, index in [
        (FIXED_INDEX, fixed_index),
        (SEMANTIC_INDEX, semantic_index),
    ]:
        stats = index.describe_index_stats()
        print(
            f"{name}: {stats.total_vector_count} vectors"
        )

    print("\nChunking experiments completed.")


if __name__ == "__main__":
    main()
