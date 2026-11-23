import os
import numpy as np
import pandas as pd

from pathlib import Path
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parent.parent
INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 5

QUERY_SEMANTIC = "teaching machines to recognize objects in pictures"
QUERY_RECENT = "reinforcement learning for autonomous decision making"
QUERY_OLD = "neural networks for pattern recognition"


def print_results(title, results):
    print(f"\n=== {title} ===")

    if not results:
        print("No matching papers.")
        return

    for rank, paper in enumerate(results, start=1):
        print(
            f"{rank}. {paper['title']}\n"
            f"   Score: {paper['score']:.4f} | "
            f"Category: {paper['category']} | "
            f"Year: {paper['year']}\n"
            f"   Abstract: {paper['abstract'][:160]}...\n"
        )


def main():
    load_dotenv(ROOT / ".env")

    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY is missing")

    df = pd.read_parquet(
        ROOT / "data" / "arxiv_subset.parquet"
    ).reset_index(drop=True)

    embeddings = np.load(
        ROOT / "embeddings" / "embeddings.npy"
    )

    if embeddings.shape != (len(df), 768):
        raise ValueError("Dataset and embeddings do not match")

    pc = Pinecone(api_key=api_key)
    index = pc.Index(INDEX_NAME)

    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    def encode_query(text):
        return model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

    def pinecone_search(query, metadata_filter=None):
        query_vector = encode_query(query)

        arguments = {
            "vector": query_vector.tolist(),
            "top_k": TOP_K,
            "include_metadata": True,
        }

        if metadata_filter is not None:
            arguments["filter"] = metadata_filter

        response = index.query(**arguments)

        results = []

        for match in response.matches:
            metadata = match.metadata or {}

            # Retrieve the full abstract using our stable paper ID
            position = int(match.id.removeprefix("paper_"))
            if not 0 <= position < len(df):
                raise ValueError(f"Unexpected vector ID: {match.id}")

            paper = df.iloc[position]

            results.append({
                "title": metadata.get("title", paper["title"]),
                "category": metadata.get("category", paper["category"]),
                "year": metadata.get("year", int(paper["year"])),
                "abstract": paper["abstract"],
                "score": float(match.score),
            })

        return results

    # --------------------------------------------------
    # 1. Semantic Search
    # --------------------------------------------------

    print_results(
        "1. Semantic Search",
        pinecone_search(QUERY_SEMANTIC),
    )

    # --------------------------------------------------
    # 2. Metadata Filtering
    # --------------------------------------------------

    print_results(
        "2A. Recent Reinforcement Learning (cs.LG, 2022+)",
        pinecone_search(
            QUERY_RECENT,
            metadata_filter={
                "$and": [
                    {"category": {"$eq": "cs.LG"}},
                    {"year": {"$gte": 2022}},
                ]
            },
        ),
    )

    print_results(
        "2B. Older Papers (before 2015)",
        pinecone_search(
            QUERY_OLD,
            metadata_filter={"year": {"$lt": 2015}},
        ),
    )

    # --------------------------------------------------
    # 3. Local Similarity Comparison
    # --------------------------------------------------

    print("\n=== 3. Local Similarity Comparison ===")

    query_vector = encode_query(QUERY_SEMANTIC)

    # Both document and query embeddings are normalized.
    cosine_scores = (
        embeddings @ query_vector
    )

    dot_scores = (
        embeddings @ query_vector
    )

    l2_distances = np.linalg.norm(
        embeddings - query_vector,
        axis=1,
    )

    methods = {
        "Cosine Similarity": (
            cosine_scores,
            True,
        ),
        "Dot Product": (
            dot_scores,
            True,
        ),
        "L2 Distance": (
            l2_distances,
            False,
        ),
    }

    rankings = {}

    for method, (scores, descending) in methods.items():
        if descending:
            top_indices = np.argsort(-scores)[:TOP_K]
        else:
            top_indices = np.argsort(scores)[:TOP_K]

        rankings[method] = top_indices.tolist()

        results = []
        for i in top_indices:
            paper = df.iloc[i]

            results.append({
                "title": paper["title"],
                "category": paper["category"],
                "year": int(paper["year"]),
                "abstract": paper["abstract"],
                "score": float(scores[i]),
            })

        print_results(method, results)

    print("\n=== Ranking Comparison ===")

    print(
        "Cosine == Dot:",
        rankings["Cosine Similarity"]
        == rankings["Dot Product"],
    )

    print(
        "Cosine == L2:",
        rankings["Cosine Similarity"]
        == rankings["L2 Distance"],
    )

    print("\nSearch experiments completed.")


if __name__ == "__main__":
    main()
