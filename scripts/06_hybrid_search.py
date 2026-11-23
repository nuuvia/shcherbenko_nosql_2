import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = "allenai/specter2_base"
INDEX_NAME = "arxiv-papers"

CANDIDATE_K = 20
TOP_K = 5
RRF_K = 60

QUERIES = [
    "BERT fine-tuning",
    "Yann LeCun convolutional networks",
    "making computers understand human emotions from text",
]


def tokenize(text):
    """Basic, consistent tokenization for BM25."""
    return re.findall(r"[a-z0-9]+", text.lower())


def rrf_fusion(bm25_results, vector_results, k=60):
    """Combine rankings using reciprocal rank fusion."""
    scores = {}

    for ranking in (bm25_results, vector_results):
        for rank, paper_id in enumerate(ranking, start=1):
            scores[paper_id] = scores.get(paper_id, 0.0) + (
                1.0 / (k + rank)
            )

    return sorted(
        scores,
        key=lambda paper_id: (-scores[paper_id], paper_id),
    ), scores


def print_results(label, ids, df, scores=None):
    print(f"\n{label}")

    for rank, paper_id in enumerate(ids[:TOP_K], 1):
        i = int(paper_id.removeprefix("paper_"))
        paper = df.iloc[i]

        score_text = (
            f" | RRF={scores[paper_id]:.5f}"
            if scores is not None else ""
        )

        print(
            f"{rank}. [{paper_id}] {paper['title']}"
            f"{score_text}"
        )


def main():
    load_dotenv(ROOT / ".env")

    df = pd.read_parquet(
        ROOT / "data" / "arxiv_subset.parquet"
    ).reset_index(drop=True)

    api_key = os.environ["PINECONE_API_KEY"]
    index = Pinecone(api_key=api_key).Index(INDEX_NAME)
    model = SentenceTransformer(MODEL_NAME)

    # 1. Build a local BM25 index
    documents = (
        df["title"].astype(str)
        + " "
        + df["abstract"].astype(str)
    ).tolist()

    tokenized_docs = [tokenize(text) for text in documents]
    bm25 = BM25Okapi(tokenized_docs)

    print(f"Indexed {len(df)} documents for BM25")

    for query in QUERIES:
        print(f"\n{'=' * 65}")
        print(f"QUERY: {query}")

        # 2. Lexical search
        bm25_scores = bm25.get_scores(tokenize(query))

        bm25_positions = np.argsort(-bm25_scores)[:CANDIDATE_K]

        # Exclude papers with no positive lexical evidence
        bm25_ids = [
            f"paper_{int(i)}"
            for i in bm25_positions
            if bm25_scores[i] > 0
        ]

        # 3. Vector search
        query_embedding = model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        response = index.query(
            vector=query_embedding.tolist(),
            top_k=CANDIDATE_K,
            include_metadata=False,
        )

        vector_ids = [match.id for match in response.matches]

        # 4. Reciprocal Rank Fusion
        hybrid_ids, hybrid_scores = rrf_fusion(
            bm25_ids,
            vector_ids,
            k=RRF_K,
        )

        # 5. Compare rankings
        print_results("BM25 TOP-5", bm25_ids, df)
        print_results("VECTOR TOP-5", vector_ids, df)
        print_results(
            "HYBRID TOP-5",
            hybrid_ids,
            df,
            hybrid_scores,
        )

    print("\nHybrid search experiments completed.")


if __name__ == "__main__":
    main()
