import argparse
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT / "data" / "arxiv_subset.parquet"
OUTPUT_FILE = ROOT / "embeddings" / "embeddings.npy"

MODEL_NAME = "allenai/specter2_base"
BATCH_SIZE = 32


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    df = pd.read_parquet(INPUT_FILE)

    if df.empty:
        raise ValueError("Dataset is empty")

    texts = (
        df["title"].astype(str)
        + " [SEP] "
        + df["abstract"].astype(str)
    ).tolist()

    if args.smoke_test:
        texts = texts[:2]

    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    print(f"Processed texts: {len(texts)}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"First vector norm: {np.linalg.norm(embeddings[0]):.6f}")

    assert embeddings.shape == (len(texts), 768)
    assert np.allclose(
        np.linalg.norm(embeddings, axis=1),
        1.0,
        atol=1e-4,
    )

    if args.smoke_test:
        print("Smoke test passed; embeddings not saved.")
        return

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_FILE, embeddings)

    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
