import json
import random
import pandas as pd

from pathlib import Path
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT / "arxiv-metadata-oai-snapshot.json"
OUTPUT_FILE = ROOT / "data" / "arxiv_subset.parquet"

MAX_RECORDS = 5_000


def extract_year(paper: dict) -> int:
    """Extract publication year from first arXiv version."""
    versions = paper.get("versions") or []

    if versions:
        try:
            return int(versions[0]["created"].split()[3])
        except (IndexError, KeyError, TypeError, ValueError):
            pass

    return int((paper.get("update_date") or "2000-01-01")[:4])


def format_authors(paper: dict) -> str:
    """Convert parsed authors into a readable string."""
    parsed = paper.get("authors_parsed") or []

    if parsed:
        authors = []
        for entry in parsed[:10]:
            if not entry:
                continue

            surname = str(entry[0]).strip()
            initials = str(entry[1]).strip() if len(entry) > 1 else ""

            if surname:
                authors.append(f"{surname} {initials}".strip())

        if authors:
            return ", ".join(authors)

    return (paper.get("authors") or "").replace("\n", " ").strip()



def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(42)
    records = []
    valid_count = 0

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        for line in tqdm(file, desc="Sampling arXiv JSONL"):
            if not line.strip():
                continue

            paper = json.loads(line)

            title = " ".join((paper.get("title") or "").split())
            abstract = " ".join((paper.get("abstract") or "").split())

            if not title or not abstract:
                continue

            categories = (paper.get("categories") or "unknown").split()

            record = {
                "id": paper["id"],
                "title": title,
                "abstract": abstract,
                "authors": format_authors(paper),
                "year": extract_year(paper),
                "category": categories[0] if categories else "unknown",
            }

            valid_count += 1

            if len(records) < MAX_RECORDS:
                records.append(record)
            else:
                j = rng.randrange(valid_count)
                if j < MAX_RECORDS:
                    records[j] = record

    if len(records) != MAX_RECORDS:
        raise ValueError(
            f"Expected {MAX_RECORDS} records, got {len(records)}"
        )

    df = pd.DataFrame(records)
    df = df.sort_values("id").reset_index(drop=True)

    if df["id"].nunique() != len(df):
        raise ValueError("Duplicate arXiv IDs detected")

    df.to_parquet(OUTPUT_FILE, index=False)

    print(f"\nValid records scanned: {valid_count}")
    print(f"Selected records: {len(df)}")
    print(f"Unique IDs: {df['id'].nunique()}")
    print(f"Missing abstracts: {df['abstract'].eq('').sum()}")

    print("\nTop categories:")
    print(df["category"].value_counts().head(10))

    print("\nYear distribution:")
    print(df["year"].value_counts().sort_index().tail(10))

    print(f"\nYear range: {df['year'].min()} - {df['year'].max()}")
    print(
        "Recent cs.LG papers:",
        len(df[
            (df["year"] >= 2021) &
            (df["category"] == "cs.LG")
        ])
    )

    print(f"\nSaved to: {OUTPUT_FILE}")



if __name__ == "__main__":
    main()
