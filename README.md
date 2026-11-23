## Part 1 — Data Preparation and Embeddings

### 1.1 Dataset Preparation

A reproducible sample of **5,000 arXiv papers** was selected using reservoir sampling (`seed=42`) from the [arXiv Dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv).

- Valid records scanned: 3,252,452
- Selected papers: 5,000 (all unique, no missing abstracts)
- Publication years: 1991–2026
- Recent `cs.LG` papers (2021+): 197
- Output: `data/arxiv_subset.parquet`

### 1.2 Tool Selection

#### Q1. Pinecone vs Qdrant vs Chroma

| Criterion | Pinecone | Qdrant | Chroma |
|---|---|---|---|
| Deployment | Managed cloud | Self-hosted / Cloud | Embedded / Server / Cloud |
| License | Proprietary | Apache 2.0 | Apache 2.0 |
| Strength | Minimal maintenance | Control, filtering, hybrid search | Easy prototyping |
| Trade-off | Cost, vendor lock-in | Self-hosting overhead | Embedded scalability |
| Best use | Managed production search | Enterprise self-hosted search | Local RAG / PoC |

**My assessment:** I would choose **Pinecone** for production search with minimal infrastructure management, **Qdrant** when deployment control and advanced filtering are priorities, and **Chroma** for rapid local development. Performance depends on workload and deployment configuration.

#### Q2. Why SPECTER2 instead of all-MiniLM-L6-v2?

[SPECTER2](https://huggingface.co/allenai/specter2_base) specializes in scientific document representations using citation relationships and SciRepEval tasks. Its supported task formats include **Classification, Regression, Proximity (Retrieval), and Adhoc Search**.

Unlike the general-purpose [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (384 dimensions), SPECTER2 produces 768-dimensional representations and is designed for scientific text. However, retrieval quality still requires evaluation.

**Implementation note:** Sentence Transformers automatically applies mean pooling to `specter2_base`, while the official SPECTER2 implementation uses CLS pooling and task-specific adapters.

#### Q3. Which similarity metric should be used?

The SPECTER2 model card does not explicitly require one similarity metric. **Cosine similarity** was selected because the embeddings are L2-normalized. It measures vector orientation independently of magnitude, making it suitable for semantic similarity.

The Pinecone index metric must match the intended embedding comparison and ranking behavior.

### 1.3 Embedding Generation

Texts were encoded as `title + " [SEP] " + abstract` using `allenai/specter2_base`, batch processing, and L2 normalization.

| Metric | Result |
|---|---|
| Papers processed | 5,000 |
| Embedding shape | `(5000, 768)` |
| First vector L2 norm | 1.000000 |
| Processing time | ~44 minutes |
| Output | `embeddings/embeddings.npy` |

#### Why are cosine similarity and dot product equivalent for normalized embeddings?

Cosine similarity is the dot product divided by both vector lengths. With L2-normalized embeddings, both lengths equal 1, so **cosine similarity equals dot product**.

Therefore, both metrics produce identical rankings for normalized vectors, apart from minor floating-point differences.

## Part 2 — Pinecone Index & Metadata Upload

### Implementation

Created the `arxiv-papers` Pinecone serverless index with **768 dimensions** and the `cosine` similarity metric.

Uploaded embeddings in batches of 100 using `upsert()`. Each vector includes a unique ID and metadata: arXiv ID, title, truncated abstract, authors, publication year, and primary category.

### Results

| Metric | Result |
|---|---|
| Index | `arxiv-papers` |
| Dimensions | 768 |
| Similarity metric | Cosine |
| Vectors uploaded | 5,000 |
| Batch size | 100 |
| Upload batches | 50 |
| Final index count | 5,000 |

The uploaded vector count matches the number of source articles. Metadata enables filtering by publication year and research category in subsequent searches.

## Part 3 — Semantic Search and Similarity Metrics

### Implementation

Implemented semantic search with Pinecone and `allenai/specter2_base`, including metadata filtering by publication year and category.

Compared three similarity metrics locally using normalized embeddings: cosine similarity, dot product, and Euclidean (L2) distance.

### Results

| Experiment | Result |
|---|---|
| Semantic search | Top-5 papers retrieved |
| Recent `cs.LG` papers (2022+) | 5 matching results |
| Older papers (before 2015) | 5 matching results |
| Cosine vs Dot Product | Identical Top-5 |
| Cosine vs L2 | Identical Top-5 |

For the query *"teaching machines to recognize objects in pictures"*, the top result was a paper about neural visual recognition (score: 0.8602).

Metadata filtering successfully restricted the results by category and publication year. Local similarity calculations produced identical rankings across all three metrics.

### Theoretical Questions

**Q1. Do cosine and dot product return the same Top-5?**

Yes. All embeddings were L2-normalized, so cosine similarity equals dot product. Both metrics returned identical Top-5 rankings.

**Q2. Do L2 results differ? Why?**

No. For unit-length vectors, squared Euclidean distance equals `2 - 2 × cosine_similarity`. Therefore, minimizing L2 distance is equivalent to maximizing cosine similarity, resulting in the same ranking.

**Q3. What happens without normalization?**

Without normalization, dot product becomes sensitive to vector magnitude, potentially producing different rankings from cosine similarity. Euclidean distance also depends on magnitude, so the equivalence between all three ranking methods no longer holds.

## Part 4 — Chunking Strategies

### Implementation

Selected the 30 papers with the longest abstracts (294–338 words) and compared two chunking strategies:

- **Fixed-size:** 100-word chunks with 20-word overlap.
- **Sentence-aware semantic:** groups complete sentences into chunks of up to 100 words.

Generated normalized SPECTER2 embeddings and uploaded the chunks into separate Pinecone indexes.

### Results

| Metric | Fixed-size | Semantic |
|---|---:|---:|
| Chunks | 120 | 122 |
| Average length | 91.6 words | 75.3 words |
| Maximum length | 100 words | 100 words |
| Pinecone vectors | 120 | 122 |
| Search | Top-5 returned | Top-5 returned |

Both strategies successfully retrieved relevant chunks. For the microgravity heat pipe query, the top similarity scores were **0.9273 (fixed)** and **0.9533 (semantic)**. This is an individual example, not proof of overall retrieval superiority.

### Theoretical Questions

**Q1. Which strategy produces more meaningful chunks?**

Sentence-aware chunking generally preserves context better because it avoids splitting sentences. Fixed-size chunking provides more uniform lengths but can interrupt ideas. Our experiments showed comparable retrieval results, with some higher scores for sentence-aware chunks.

**Q2. How do split sentences affect embeddings?**

Fixed-size chunking may break sentences, removing important context and reducing embedding quality. Sentence-aware chunking avoids most such splits, except when individual sentences exceed the maximum length.

**Q3. How does overlap affect chunk count and coverage?**

Overlap repeats text between neighboring chunks, improving context preservation near boundaries. However, larger overlap increases the number of chunks, storage requirements, and embedding costs. Our fixed-size strategy used 20-word overlap, while sentence-aware chunking used none.

Full output: `output/05_chunking_log.txt`

## Part 5 — Hybrid Search

### Implementation

Implemented hybrid retrieval using local **BM25** and Pinecone vector search. The Top-20 candidates from each method are merged using **Reciprocal Rank Fusion (RRF)** with `k=60`.

RRF combines document ranks instead of incompatible BM25 and cosine similarity scores.

### Results

| Query | BM25 | Vector Search | Hybrid RRF |
|---|---|---|---|
| BERT fine-tuning | Finds lexical matches, including unrelated uses of "fine-tuned" | Mostly weak matches | Combines results but remains limited in relevance |
| Yann LeCun convolutional networks | Finds CNN-related papers | Finds related neural network research | Promotes papers retrieved by both methods |
| Understanding human emotions from text | Relies on matching words | Finds depression, happiness, and sentiment papers | Combines lexical and semantic results |

### Theoretical Questions

**Q1. Which method performed better?**

Results depended on query type. BM25 was useful for technical terminology, while vector search found related concepts without exact keyword matches. Hybrid search improved ranking agreement for the CNN query, but did not consistently outperform individual methods.

**Q2. Can Hybrid Top-5 contain papers absent from both individual Top-5 lists?**

Yes. For the CNN query, *Scale Space Approximation in Convolutional Neural Networks for Retinal Vessel Segmentation* appeared at Hybrid rank 3 despite being absent from both individual Top-5 lists. RRF considers the wider Top-20 candidate lists and rewards documents ranked by both methods.

**Q3. How does the RRF parameter k affect rankings?**

With `k=60`, differences between individual ranks have a smaller impact, favoring agreement across retrieval methods. With `k=1`, top-ranked documents receive substantially higher weight, making the final ranking more sensitive to the highest positions.

## Part 6 — Analysis & Conclusions

### Q1. Semantic Search vs Keyword Search

Semantic search is effective for natural-language queries, paraphrases, and related concepts, while BM25 works better for exact terms, technical identifiers, and author names.

In our experiments, vector search retrieved papers about depression, happiness, and sentiment for the query about understanding human emotions from text. BM25 mainly relied on matching words. Hybrid search combined both approaches through RRF, but did not consistently produce more relevant results.

**Conclusion:** Use semantic search for conceptual similarity, BM25 for exact terminology, and hybrid retrieval when both types of relevance matter. Actual quality should be validated against a relevance-labeled evaluation dataset.

### Q2. Impact of Chunk Size

Very small chunks (10–15 words) may lose essential context and produce ambiguous embeddings. Very large chunks (500+ words) may combine unrelated ideas, reduce retrieval precision, and exceed the embedding model's token limit.

Our experiment compared 100-word fixed-size chunks with 20-word overlap against sentence-aware chunks. The sentence-aware strategy preserved sentence boundaries and produced shorter chunks on average (75.3 vs 91.6 words).

**Conclusion:** There is no universally optimal chunk size. It depends on document structure, embedding model, retrieval task, and evaluation results. Overlap improves boundary coverage but increases storage and embedding costs.

### Q3. Euclidean Distance vs Cosine Similarity

For L2-normalized vectors, cosine similarity, dot product, and Euclidean distance produce equivalent rankings.

The mathematical relationship for unit-length vectors is:

`squared_L2_distance = 2 - 2 * cosine_similarity`

Therefore, maximizing cosine similarity is equivalent to minimizing Euclidean distance.

Creating a Pinecone index with the `euclidean` metric instead of `cosine` would not change the ideal nearest-neighbor ranking for our normalized embeddings, although reported scores would differ. Small numerical or approximate-search differences are still possible.

Without normalization, this equivalence no longer holds because vector magnitude affects dot product and Euclidean distance.

### Q4. Pinecone Starter Limitations and Scaling

Pinecone Starter is suitable for small educational datasets but has limits on available resources, storage, requests, and deployment options.

Our experiments successfully indexed 5,000 scientific papers and created two additional indexes containing 120 fixed-size and 122 sentence-aware chunks. We did not encounter capacity limitations during these operations.

For 10 million papers, I would use a production-scale deployment, batch or parallel ingestion, metadata filtering, and incremental updates. I would also evaluate index capacity, retrieval latency, storage costs, embedding generation costs, and retrieval quality.

At 768 dimensions and float32 precision, 10 million raw embeddings alone require approximately 30.7 GB (28.6 GiB), excluding index structures, metadata, replicas, and other overhead.

**Conclusion:** The experiments demonstrate the complete semantic retrieval pipeline, but production scalability and quality require additional benchmarking, monitoring, and cost optimization.
