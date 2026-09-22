# Scour

A search engine built from scratch in Python: tokeniser, inverted index, BM25 ranking, quoted phrase queries, then dense retrieval, hybrid fusion and a cross-encoder reranker on top, all measured on a public IR benchmark with the same harness.

No search libraries for the core. The point was to build every stage myself and measure whether each one actually helps.

```
                 ┌─> BM25 over inverted index ─┐
query ──> tokenizer                            ├─> RRF fusion ──> cross-encoder rerank (top 50) ──> top-k
                 └─> cosine over dense index ──┘
```

**Live demo:** _paste the deployed URL here_

## Quick start

```bash
pip install -e ".[dev]"
pytest                                   # 38 tests, offline, under a second

python -m scour.eval.cranfield           # downloads the Cranfield collection to data/cranfield
python -m scour index  --docs data/cranfield/docs.jsonl --out index.pkl
python -m scour search --index index.pkl '"boundary layer" transition'
python -m scour.eval.run --data data/cranfield

pip install -e ".[neural]"               # sentence-transformers, downloads two small models on first run
python -m scour.eval.run --data data/cranfield --neural

pip install -e ".[api]"
SCOUR_DATA=data/cranfield uvicorn scour.api:app   # http://127.0.0.1:8000
```

## Results on Cranfield (1,400 docs, 225 queries)

<!-- Paste the table printed by `python -m scour.eval.run` here. -->

| Ranker | P@10 | R@10 | MAP | nDCG@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|
| TF-IDF (baseline) | | | | | | |
| BM25 (k1=1.5, b=0.75) | | | | | | |
| Dense (all-MiniLM-L6-v2) | | | | | | |
| Hybrid (BM25 + dense, RRF) | | | | | | |
| Hybrid + cross-encoder rerank (top 50) | | | | | | |

Index build time: __ s. Machine: __.

## Design decisions

**Inverted index with positions, not a document-term matrix.** A matrix over 1,400 docs and a few thousand terms is almost entirely zeros. The index stores only the terms each document actually contains, and keeps term positions so phrase queries can check adjacency without re-reading the document.

**BM25 over TF-IDF.** TF-IDF keeps rewarding a term every time it repeats, and it does not care that a long document has more chances to match. BM25 adds term-frequency saturation (`k1`) and length normalisation (`b`). The results table shows what that is worth on this collection.

**Stopwords removed before positions are assigned.** Phrase matching runs on the same token stream the index was built from, so the behaviour is consistent even though "flow of the fluid" and "flow fluid" become the same phrase. That is a deliberate trade for a smaller index.

**Phrase filtering after scoring, not instead of it.** A quoted phrase decides which documents are allowed in; BM25 still decides their order. Candidate documents are intersected starting from the rarest term so the adjacency check touches as few postings as possible.

**Hybrid via reciprocal rank fusion, not score addition.** BM25 scores and cosine similarities are on different scales, so adding them means one ranker silently dominates. RRF combines ranks, needs no tuning, and rewards documents both rankers agree on.

**Reranker after retrieval, not instead of it.** A cross-encoder reads the query and document together and is far more accurate than comparing two embeddings, but it costs one forward pass per pair. Scoring the top 50 from the hybrid stage is affordable; scoring all 1,400 per query is not.

**Deterministic ranking.** Ties are broken on doc id so the same query always returns the same order, which matters when comparing evaluation runs.

## Layout

```
scour/
  tokenizer.py   lowercase, split, stopwords
  index.py       InvertedIndex: postings with tf and positions, save/load
  bm25.py        BM25 scorer
  tfidf.py       TF-IDF baseline
  search.py      SearchEngine: BM25 + quoted phrase filtering
  dense.py       DenseIndex: sentence-transformer embeddings, cosine search
  hybrid.py      reciprocal rank fusion
  rerank.py      cross-encoder second stage
  api.py         FastAPI: /search endpoint and a one-page UI
  __main__.py    CLI: index / search
  eval/
    metrics.py   P@k, R@k, MAP, nDCG@k
    cranfield.py download and convert the Cranfield collection
    run.py       evaluate rankers, print a Markdown table
tests/           pytest suite covering every module
```

## Roadmap

- [x] Stage 1: tokeniser, inverted index, BM25, phrase queries, evaluation harness, CI
- [x] Stage 2: dense retrieval, hybrid fusion (RRF), cross-encoder reranking, FastAPI demo, Dockerfile
- [ ] Stage 3: Porter stemmer as a measured change; k1/b sweep; latency benchmark script
- [ ] Stage 4: autocomplete (trie, top-k by frequency, edit-distance typo tolerance)
- [ ] Stage 5: compressed postings (delta + varint) and index size comparison

## Benchmark notes

Cranfield relevance codes run 1 (complete answer) to 4 (minimal interest); they are mapped to gains 4..1 so nDCG rewards the best documents most. Query ids in `cranqrel` are the 1-based position of the query in `cran.qry`, not the `.I` number printed in the file. The converter handles both; see `scour/eval/cranfield.py`.
