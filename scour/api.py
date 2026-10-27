"""A small web front end for the engine: one search box, one JSON endpoint.

    SCOUR_DATA=data/cranfield uvicorn scour.api:app

GET /search?q=boundary+layer&mode=bm25|dense|hybrid|rerank&k=10

Dense, hybrid and rerank modes need the [neural] extra and download two small
models on first start; set SCOUR_NEURAL=0 to serve BM25 only.
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse

from scour.bm25 import BM25
from scour.eval.run import load_jsonl
from scour.hybrid import HybridRanker
from scour.index import InvertedIndex
from scour.search import SearchEngine

STATE: dict = {}


def _load() -> None:
    data = Path(os.environ.get("SCOUR_DATA", "data/cranfield"))
    docs = load_jsonl(data / "docs.jsonl")
    triples = [(d["id"], d.get("title", ""), d["text"]) for d in docs]
    STATE["texts"] = {d["id"]: d["text"] for d in docs}
    STATE["titles"] = {d["id"]: d.get("title", "") for d in docs}
    index = InvertedIndex.build(triples)
    STATE["bm25"] = SearchEngine(index)
    STATE["modes"] = ["bm25"]

    if os.environ.get("SCOUR_NEURAL", "1") != "0":
        try:
            from scour.dense import DenseIndex, SentenceTransformerEmbedder
            from scour.rerank import CrossEncoderScorer, RerankedRanker

            embedder = SentenceTransformerEmbedder()
            dense = DenseIndex.cached(data / "dense_index", triples, embedder, embedder.model_name)
            hybrid = HybridRanker([BM25(index), dense])
            STATE["dense"] = dense
            STATE["hybrid"] = hybrid
            STATE["rerank"] = RerankedRanker(hybrid, CrossEncoderScorer(), STATE["texts"])
            STATE["modes"] += ["dense", "hybrid", "rerank"]
        except ImportError:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load()
    yield


app = FastAPI(title="scour", version="0.2.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    """Liveness + readiness for load balancers and hosting health checks."""
    return {"status": "ok", "docs": len(STATE.get("texts", {})), "modes": STATE.get("modes", [])}


@app.get("/search")
def search(q: str = Query(..., min_length=1), mode: str = "bm25", k: int = Query(10, ge=1, le=50)) -> dict:
    if mode not in STATE["modes"]:
        raise HTTPException(400, f"mode must be one of {STATE['modes']}")
    start = time.perf_counter()
    results = STATE[mode].search(q, k=k)
    ms = (time.perf_counter() - start) * 1000
    hits = []
    for r in results:
        doc_id, score = (r.doc_id, r.score) if hasattr(r, "doc_id") else r
        hits.append({"id": doc_id, "score": round(float(score), 4), "title": STATE["titles"].get(doc_id, ""),
                     "snippet": STATE["texts"][doc_id][:220]})
    return {"query": q, "mode": mode, "latency_ms": round(ms, 2), "hits": hits}


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    options = "".join(f'<option value="{m}">{m}</option>' for m in STATE["modes"])
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>scour</title>
<style>body{{font-family:system-ui,sans-serif;max-width:760px;margin:40px auto;padding:0 16px;color:#222}}
input,select,button{{font-size:16px;padding:8px}} input{{width:60%}} .hit{{margin:14px 0}} .t{{font-weight:600}}
.m{{color:#666;font-size:13px}} p{{margin:4px 0}}</style></head><body>
<h2>scour</h2><p class="m">A search engine built from scratch. Corpus: Cranfield aerodynamics abstracts.</p>
<form onsubmit="go();return false"><input id="q" placeholder="boundary layer transition" autofocus>
<select id="mode">{options}</select> <button>Search</button></form><div id="out"></div>
<script>const esc=s=>String(s).replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}})[c]);
async function go(){{const q=document.getElementById('q').value,m=document.getElementById('mode').value;
const r=await fetch(`/search?q=${{encodeURIComponent(q)}}&mode=${{encodeURIComponent(m)}}`);const d=await r.json();
if(!r.ok){{document.getElementById('out').innerHTML=`<p class="m">${{esc(d.detail||r.status)}}</p>`;return;}}
document.getElementById('out').innerHTML=`<p class="m">${{d.hits.length}} results in ${{esc(d.latency_ms)}} ms (${{esc(d.mode)}})</p>`+
d.hits.map((h,i)=>`<div class="hit"><p><span class="t">${{i+1}}. ${{esc(h.title||'doc '+h.id)}}</span> <span class="m">score ${{esc(h.score)}}</span></p><p>${{esc(h.snippet)}}...</p></div>`).join('');}}</script>
</body></html>"""
