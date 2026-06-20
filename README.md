# Retrieval Benchmark

Modular framework for benchmarking multimodal embedding models on biomedical document retrieval. Given a dataset of images and text, it encodes everything, stores embeddings, and evaluates retrieval quality via NDCG.

## What it does

- Encodes queries (images + captions) and documents (text) using multiple embedding models
- Saves embeddings to disk as safetensors + a manifest for reproducibility
- Evaluates retrieval via exhaustive similarity scoring + pytrec_eval (NDCG@1/5/10)
- Produces paper-ready comparison tables across models

## Models supported

| Model | Type | Modalities |
|---|---|---|
| `vidore/colSmol-500M` | Multi-vector (ColBERT/MaxSim) | text + image |
| `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` | Dense (512-dim) | text + image |
| `MahmoodLab/conch` | Dense (512-dim) | text + image |
| `flaviagiammarino/pubmed-clip-vit-base-patch32` | Dense (512-dim) | text + image |

## Project structure

```
src/
├── embedder/               # Model wrappers (ColSmol, BiomedCLIP, CONCH, PubMedCLIP)
│   └── _base.py            # BaseEmbedder ABC — batching, resource tracking, progress
├── embedding_factory.py    # Load model once, encode multiple sets, unload
├── evaluation/             # Evaluators (ColSmol MaxSim, Dense cosine)
│   └── base.py             # BaseEvaluator — rank all docs, NDCG via pytrec_eval
├── evaluation_factory.py   # evaluate(model_type, ..., cutoffs=[1,5,10])
└── core/
    ├── schema.py           # EmbeddingOutput, EmbeddingMetadata (Pydantic)
    ├── manifest.py         # Manifest — metadata saved alongside embeddings
    ├── artifact_store.py   # save_artifacts() — safetensors + manifest + qrels
    ├── retrieval_context.py# load_artifacts_hf() — load from HF Hub
    └── utils.py            # compare_ndcg(), print_embedding_output()
```

## Typical workflow

```python
from src.embedding_factory import EmbeddingFactory
from src.evaluation_factory import evaluate
from src.core import save_artifacts, compare_ndcg

# 1. Encode
factory = EmbeddingFactory(device="cuda")
embd_out = factory.generate_sets(
    model_name="vidore/colSmol-500M",
    sets={
        "query_image":   {"images": query_images},
        "query_caption": {"texts":  query_texts},
        "doc":           {"texts":  doc_texts},
    },
    batch_size=8,
)

# 2. Save
save_artifacts(
    model_name="vidore/colSmol-500M",
    model_type="colsmol",
    sets_out=embd_out,
    roles={"query_image": "query", "query_caption": "query", "doc": "doc"},
    query_ids=query_ids, doc_ids=doc_ids, qrels=qrels,
    artifacts_dir="artifacts", hf_repo_id="your/dataset",
)

# 3. Evaluate
result = evaluate(
    model_type="colsmol",
    query_embs=embd_out["query_image"].img_embd,
    doc_embs=embd_out["doc"].text_embd,
    query_ids=query_ids, doc_ids=doc_ids, qrels=qrels,
    cutoffs=[1, 5, 10],
)

# 4. Compare across models
compare_ndcg(eval_results)  # paper-ready table, all cutoffs
```

## Evaluation methodology

Follows BEIR / GQR (arXiv:2510.05038) standard:
- Exhaustive similarity matrix `[Nq, Nd]` — all docs ranked, no ANN approximation
- NDCG@k computed by pytrec_eval (trec_eval reference implementation)
- Primary metric: NDCG@5 (ViDoRe standard)
