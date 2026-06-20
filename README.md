# Retrieval Benchmark

Modular framework for benchmarking multimodal embedding models on biomedical document retrieval. Given a dataset of images and text, it encodes everything, stores embeddings, and evaluates retrieval quality via NDCG.

## What it does

- Encodes queries (images + captions) and documents (text) using multiple embedding models
- Saves embeddings to disk as safetensors + a manifest for reproducibility
- Evaluates retrieval via exhaustive similarity scoring + pytrec_eval (NDCG@1/5/10)
- Produces paper-ready comparison tables across models
- Supports test-time query refinement (GQR gradient-based, rank/score fusion)

## Models supported

| Model | Type | Modalities |
|---|---|---|
| `vidore/colSmol-500M` | Multi-vector (ColBERT/MaxSim) | text + image |
| `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` | Dense (512-dim) | text + image |
| `MahmoodLab/conch` | Dense (512-dim) | text + image |
| `flaviagiammarino/pubmed-clip-vit-base-patch32` | Dense (512-dim) | text + image |

## Test-time methods

| Method | Class | Description |
|---|---|---|
| GQR | `GQRMethod` | Gradient-based query refinement guided by a feedback retriever (KL loss) |
| Rank fusion | `AverageRankFusion` | Average per-document rank positions from two retrievers |
| Score fusion | `AverageScoreFusion` | Softmax-normalise then average scores from two retrievers |

## Project structure

```
src/
├── embedder/               # Model wrappers (ColSmol, BiomedCLIP, CONCH, PubMedCLIP)
│   └── _base.py            # BaseEmbedder ABC — batching, resource tracking, progress
├── embedding_factory.py    # EmbeddingFactory — load once, encode multiple sets, unload
├── evaluation/
│   ├── base.py             # BaseEvaluator — score → rank → NDCG via pytrec_eval
│   ├── scoring.py          # score_multi_vector, score_cosine, gqr_score_* functions
│   └── test_time/          # Test-time methods
│       ├── base.py         # BaseTestTimeMethod ABC
│       ├── gqr.py          # GQRMethod + plot_query_trajectory + plot_loss_curves
│       └── fusion.py       # AverageRankFusion, AverageScoreFusion
├── evaluation_factory.py   # evaluate(), apply_test_time_method(), register_evaluator()
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
from src.evaluation_factory import evaluate, apply_test_time_method
from src.evaluation.test_time import GQRMethod
from src.evaluation.scoring import gqr_score_multi_vector
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

# 3. Evaluate (primary + feedback)
primary  = evaluate("colsmol",    q_embs_col,   d_embs_col,   query_ids, doc_ids, qrels)
feedback = evaluate("biomedclip", q_embs_dense, d_embs_dense, query_ids, doc_ids, qrels)

# 4. Test-time refinement
gqr = GQRMethod(
    primary_evaluator=ColSmolEvaluator(device="cuda"),
    sim_func=gqr_score_multi_vector,
    lr=5e-3, n_steps=15,
)
gqr_result = apply_test_time_method(
    gqr,
    q_embs_col, d_embs_col,
    primary["similarity_matrix"],
    feedback["similarity_matrix"],
    query_ids, doc_ids, qrels,
    plot_trajectory=True,   # PCA plot of query path in embedding space
    plot_losses=True,       # per-query KL loss curves
    plots_save_path="results/figures",
    plot_query_idx=0,       # which query to visualise (default 0)
)

# 5. Compare across models
compare_ndcg(eval_results)  # paper-ready table, all cutoffs
```

## Evaluation methodology

Follows BEIR / GQR (arXiv:2510.05038) standard:
- Exhaustive similarity matrix `[Nq, Nd]` — all docs ranked, no ANN approximation
- NDCG@k computed by pytrec_eval (trec_eval reference implementation)
- Primary metric: NDCG@5 (ViDoRe standard)

---

See [CONTRIBUTIONS.md](CONTRIBUTIONS.md) for how to add new embedders, evaluators, and test-time methods.
