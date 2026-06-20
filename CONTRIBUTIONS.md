# Contributing — Extension Guide

This document covers the three main extension points: adding a new **embedder**, a new **static evaluator**, and a new **test-time method**.

---

## Adding a new embedder

### 1. Implement the class

Create `src/embedder/my_model.py`. Subclass `BaseEmbedder` and implement three methods:

```python
from src.embedder._base import BaseEmbedder
import torch

class MyEmbedder(BaseEmbedder):
    model_type     = "mymodel"  # short key, shows up in logs and metadata
    is_multivector = False      # set True for ColBERT-style token embeddings

    def _load(self) -> None:
        # Load weights onto self.device; assign to self._model / self._processor
        self._model     = MyModel.from_pretrained(self.model_name).to(self.device)
        self._processor = MyProcessor.from_pretrained(self.model_name)

    @torch.no_grad()
    def _encode_text(self, texts: list[str]):
        # Called with one batch at a time — slicing is handled by the base class
        # Dense:        return Tensor [B, D]
        # Multi-vector: return list[Tensor[T, D]] or Tensor [B, T, D]
        inputs = self._processor(texts, return_tensors="pt").to(self.device)
        return self._model(**inputs).last_hidden_state[:, 0]  # CLS token example

    @torch.no_grad()
    def _encode_image(self, images: list):
        inputs = self._processor(images=images, return_tensors="pt").to(self.device)
        return self._model.get_image_features(**inputs)
```

You never call `_encode_text` / `_encode_image` directly. The public API is `encode()` (single batch) and `encode_batch()` (auto-batching with progress bars, VRAM tracking, CPU offload). Both return an `EmbeddingOutput(text_embd, img_embd, metadata)`.

If your model is **text-only or image-only**, just `raise NotImplementedError` in the unused method — the factory will skip that modality gracefully when running sweeps.

### 2. Export it

Add to `src/embedder/__init__.py`:

```python
from src.embedder.my_model import MyEmbedder
```

### 3. Register it

Add an entry to `EmbeddingFactory.SUPPORT_MAP` in `src/embedding_factory.py`:

```python
SUPPORT_MAP = {
    ...
    "org/my-checkpoint": (MyEmbedder, {}, {"text", "image"}),
    #                      ^class   ^extra kwargs  ^supported modalities
}
```

The modality set controls what gets silently skipped in multi-model sweeps. Use `{"text"}` for text-only, `{"image"}` for image-only. Any kwargs in the second slot are forwarded to the constructor.

You can also register at runtime without editing the source:

```python
EmbeddingFactory.register("org/my-checkpoint", MyEmbedder, modalities={"text", "image"})
```

---

## Adding a new evaluator

Evaluation is split into **static evaluators** (score fixed embeddings once) and **test-time methods** (adapt at inference). Pick the right tier.

### Static evaluator

Subclass `BaseEvaluator` and implement `score()`. Ranking, NDCG via pytrec_eval, and the `run()` entrypoint are all inherited — you only define how similarity is computed.

```python
from src.evaluation.base import BaseEvaluator
import torch

class MyEvaluator(BaseEvaluator):
    def score(self, query_embs, doc_embs) -> torch.Tensor:
        # Must return float32 [Nq, Nd] on any device
        # query_embs / doc_embs shape depends on what your embedder produces
        ...
```

Register it so it's accessible via the string API in `src/evaluation_factory.py`:

```python
_EVALUATOR_MAP["mymodel"] = MyEvaluator
```

Or at runtime:

```python
from src.evaluation_factory import register_evaluator
register_evaluator("mymodel", MyEvaluator)
```

Then call it like any built-in:

```python
result = evaluate("mymodel", query_embs, doc_embs, query_ids, doc_ids, qrels)
# result keys: mean_ndcg, ndcg, retrieval_results, similarity_matrix
```

---

## Adding a test-time method

Test-time methods sit between embedding and evaluation: they receive two similarity matrices (primary + feedback model) and return a result dict. They live in `src/evaluation/test_time/`.

### 1. Implement the class

Subclass `BaseTestTimeMethod`. The positional signature of `apply()` is fixed — the first seven arguments must match exactly so `apply_test_time_method()` can call any method uniformly.

```python
from src.evaluation.test_time.base import BaseTestTimeMethod
from typing import Any

class MyMethod(BaseTestTimeMethod):
    def __init__(self, cutoffs=(1, 5, 10), my_param=0.5):
        self.cutoffs  = list(cutoffs)
        self.my_param = my_param

    def apply(
        self,
        query_emb_main: Any,            # primary query embeddings
        doc_emb_main: Any,              # primary doc embeddings
        sim_main: Any,                  # [Nq, Nd] — primary model scores
        sim_feedback: Any,              # [Nq, Nd] — feedback model scores
        query_ids: list[str],
        doc_ids: list[str],
        qrels: dict[str, dict[str, int]],
        **kwargs,                       # any extra params you want to expose
    ) -> dict:
        ...
        return {
            "method":            "MyMethod",
            "mean_ndcg":         {k: float(v.mean()) for k, v in ndcg.items()},
            "ndcg":              ndcg,            # {cutoff: np.ndarray}
            "retrieval_results": ranked,          # {qid: {did: score}}
        }
```

**Two flavours to consider:**

- **Score-level** (like `AverageRankFusion`): ignore `query_emb_main` / `doc_emb_main` — prefix them with `_` to signal they're unused. Fuse or re-rank using `sim_main` / `sim_feedback` directly.
- **Embedding-level** (like `GQRMethod`): modify `query_emb_main` via gradient steps or other transforms, then re-score against `doc_emb_main`. Hold a reference to a `BaseEvaluator` for re-scoring.

Extra parameters (beyond the fixed seven) are passed via `**kwargs`. They flow through `apply_test_time_method` automatically since it forwards `**kwargs` to `method.apply()`.

### 2. Export it (optional)

If you want `from src.evaluation import MyMethod` to work, add it to `src/evaluation/test_time/__init__.py` and `src/evaluation/__init__.py`.

### 3. Use it

No registration step needed — pass an instance directly:

```python
from src.evaluation_factory import apply_test_time_method

result = apply_test_time_method(
    MyMethod(my_param=0.7),
    query_emb_main, doc_emb_main,
    primary["similarity_matrix"],
    feedback["similarity_matrix"],
    query_ids, doc_ids, qrels,
    some_extra_kwarg=True,   # forwarded to apply() via **kwargs
)
```
