from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pytrec_eval  # type: ignore[import-untyped]
import torch


class BaseEvaluator(ABC):
    """Abstract retrieval evaluator.

    Subclasses implement `score()` only. Everything else (retrieve, ndcg, run)
    is shared. To add a new evaluator:

        class MyEvaluator(BaseEvaluator):
            def score(self, query_embs, doc_embs) -> torch.Tensor:
                ...  # return [Nq, Nd] float32 similarity matrix
    """

    def __init__(self, device: str = "cpu", k: int = 10, cutoff: int = 5):
        self.device = device
        self.k = k
        self.cutoff = cutoff

    @abstractmethod
    def score(self, query_embs: Any, doc_embs: Any) -> torch.Tensor:
        """Compute and return [Nq, Nd] float32 similarity matrix."""
        ...

    def retrieve(
        self,
        similarity: torch.Tensor,
        query_ids: list[str],
        doc_ids: list[str],
    ) -> dict[str, dict[str, float]]:
        """Top-k from similarity matrix, keyed by actual IDs (not positions)."""
        k_eff = min(self.k, similarity.shape[1])
        top_vals, top_idx = torch.topk(similarity, k=k_eff, dim=-1)
        results: dict[str, dict[str, float]] = {}
        for i, (val_row, idx_row) in enumerate(
            zip(top_vals.cpu().tolist(), top_idx.cpu().tolist())
        ):
            results[query_ids[i]] = {doc_ids[j]: float(v) for j, v in zip(idx_row, val_row)}
        return results

    def eval_ndcg(
        self,
        qrels: dict[str, dict[str, int]],
        retrieval_results: dict[str, dict[str, float]],
    ) -> np.ndarray:
        """NDCG@cutoff via pytrec_eval. Returns per-query array."""
        metric_key = f"ndcg_cut_{self.cutoff}"
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {f"ndcg_cut.{self.cutoff}"})
        scores = evaluator.evaluate(retrieval_results)
        return np.array([v[metric_key] for v in scores.values()])

    def run(
        self,
        query_embs: Any,
        doc_embs: Any,
        query_ids: list[str],
        doc_ids: list[str],
        qrels: dict[str, dict[str, int]],
    ) -> dict:
        """Score → retrieve → evaluate. Returns result dict."""
        assert len(query_embs) == len(query_ids), (
            f"query_embs {len(query_embs)} != query_ids {len(query_ids)}"
        )
        assert len(doc_embs) == len(doc_ids), (
            f"doc_embs {len(doc_embs)} != doc_ids {len(doc_ids)}"
        )

        sim = self.score(query_embs, doc_embs)
        ranked = self.retrieve(sim, query_ids, doc_ids)
        ndcg = self.eval_ndcg(qrels, ranked)

        return {
            "evaluator": type(self).__name__,
            "mean_ndcg": float(ndcg.mean()),
            "ndcg": ndcg,
            "retrieval_results": ranked,
            "similarity_matrix": sim,
        }
