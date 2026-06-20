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

    def __init__(self, device: str = "cpu", cutoffs: list[int] = [1, 5, 10]):
        self.device = device
        self.cutoffs = cutoffs

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
        """Rank all docs by similarity score, keyed by actual IDs."""
        sorted_idx = torch.argsort(similarity, dim=-1, descending=True)
        sorted_vals = torch.gather(similarity, 1, sorted_idx)
        results: dict[str, dict[str, float]] = {}
        for i, (val_row, idx_row) in enumerate(
            zip(sorted_vals.cpu().tolist(), sorted_idx.cpu().tolist())
        ):
            results[query_ids[i]] = {doc_ids[j]: float(v) for j, v in zip(idx_row, val_row)}
        return results

    def eval_ndcg(
        self,
        qrels: dict[str, dict[str, int]],
        retrieval_results: dict[str, dict[str, float]],
    ) -> dict[int, np.ndarray]:
        """NDCG@k for all cutoffs in one pytrec_eval pass.

        Returns {cutoff: per_query_ndcg_array} aligned to qrels query order.
        """
        measures = {f"ndcg_cut.{k}" for k in self.cutoffs}
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, measures)
        scores = evaluator.evaluate(retrieval_results)
        query_order = [qid for qid in qrels if qid in scores]
        return {
            k: np.array([scores[qid][f"ndcg_cut_{k}"] for qid in query_order])
            for k in self.cutoffs
        }

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
            "mean_ndcg": {k: float(v.mean()) for k, v in ndcg.items()},
            "ndcg":      ndcg,
            "retrieval_results": ranked,
            "similarity_matrix": sim,
        }
