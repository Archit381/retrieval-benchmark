from __future__ import annotations

from typing import Any

import numpy as np
import pytrec_eval
import torch

from src.evaluation.test_time.base import BaseTestTimeMethod


def _eval_ndcg(
    qrels: dict[str, dict[str, int]],
    retrieval_results: dict[str, dict[str, float]],
    cutoffs: list[int],
) -> dict[int, np.ndarray]:
    measures = {f"ndcg_cut.{k}" for k in cutoffs}
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, measures)
    scores = evaluator.evaluate(retrieval_results)
    query_order = [qid for qid in qrels if qid in scores]
    return {
        k: np.array([scores[qid][f"ndcg_cut_{k}"] for qid in query_order])
        for k in cutoffs
    }


def _sim_to_retrieval(
    sim: torch.Tensor,
    query_ids: list[str],
    doc_ids: list[str],
) -> dict[str, dict[str, float]]:
    sorted_idx  = torch.argsort(sim, dim=-1, descending=True)
    sorted_vals = torch.gather(sim, 1, sorted_idx)
    return {
        query_ids[i]: {doc_ids[j]: float(v) for j, v in zip(idx_row, val_row)}
        for i, (val_row, idx_row) in enumerate(
            zip(sorted_vals.cpu().tolist(), sorted_idx.cpu().tolist())
        )
    }


def _average_rank_fusion(
    results_a: dict[str, dict[str, float]],
    results_b: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    fused: dict[str, dict[str, float]] = {}
    for qid, ranking_a in results_a.items():
        ranking_b = results_b.get(qid, {})
        ranks_a = {did: rank for rank, did in enumerate(ranking_a)}
        ranks_b = {did: rank for rank, did in enumerate(ranking_b)}
        k = len(ranking_a)
        all_docs = set(ranking_a) | set(ranking_b)
        sorted_docs = sorted(
            all_docs,
            key=lambda d: (ranks_a.get(d, k) + ranks_b.get(d, k)) / 2,
        )
        fused[qid] = {d: 1 - (i / k) for i, d in enumerate(sorted_docs)}
    return fused


def _average_score_fusion(
    results_a: dict[str, dict[str, float]],
    results_b: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    fused: dict[str, dict[str, float]] = {}
    for qid in results_a:
        scores_a = results_a.get(qid, {})
        scores_b = results_b.get(qid, {})
        dids_a, raw_a = zip(*scores_a.items())
        dids_b, raw_b = zip(*scores_b.items())
        probs_a = dict(zip(dids_a, torch.softmax(torch.tensor(raw_a), dim=0).tolist()))
        probs_b = dict(zip(dids_b, torch.softmax(torch.tensor(raw_b), dim=0).tolist()))
        all_docs = set(scores_a) | set(scores_b)
        fused_scores = {
            d: (probs_a.get(d, 0.0) + probs_b.get(d, 0.0)) / 2
            for d in all_docs
        }
        fused[qid] = dict(sorted(fused_scores.items(), key=lambda kv: kv[1], reverse=True))
    return fused


class AverageRankFusion(BaseTestTimeMethod):
    """Hybrid retrieval by averaging per-document ranks from two retrievers.

    No model-specific scoring — works with any pair of evaluator results.
    """

    def __init__(self, cutoffs: list[int] = [1, 5, 10]) -> None:
        self.cutoffs = cutoffs

    def apply(
        self,
        _query_emb_main: Any,
        _doc_emb_main: Any,
        sim_main: Any,
        sim_feedback: Any,
        query_ids: list[str],
        doc_ids: list[str],
        qrels: dict[str, dict[str, int]],
    ) -> dict:
        ranked = _average_rank_fusion(
            _sim_to_retrieval(sim_main,     query_ids, doc_ids),
            _sim_to_retrieval(sim_feedback, query_ids, doc_ids),
        )
        ndcg = _eval_ndcg(qrels, ranked, self.cutoffs)
        return {
            "method":            "AverageRankFusion",
            "mean_ndcg":         {c: float(v.mean()) for c, v in ndcg.items()},
            "ndcg":              ndcg,
            "retrieval_results": ranked,
        }


class AverageScoreFusion(BaseTestTimeMethod):
    """Hybrid retrieval by softmax-normalising then averaging scores from two retrievers.

    No model-specific scoring — works with any pair of evaluator results.
    """

    def __init__(self, cutoffs: list[int] = [1, 5, 10]) -> None:
        self.cutoffs = cutoffs

    def apply(
        self,
        _query_emb_main: Any,
        _doc_emb_main: Any,
        sim_main: Any,
        sim_feedback: Any,
        query_ids: list[str],
        doc_ids: list[str],
        qrels: dict[str, dict[str, int]],
    ) -> dict:
        ranked = _average_score_fusion(
            _sim_to_retrieval(sim_main,     query_ids, doc_ids),
            _sim_to_retrieval(sim_feedback, query_ids, doc_ids),
        )
        ndcg = _eval_ndcg(qrels, ranked, self.cutoffs)
        return {
            "method":            "AverageScoreFusion",
            "mean_ndcg":         {c: float(v.mean()) for c, v in ndcg.items()},
            "ndcg":              ndcg,
            "retrieval_results": ranked,
        }
