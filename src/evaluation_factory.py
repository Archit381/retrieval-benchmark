from __future__ import annotations

from typing import Any

from src.evaluation.colsmol import ColSmolEvaluator
from src.evaluation.dense import DenseEvaluator

_EVALUATOR_MAP = {
    "colsmol": ColSmolEvaluator,
    "biomedclip": DenseEvaluator,
    "conch": DenseEvaluator,
}


def evaluate(
    model_type: str,
    query_embs: Any,
    doc_embs: Any,
    query_ids: list[str],
    doc_ids: list[str],
    qrels: dict[str, dict[str, int]],
    device: str = "cpu",
    k: int = 10,
    cutoff: int = 5,
) -> dict:
    """Run retrieval evaluation for a given model type.

    Args:
        model_type: one of "colsmol", "biomedclip", "conch"
        query_embs: embeddings for queries
        doc_embs:   embeddings for docs
        query_ids:  ordered query ID strings
        doc_ids:    ordered doc ID strings
        qrels:      {query_id: {doc_id: relevance_int}}
        device:     torch device string
        k:          top-k docs to retrieve
        cutoff:     NDCG cutoff

    Returns:
        dict with mean_ndcg, ndcg, retrieval_results, similarity_matrix
    """
    if model_type not in _EVALUATOR_MAP:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose from: {list(_EVALUATOR_MAP)}")
    cls = _EVALUATOR_MAP[model_type]
    return cls(device=device, k=k, cutoff=cutoff).run(query_embs, doc_embs, query_ids, doc_ids, qrels)


def register_evaluator(model_type: str, cls) -> None:
    """Register a custom evaluator class for a model type."""
    _EVALUATOR_MAP[model_type] = cls
