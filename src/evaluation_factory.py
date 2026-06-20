from __future__ import annotations

from typing import Any

from src.evaluation.colsmol import ColSmolEvaluator
from src.evaluation.dense import DenseEvaluator
from src.evaluation.test_time.base import BaseTestTimeMethod

_EVALUATOR_MAP = {
    "colsmol": ColSmolEvaluator,
    "biomedclip": DenseEvaluator,
    "conch": DenseEvaluator,
    "pubmedclip": DenseEvaluator,
}


def evaluate(
    model_type: str,
    query_embs: Any,
    doc_embs: Any,
    query_ids: list[str],
    doc_ids: list[str],
    qrels: dict[str, dict[str, int]],
    device: str = "cpu",
    cutoffs: list[int] = [1, 5, 10],
) -> dict:
    """Run retrieval evaluation for a given model type.

    Args:
        model_type: one of "colsmol", "biomedclip", "conch", "pubmedclip"
        query_embs: embeddings for queries
        doc_embs:   embeddings for docs
        query_ids:  ordered query ID strings
        doc_ids:    ordered doc ID strings
        qrels:      {query_id: {doc_id: relevance_int}}
        device:     torch device string
        cutoffs:    list of k values for NDCG@k (e.g. [1, 5, 10])

    Returns:
        dict with mean_ndcg {k: float}, ndcg {k: np.ndarray}, retrieval_results, similarity_matrix
    """
    if model_type not in _EVALUATOR_MAP:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose from: {list(_EVALUATOR_MAP)}")
    cls = _EVALUATOR_MAP[model_type]
    return cls(device=device, cutoffs=cutoffs).run(query_embs, doc_embs, query_ids, doc_ids, qrels)


def apply_test_time_method(
    method: BaseTestTimeMethod,
    query_emb_main: Any,
    doc_emb_main: Any,
    sim_main: Any,
    sim_feedback: Any,
    query_ids: list[str],
    doc_ids: list[str],
    qrels: dict[str, dict[str, int]],
) -> dict:
    """Apply any test-time method given two similarity matrices.

    Args:
        method:          An instance of a ``BaseTestTimeMethod`` subclass.
        query_emb_main:  Primary query embeddings (needed by GQR; pass None for fusion).
        doc_emb_main:    Primary doc embeddings (needed by GQR; pass None for fusion).
        sim_main:        [Nq, Nd] similarity matrix from the primary model.
        sim_feedback:    [Nq, Nd] similarity matrix from the feedback model.
        query_ids:       Ordered query ID strings.
        doc_ids:         Ordered doc ID strings.
        qrels:           {query_id: {doc_id: relevance_int}}

    Returns:
        Result dict with the same schema as ``evaluate()``.
    """
    return method.apply(
        query_emb_main=query_emb_main,
        doc_emb_main=doc_emb_main,
        sim_main=sim_main,
        sim_feedback=sim_feedback,
        query_ids=query_ids,
        doc_ids=doc_ids,
        qrels=qrels,
    )


def register_evaluator(model_type: str, cls) -> None:
    """Register a custom evaluator class for a model type."""
    _EVALUATOR_MAP[model_type] = cls
