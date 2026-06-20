from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseTestTimeMethod(ABC):
    """Abstract base for test-time retrieval methods.

    ``apply()`` mirrors the GQR paper interface: raw primary embeddings are
    passed explicitly alongside the two full evaluator result dicts.

    - Embedding-level methods (GQR) use ``query_emb_main``, ``doc_emb_main``,
      and ``primary_result["similarity_matrix"]`` / ``feedback_result["similarity_matrix"]``.
    - Score-level methods (fusion) ignore the raw embeddings and use
      ``primary_result["retrieval_results"]`` / ``feedback_result["retrieval_results"]``.
    """

    @abstractmethod
    def apply(
        self,
        query_emb_main: Any,
        doc_emb_main: Any,
        primary_result: dict,
        feedback_result: dict,
        query_ids: list[str],
        doc_ids: list[str],
        qrels: dict[str, dict[str, int]],
    ) -> dict:
        """Run the test-time method and return an evaluation result dict.

        Args:
            query_emb_main:   Primary query embeddings (list[Tensor] or Tensor).
            doc_emb_main:     Primary doc embeddings (list[Tensor] or Tensor).
            primary_result:   Output of ``BaseEvaluator.run()`` for the primary model.
                              Keys used: ``similarity_matrix``, ``retrieval_results``.
            feedback_result:  Output of ``BaseEvaluator.run()`` for the feedback model.
                              Keys used: ``similarity_matrix``, ``retrieval_results``.
            query_ids:        Ordered query ID strings.
            doc_ids:          Ordered doc ID strings.
            qrels:            {query_id: {doc_id: relevance_int}}

        Returns:
            Result dict matching the schema of ``BaseEvaluator.run()``.
        """
        ...
