from __future__ import annotations

import torch

from src.evaluation.base import BaseEvaluator
from src.evaluation.scoring import score_cosine


class DenseEvaluator(BaseEvaluator):
    """Cosine-similarity evaluator for single-vector (dense) embeddings."""

    def score(
        self,
        query_embs: torch.Tensor | list[torch.Tensor],
        doc_embs: torch.Tensor | list[torch.Tensor],
    ) -> torch.Tensor:
        if isinstance(query_embs, list):
            query_embs = torch.stack(query_embs)
        if isinstance(doc_embs, list):
            doc_embs = torch.stack(doc_embs)
        return score_cosine(query_embs, doc_embs, device=self.device)
