from __future__ import annotations

import torch

from src.evaluation.base import BaseEvaluator
from src.evaluation.scoring import score_multi_vector


class ColSmolEvaluator(BaseEvaluator):
    """Late-interaction (MaxSim / ColBERT) evaluator for ColSmol multi-vector embeddings."""

    def score(
        self,
        query_embs: list[torch.Tensor],
        doc_embs: list[torch.Tensor],
    ) -> torch.Tensor:
        return score_multi_vector(query_embs, doc_embs, device=self.device)
