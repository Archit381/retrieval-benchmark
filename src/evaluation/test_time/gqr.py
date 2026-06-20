from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.evaluation.base import BaseEvaluator
from src.evaluation.test_time.base import BaseTestTimeMethod


def guided_query_refinement(
    query_emb_main: Any,
    doc_emb_main: Any,
    sim_main: torch.Tensor,
    sim_feedback: torch.Tensor,
    sim_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    lr: float,
    n_steps: int,
    k: int = 10,
    device: str = "cpu",
) -> Any:
    """Refine query embeddings at test time using guidance from a feedback retriever.

    Works with any primary model (multi-vector or dense) as long as ``sim_func``
    and the embedding format are consistent.

    Args:
        query_emb_main:  Primary query embeddings.
                         - Multi-vector: list[Tensor] each [Tq_i, D]
                         - Dense:        Tensor [Nq, D]
        doc_emb_main:    Primary doc embeddings.
                         - Multi-vector: Tensor [Nd, Td, D] (padded)
                         - Dense:        Tensor [Nd, D]
        sim_main:        [Nq, Nd] similarity matrix from the primary model.
        sim_feedback:    [Nq, Nd] similarity matrix from the feedback model.
        sim_func:        Differentiable per-query scorer:
                         ``(q: Tensor[1, ...], docs: Tensor[Nd, ...]) -> Tensor[Nd]``
                         Use ``gqr_score_multi_vector`` for ColSmol,
                         ``gqr_score_cosine`` for dense models.
        lr:              Adam learning rate for the optimization loop.
        n_steps:         Number of gradient steps per query.
        k:               Top-k candidates to consider from each retriever.
        device:          Torch device string.

    Returns:
        Refined query embeddings in the same format as ``query_emb_main``.
    """
    is_list = isinstance(query_emb_main, list)
    n_queries = len(query_emb_main)

    sim_main     = sim_main.to(device)
    sim_feedback = sim_feedback.to(device)

    _, top_idx_main     = torch.topk(sim_main,     k=k, dim=-1)
    _, top_idx_feedback = torch.topk(sim_feedback, k=k, dim=-1)

    if is_list:
        qs = [query_emb_main[i].clone().detach().unsqueeze(0).to(device).requires_grad_(True)
              for i in range(n_queries)]
    else:
        qs = [query_emb_main[i:i+1].clone().detach().to(device).requires_grad_(True)
              for i in range(n_queries)]

    doc_emb_device = doc_emb_main if isinstance(doc_emb_main, torch.Tensor) \
        else torch.nn.utils.rnn.pad_sequence(
            [d.to(device) for d in doc_emb_main], batch_first=True
        )

    query_bar = tqdm(range(n_queries), desc="GQR queries", unit="q")
    for i in query_bar:
        opt = torch.optim.Adam([qs[i]], lr=lr)

        idx_main     = top_idx_main[i]
        idx_feedback = top_idx_feedback[i]
        u = torch.unique(torch.cat([idx_main, idx_feedback], dim=0))

        docs_u = doc_emb_device.index_select(0, u.to(device))

        d_main     = sim_main[i].index_select(0, u)
        d_feedback = sim_feedback[i].index_select(0, u)
        mixture = torch.softmax((d_main + d_feedback) / 2, dim=-1).detach().to(device)

        step_bar = tqdm(range(n_steps), desc=f"  q{i} steps", unit="step", leave=False)
        for step in step_bar:
            pred = sim_func(qs[i], docs_u).to(device)
            loss = F.kl_div(
                torch.log_softmax(pred, dim=-1),
                mixture,
                reduction="batchmean",
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step_bar.set_postfix(loss=f"{loss.item():.4f}")

    if is_list:
        return [qs[i].detach().squeeze(0).cpu() for i in range(n_queries)]
    else:
        return torch.cat([qs[i].detach().cpu() for i in range(n_queries)], dim=0)


class GQRMethod(BaseTestTimeMethod):
    """Guided Query Refinement (GQR) test-time method.

    Refines the primary model's query embeddings using gradient-based
    optimization guided by a feedback retriever's similarity scores, then
    re-evaluates with the primary model.

    The ``primary_evaluator`` is used only for ``.score()``, ``.retrieve()``,
    and ``.eval_ndcg()`` — it can be any ``BaseEvaluator`` subclass.

    The ``sim_func`` must match the primary model's embedding format:
    - ColSmol primary  → ``gqr_score_multi_vector``
    - Dense primary    → ``gqr_score_cosine``

    Example::

        from src.evaluation import ColSmolEvaluator
        from src.evaluation.scoring import gqr_score_multi_vector
        from src.evaluation.test_time import GQRMethod

        primary_eval = ColSmolEvaluator(device="cuda")
        gqr = GQRMethod(primary_eval, sim_func=gqr_score_multi_vector, lr=5e-3, n_steps=15)
        result = gqr.apply(primary_result, feedback_result,
                           query_embs_colsmol, doc_embs_colsmol,
                           query_ids, doc_ids, qrels)
    """

    def __init__(
        self,
        primary_evaluator: BaseEvaluator,
        sim_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        lr: float,
        n_steps: int,
        k: int = 10,
    ) -> None:
        self.primary_evaluator = primary_evaluator
        self.sim_func = sim_func
        self.lr = lr
        self.n_steps = n_steps
        self.k = k

    def apply(
        self,
        query_emb_main: Any,
        doc_emb_main: Any,
        sim_main: Any,
        sim_feedback: Any,
        query_ids: list[str],
        doc_ids: list[str],
        qrels: dict[str, dict[str, int]],
    ) -> dict:
        refined = guided_query_refinement(
            query_emb_main=query_emb_main,
            doc_emb_main=doc_emb_main,
            sim_main=sim_main,
            sim_feedback=sim_feedback,
            sim_func=self.sim_func,
            lr=self.lr,
            n_steps=self.n_steps,
            k=self.k,
            device=self.primary_evaluator.device,
        )

        new_sim = self.primary_evaluator.score(refined, doc_emb_main)
        ranked  = self.primary_evaluator.retrieve(new_sim, query_ids, doc_ids)
        ndcg    = self.primary_evaluator.eval_ndcg(qrels, ranked)

        return {
            "method":             "GQR",
            "lr":                 self.lr,
            "n_steps":            self.n_steps,
            "k":                  self.k,
            "evaluator":          type(self.primary_evaluator).__name__,
            "mean_ndcg":          {c: float(v.mean()) for c, v in ndcg.items()},
            "ndcg":               ndcg,
            "retrieval_results":  ranked,
            "similarity_matrix":  new_sim,
            "refined_query_embs": refined,
        }
