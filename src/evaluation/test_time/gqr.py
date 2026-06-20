from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn.functional as F
from src.evaluation.base import BaseEvaluator
from src.evaluation.test_time.base import BaseTestTimeMethod



def _to_dense_numpy(q: torch.Tensor) -> Any:
    """Mean-pool multi-vector [1,T,D] or pass through dense [1,D], then L2-normalize."""
    t = q.detach().cpu().float()
    if t.dim() == 3:
        t = t.mean(dim=1)
    t = t.squeeze(0)
    norm = t.norm()
    return (t / norm.clamp(min=1e-8)).numpy()


def plot_query_trajectory(
    doc_emb: Any,
    trajectory: list,
    feedback_idx: list[int],
    pos_idx: set[int],
    title: str = "GQR Query Trajectory",
    save_path: str | None = None,
) -> None:
    """PCA 2-D plot of a GQR query trajectory overlaid on the document manifold.

    Args:
        doc_emb:      Dense numpy array [Nd, D] of all doc embeddings.
        trajectory:   List of [D] numpy arrays — initial query + one per recorded step.
        feedback_idx: Indices into doc_emb that form the feedback set D_K.
        pos_idx:      Indices into doc_emb that are ground-truth positives.
        title:        Figure title.
        save_path:    If given, save PNG to this path (parent dirs created automatically).
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    path = np.stack(trajectory)
    pca  = PCA(n_components=2).fit(doc_emb)
    D_proj    = pca.transform(doc_emb)
    path_proj = pca.transform(path)

    fb_set   = set(feedback_idx)
    neg_mask = np.array([i not in pos_idx and i not in fb_set for i in range(len(doc_emb))])
    pos_list = sorted(pos_idx)

    fig: Any = plt.figure(figsize=(6.5, 5.5))
    ax = fig.add_subplot(111)
    ax.scatter(D_proj[neg_mask, 0], D_proj[neg_mask, 1],
               s=12, c="#f0c4de", alpha=0.50, zorder=1, label="Negative docs")
    ax.scatter(D_proj[pos_list, 0], D_proj[pos_list, 1],
               s=70, c="#66bb6a", zorder=4, label="Positive docs")
    ax.scatter(D_proj[feedback_idx, 0], D_proj[feedback_idx, 1], s=55,
               facecolors="none", edgecolors="#444", linewidths=1.0,
               zorder=3, label=f"Feedback set $D_K$ ({len(feedback_idx)})")
    ax.plot(path_proj[:, 0], path_proj[:, 1],
            "-o", ms=4, c="#4c78a8", lw=1.8, zorder=5, label="GQR path")
    ax.scatter(*path_proj[0],  marker="s", s=100, c="#4c78a8",
               edgecolors="white", lw=0.6, zorder=6, label="$q_0$ (initial)")
    ax.scatter(*path_proj[-1], marker="*", s=200, c="#1a237e",
               edgecolors="white", lw=0.5, zorder=6, label="$q^*$ (refined)")
    ax.annotate("", xy=path_proj[-1], xytext=path_proj[0],
                arrowprops=dict(arrowstyle="-|>", color="#4c78a8", lw=1.5), zorder=7)

    ax.set_xlabel("PC 1", fontsize=9)
    ax.set_ylabel("PC 2", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()  # type: ignore[union-attr]

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")  # type: ignore[union-attr]
        print(f"[GQR] trajectory plot saved → {save_path}")
    plt.show()


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
    trajectory_out: list | None = None,
    trajectory_query_idx: int = 0,
) -> Any:
    """Refine query embeddings at test time using guidance from a feedback retriever.

    Works with any primary model (multi-vector or dense) as long as ``sim_func``
    and the embedding format are consistent.

    Args:
        query_emb_main:       Primary query embeddings.
                              - Multi-vector: list[Tensor] each [Tq_i, D]
                              - Dense:        Tensor [Nq, D]
        doc_emb_main:         Primary doc embeddings.
                              - Multi-vector: Tensor [Nd, Td, D] (padded)
                              - Dense:        Tensor [Nd, D]
        sim_main:             [Nq, Nd] similarity matrix from the primary model.
        sim_feedback:         [Nq, Nd] similarity matrix from the feedback model.
        sim_func:             Differentiable per-query scorer:
                              ``(q: Tensor[1, ...], docs: Tensor[Nd, ...]) -> Tensor[Nd]``
                              Use ``gqr_score_multi_vector`` for ColSmol,
                              ``gqr_score_cosine`` for dense models.
        lr:                   Adam learning rate for the optimization loop.
        n_steps:              Number of gradient steps per query.
        k:                    Top-k candidates to consider from each retriever.
        device:               Torch device string.
        trajectory_out:       If a list is passed, the query state for
                              ``trajectory_query_idx`` is appended at the start
                              and after every ``max(1, n_steps//10)`` steps.
                              Pass an empty list from ``GQRMethod.apply`` to
                              collect the path for plotting.
        trajectory_query_idx: Which query index to record (default 0).

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

    record_every = max(1, n_steps // 10)

    for i in range(n_queries):
        print(f"[GQR] query {i+1}/{n_queries}")
        opt = torch.optim.Adam([qs[i]], lr=lr)

        idx_main     = top_idx_main[i]
        idx_feedback = top_idx_feedback[i]
        u = torch.unique(torch.cat([idx_main, idx_feedback], dim=0))

        docs_u = doc_emb_device.index_select(0, u.to(device))

        d_main     = sim_main[i].index_select(0, u)
        d_feedback = sim_feedback[i].index_select(0, u)
        mixture = torch.softmax((d_main + d_feedback) / 2, dim=-1).detach().to(device)

        record = (trajectory_out is not None and i == trajectory_query_idx)
        if record and trajectory_out is not None:
            trajectory_out.append(_to_dense_numpy(qs[i]))

        for step in range(n_steps):
            pred = sim_func(qs[i], docs_u).to(device)
            loss = F.kl_div(
                torch.log_softmax(pred, dim=-1),
                mixture,
                reduction="batchmean",
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            print(f"  step {step+1}/{n_steps}  loss={loss.item():.6f}")

            if record and trajectory_out is not None and (step + 1) % record_every == 0:
                trajectory_out.append(_to_dense_numpy(qs[i]))

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
                           query_ids, doc_ids, qrels,
                           plot_trajectory=True,
                           trajectory_save_path="results/figures/gqr_trajectory.png")
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
        plot_trajectory: bool = False,
        plot_query_idx: int = 0,
        trajectory_save_path: str | None = None,
    ) -> dict:
        trajectory: list | None = [] if plot_trajectory else None

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
            trajectory_out=trajectory,
            trajectory_query_idx=plot_query_idx,
        )

        if plot_trajectory and trajectory:
            self._emit_trajectory_plot(
                doc_emb_main, sim_main, sim_feedback,
                query_ids, doc_ids, qrels,
                trajectory, plot_query_idx, trajectory_save_path,
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

    def _emit_trajectory_plot(
        self,
        doc_emb_main: Any,
        sim_main: Any,
        sim_feedback: Any,
        query_ids: list[str],
        doc_ids: list[str],
        qrels: dict[str, dict[str, int]],
        trajectory: list,
        query_idx: int,
        save_path: str | None,
    ) -> None:
        import numpy as np

        # Dense numpy [Nd, D] for PCA (mean-pool multi-vector)
        if isinstance(doc_emb_main, torch.Tensor):
            d = doc_emb_main.float().cpu()
            doc_np = (d.mean(dim=1) if d.dim() == 3 else d).numpy()
        else:
            doc_np = np.stack([t.float().mean(0).cpu().numpy() for t in doc_emb_main])

        # Feedback set indices for this query
        dev = self.primary_evaluator.device
        sm = (sim_main if isinstance(sim_main, torch.Tensor)
              else torch.tensor(sim_main)).to(dev)
        sf = (sim_feedback if isinstance(sim_feedback, torch.Tensor)
              else torch.tensor(sim_feedback)).to(dev)
        _, top_m = torch.topk(sm[query_idx], k=self.k)
        _, top_f = torch.topk(sf[query_idx], k=self.k)
        feedback_idx = torch.unique(torch.cat([top_m, top_f])).cpu().tolist()

        # Positive doc indices for this query
        qid = query_ids[query_idx]
        pos_doc_ids = set(qrels.get(qid, {}).keys())
        pos_idx = {j for j, did in enumerate(doc_ids) if did in pos_doc_ids}

        plot_query_trajectory(
            doc_emb=doc_np,
            trajectory=trajectory,
            feedback_idx=feedback_idx,
            pos_idx=pos_idx,
            title=f"GQR Query Trajectory  [qid={qid[:30].replace('$', '')}]",
            save_path=save_path,
        )
