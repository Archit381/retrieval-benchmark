import torch


def _l2_norm(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def score_multi_vector(
    query_embs: list[torch.Tensor],
    doc_embs: list[torch.Tensor],
    device: str = "cpu",
) -> torch.Tensor:
    """MaxSim scoring for ColPali-style multi-vector embeddings.

    score(q_i, d_j) = sum_t max_s cos_sim(q_i[t], d_j[s])
    Returns [Nq, Nd] float32.
    """
    Nq, Nd = len(query_embs), len(doc_embs)
    scores = torch.zeros(Nq, Nd, dtype=torch.float32)

    # pre-normalize docs once
    d_norms = [_l2_norm(d.to(device).float()) for d in doc_embs]

    for i, q in enumerate(query_embs):
        q_n = _l2_norm(q.to(device).float())       # [Tq, D]
        for j, dn in enumerate(d_norms):
            sim = q_n @ dn.T                        # [Tq, Td]
            scores[i, j] = sim.max(dim=1).values.sum()

    return scores


def score_cosine(
    query_embs: torch.Tensor,
    doc_embs: torch.Tensor,
    device: str = "cpu",
) -> torch.Tensor:
    """Cosine similarity for dense single-vector embeddings.

    Returns [Nq, Nd] float32.
    """
    q = _l2_norm(query_embs.to(device).float())
    d = _l2_norm(doc_embs.to(device).float())
    return q @ d.T
