import json
from dataclasses import dataclass
from typing import Any

import torch
from huggingface_hub import snapshot_download

from src.core.manifest import Manifest
from src.core.artifacts_io import load_multi_vector, load_single_vector


@dataclass
class ArtifactContext:
    """Loaded embeddings for one model subfolder (colsmol or biomedclip)."""
    sets: dict[str, Any]                # name -> list[Tensor] or Tensor
    qrels: dict[str, dict[str, int]]


@dataclass
class RetrievalContext:
    """Merged context from both model subfolders, ready for eval."""
    query_ids: list[str]
    doc_ids: list[str]
    col_img_q: list[torch.Tensor]
    col_cap_q: list[torch.Tensor]
    col_d: list[torch.Tensor]
    bio_img_q: torch.Tensor
    bio_cap_q: torch.Tensor
    bio_d: torch.Tensor
    qrels: dict[str, dict[str, int]]


def load_artifacts(
    folder: str,
    device: str = "cpu",
) -> tuple[ArtifactContext, Manifest]:
    """Load all embeddings and metadata from a model subfolder.

    GQR sets (names starting with 'query_gqr') go into ArtifactContext.refined_img_q.
    All other sets go into ArtifactContext.sets.
    """
    with open(f"{folder}/manifest.json") as f:
        manifest = Manifest.from_json(f.read())
    with open(f"{folder}/qrels.json") as f:
        qrels = json.load(f)

    sets: dict[str, Any] = {}

    for name, info in manifest.sets.items():
        path = f"{folder}/{info.filename}"
        if info.is_multivector:
            sets[name] = load_multi_vector(path, device=device)
        else:
            sets[name] = load_single_vector(path, device=device)

    ctx = ArtifactContext(sets=sets, qrels=qrels)
    return ctx, manifest


def load_artifacts_hf(
    repo_id: str,
    subfolder: str,
    device: str = "cpu",
    token: str | None = None,
    force_download: bool = False,
) -> tuple[ArtifactContext, Manifest]:
    """Download artifacts from an HF dataset repo and load them.

    Args:
        repo_id:        e.g. "architojha/m3-retrieve-eval"
        subfolder:      model subfolder inside the repo, e.g. "colsmol"
        device:         torch device for loaded tensors
        token:          HF token (needed if repo is private)
        force_download: re-download even if cached (use when hub has new files)
    """
    local_dir = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        force_download=force_download,
    )
    return load_artifacts(f"{local_dir}/{subfolder}", device=device)
