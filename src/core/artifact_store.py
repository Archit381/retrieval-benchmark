"""High-level artifact save/load that handles both multi-vector and single-vector sets."""

from __future__ import annotations

import json
import os
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from src.core.artifacts_io import save_multi_vector, save_single_vector
from src.core.manifest import Manifest, SetInfo
from src.core.schema import EmbeddingOutput


def _pick_emb(out: EmbeddingOutput) -> Any:
    """Return whichever of img_embd / text_embd is populated."""
    if out.img_embd is not None:
        return out.img_embd
    if out.text_embd is not None:
        return out.text_embd
    raise ValueError(f"EmbeddingOutput for has no embeddings (both img_embd and text_embd are None)")


def save_artifacts(
    model_name: str,
    model_type: str,
    sets_out: dict[str, EmbeddingOutput],
    roles: dict[str, str],
    query_ids: list[str],
    doc_ids: list[str],
    qrels: dict[str, dict[str, int]],
    artifacts_dir: str,
    hf_repo_id: str,
    df: Any = None,
    notes: dict | None = None,
) -> None:
    """Save embeddings, qrels, and manifest for any model type.

    Works for both multi-vector (list[Tensor]) and single-vector (Tensor) sets.
    Dispatches save_multi_vector / save_single_vector based on EmbeddingOutput metadata.

    Args:
        model_name:   HF model id, stored in notes.
        model_type:   subfolder name under artifacts_dir (e.g. "colsmol", "biomedclip").
        sets_out:     {set_name: EmbeddingOutput} from EmbeddingFactory.generate_sets().
        roles:        {set_name: "query" | "doc"} — controls which ID list gets attached.
        query_ids:    ordered query ID strings.
        doc_ids:      ordered doc ID strings.
        qrels:        {query_id: {doc_id: relevance_int}}.
        artifacts_dir: root artifacts folder (e.g. "artifacts").
        hf_repo_id:   HF dataset repo to reference in manifest.
        df:           optional DataFrame; saved as data/train.parquet with image_path dropped.
        notes:        extra metadata stored in manifest (defaults to {"model": model_name}).
    """
    subfolder = f"{artifacts_dir}/{model_type}"
    os.makedirs(subfolder, exist_ok=True)

    sets_info: dict[str, SetInfo] = {}

    for set_name, out in sets_out.items():
        emb = _pick_emb(out)
        is_mv = out.metadata.is_multivector
        dim = out.metadata.embedding_dim
        role = roles[set_name]
        filename = f"{set_name}.safetensors"

        if is_mv:
            save_multi_vector(f"{subfolder}/{filename}", emb)
        else:
            save_single_vector(f"{subfolder}/{filename}", emb)

        sets_info[set_name] = SetInfo(
            name=set_name,
            model_type=model_type,
            role=role,
            is_multivector=is_mv,
            dim=dim or 0,
            count=len(emb),
            dtype="float32",
            filename=filename,
            ids=query_ids if role == "query" else doc_ids,
        )

    with open(f"{subfolder}/qrels.json", "w") as f:
        json.dump(qrels, f)

    manifest = Manifest(
        repo_id=hf_repo_id,
        sets=sets_info,
        query_ids=query_ids,
        doc_ids=doc_ids,
        notes=notes or {"model": model_name},
    )
    with open(f"{subfolder}/manifest.json", "w") as f:
        f.write(manifest.to_json())

    if df is not None:
        data_dir = f"{artifacts_dir}/data"
        os.makedirs(data_dir, exist_ok=True)
        df_save = df.drop(columns=["image_path"], errors="ignore")
        pq.write_table(pa.Table.from_pandas(df_save), f"{data_dir}/train.parquet")
        print("train.parquet saved.")

    print(f"Saved {len(sets_out)} set(s) → {subfolder}/")
