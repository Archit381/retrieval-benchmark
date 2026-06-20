from typing import Any
from contextlib import contextmanager
import psutil
import time
import os
import numpy as np
import torch
from PIL import Image as PILImage
import io

from rich.console import Console
from rich.table import Table
from rich import box
from src.core.schema import EmbeddingOutput
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
    SpinnerColumn,
    TaskProgressColumn,
)

@contextmanager
def _track_resources(device: str):
    proc = psutil.Process()
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()

    if use_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start_vram = torch.cuda.memory_allocated() / 1024**2

    ram_before = proc.memory_info().rss / 1024**2
    t0 = time.perf_counter()

    stats: dict[str, Any] = {}
    try:
        yield stats
    finally:
        if use_cuda:
            torch.cuda.synchronize()
        stats["encode_seconds"] = time.perf_counter() - t0
        stats["ram_used_mb"] = proc.memory_info().rss / 1024**2 - ram_before
        if use_cuda:
            stats["peak_vram_mb"] = torch.cuda.max_memory_allocated() / 1024**2
            stats["delta_vram_mb"] = (
                torch.cuda.memory_allocated() / 1024**2 - start_vram
            )


def _get_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("{task.description:<40}"),
        BarColumn(bar_width=30),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        refresh_per_second=20,
        expand=False,
    )


def print_memory_usage(label: str = "") -> None:
    header = f" {label} " if label else ""
    print(f"\n{'─'*20}{header}{'─'*20}")

    # RAM
    proc = psutil.Process(os.getpid())
    ram_used = proc.memory_info().rss / 1024**2
    ram_total = psutil.virtual_memory().total / 1024**2
    ram_pct = (ram_used / ram_total) * 100
    print(f"  RAM   used : {ram_used:>8.1f} MB / {ram_total:>8.1f} MB  ({ram_pct:.1f}%)")

    # VRAM
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            alloc    = torch.cuda.memory_allocated(i)  / 1024**2
            reserved = torch.cuda.memory_reserved(i)   / 1024**2
            total    = torch.cuda.get_device_properties(i).total_memory / 1024**2
            free     = total - reserved
            print(f"  GPU {i} alloc  : {alloc:>8.1f} MB")
            print(f"  GPU {i} reserv : {reserved:>8.1f} MB / {total:>8.1f} MB  ({reserved/total*100:.1f}%)")
            print(f"  GPU {i} free   : {free:>8.1f} MB")
    else:
        print("  VRAM  : no CUDA device")

    print(f"{'─'*40}\n")



def print_embedding_output(out: EmbeddingOutput, title: str = "") -> None:
    m = out.metadata
    console = Console()

    table = Table(
        title=title or m.model_name,
        box=box.ROUNDED,
        show_header=False,
        padding=(0, 2),
    )
    table.add_column("field", style="bold cyan", width=18)
    table.add_column("value", style="white")

    # embeddings
    table.add_row("text_embd", m.text_embd_shape or "None")
    table.add_row("img_embd",  m.img_embd_shape  or "None")
    table.add_section()

    # model info
    table.add_row("model",      m.model_name)
    table.add_row("model_type", m.model_type)
    table.add_row("device",     m.device or "N/A")
    table.add_row("dtype",      m.dtype  or "N/A")
    table.add_row("multivec",   str(m.is_multivector))
    table.add_row("embd_dim",   str(m.embedding_dim) if m.embedding_dim else "N/A")
    table.add_section()

    # perf
    table.add_row("encode",     f"{m.encode_seconds:.2f}s")
    table.add_row("peak_vram",  f"{m.peak_vram_mb:.1f} MB"  if m.peak_vram_mb  is not None else "N/A")
    table.add_row("delta_vram", f"{m.delta_vram_mb:.1f} MB" if m.delta_vram_mb is not None else "N/A")
    table.add_row("ram_delta",  f"{m.ram_used_mb:.1f} MB"   if m.ram_used_mb   is not None else "N/A")

    if m.extra:
        table.add_section()
        for k, v in m.extra.items():
            table.add_row(k, str(v))

    console.print(table)


_PRETTY_NAMES: dict[str, str] = {
    "colsmol":    "ColSmol-500M",
    "biomedclip": "BiomedCLIP",
    "conch":      "CONCH",
    "pubmedclip": "PubMedCLIP",
}


def compare_ndcg(
    eval_results: dict[str, dict],
    pretty_names: dict[str, str] | None = None,
    save_dir: str | None = None,
) -> "Any":
    """Paper-ready NDCG comparison table across all cutoffs.

    Args:
        eval_results:  {model_type: result_dict} from evaluation_factory.evaluate()
        pretty_names:  optional display name overrides keyed by model_type
        save_dir:      if set, writes ndcg.tex here

    Returns:
        pandas DataFrame — rows=models, columns=nDCG@k for each cutoff
    """
    import os
    import pandas as pd
    from IPython.display import display

    names = {**_PRETTY_NAMES, **(pretty_names or {})}

    # collect all cutoffs present in results (sorted)
    cutoffs: list[int] = sorted(next(iter(eval_results.values()))["ndcg"].keys())

    def fmt_cell(val: float, base: float) -> str:
        gain = 100.0 * (val - base) / max(base, 1e-9)
        return f"{val:.3f} ({gain:+.1f}\\%)"

    # sort models by mean NDCG across all cutoffs
    model_means: dict[str, float] = {
        mt: float(np.mean([np.asarray(res["ndcg"][k]).mean() for k in cutoffs]))
        for mt, res in eval_results.items()
    }
    model_order = sorted(model_means, key=lambda m: model_means[m], reverse=True)
    best_model  = model_order[0]

    rows = {}
    for mt in model_order:
        pretty = names.get(mt, mt)
        res    = eval_results[mt]
        row    = {}
        for k in cutoffs:
            val  = float(np.asarray(res["ndcg"][k]).mean())
            best = float(np.asarray(eval_results[best_model]["ndcg"][k]).mean())
            row[f"nDCG@{k}"] = f"{val:.3f}" if mt == best_model else fmt_cell(val, best)
        rows[pretty] = row

    ndcg_df = pd.DataFrame(rows).T
    ndcg_df.index.name = "Model"

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        cols = " + ".join(f"nDCG@{k}" for k in cutoffs)
        tex = ndcg_df.to_latex(
            escape=False,
            column_format="l" + "r" * len(cutoffs),
            caption=f"{cols}. Percentages are relative gains over best model ({names.get(best_model, best_model)}).",
            label="tab:ndcg",
        )
        with open(f"{save_dir}/ndcg.tex", "w") as f:
            f.write(tex)

    print(f"\n================ NDCG @ {cutoffs} ================")
    display(ndcg_df)
    return ndcg_df


def _decode_hf_image(img) -> PILImage.Image:
    """HF Image features arrive as dicts after .to_pandas(); decode to PIL."""
    if isinstance(img, PILImage.Image):
        return img
    if isinstance(img, dict) and "bytes" in img and img["bytes"]:
        return PILImage.open(io.BytesIO(img["bytes"])).convert("RGB")
    if isinstance(img, dict) and "path" in img and img["path"]:
        return PILImage.open(img["path"]).convert("RGB")
    raise TypeError(f"Cannot decode image: {type(img)}, keys={list(img) if isinstance(img, dict) else 'N/A'}")

    