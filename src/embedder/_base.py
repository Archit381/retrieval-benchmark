from abc import ABC, abstractmethod
from typing import Optional, Any
import gc

import torch
from tqdm.auto import tqdm
from src.core.schema import EmbeddingOutput, EmbeddingMetadata
from src.core.utils import _track_resources, _get_progress


class BaseEmbedder(ABC):
    """Base Embedding class to be inherited."""

    model_type: str = "base"
    is_multivector: bool = False

    def __init__(self, model_name: str, device: Optional[str] = None, dtype: Any = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self._model = None
        self._processor = None

    def load(self) -> None:
        if self._model is not None:
            return                             # already loaded, skip
        self._load()

    @abstractmethod
    def _load(self) -> None:
        """Load weights + processor onto self.device."""
        ...

    @abstractmethod
    def _encode_text(self, texts: list[str]) -> Any: ...

    @abstractmethod
    def _encode_image(self, images: list[Any]) -> Any: ...

    """---------- helpers ----------"""

    def _embedding_dim(self, emb: Any) -> Optional[int]:
        try:
            if isinstance(emb, torch.Tensor):
                return emb.shape[-1]
            if isinstance(emb, list) and len(emb):
                first = emb[0]
                if isinstance(first, torch.Tensor):
                    return first.shape[-1]
        except Exception:
            return None
        return None

    def _embd_shape(self, emb: Any) -> Optional[str]:
        """Human-readable shape string stored in metadata."""
        if emb is None:
            return None
        if isinstance(emb, torch.Tensor):
            return str(tuple(emb.shape))           # (N, dim)
        if isinstance(emb, list) and len(emb):
            shapes = [tuple(e.shape) for e in emb]
            if len(set(shapes)) == 1:
                return f"list[{len(emb)}] each {shapes[0]}"
            return f"list[{len(emb)}] shapes={shapes}"
        return str(type(emb))

    @staticmethod
    def _to_cpu(emb):
        if isinstance(emb, torch.Tensor):
            return emb.detach().cpu()
        if isinstance(emb, (list, tuple)):
            return [e.detach().cpu() for e in emb]
        return emb

    @staticmethod
    def _concat(chunks):
        if not chunks:
            return None
        if isinstance(chunks[0], torch.Tensor):
            try:
                return torch.stack(chunks, dim=0)
            except RuntimeError:
                return chunks
        return chunks

    @staticmethod
    def _to_list(emb) -> list:
        """Normalise a batch output to a flat list of per-item tensors."""
        if isinstance(emb, torch.Tensor):
            if emb.dim() == 3:
                return list(emb.unbind(dim=0))
            if emb.dim() == 2:
                return list(emb.unbind(dim=0))
            if emb.dim() == 1:
                return [emb]
        if isinstance(emb, list):
            return emb
        return [emb]

    def _run_batched(self, items, encode_fn, batch_size, kind, show_progress, to_cpu, progress=None):
        out = []
        chunks = list(range(0, len(items), batch_size))

        if not show_progress or not chunks:
            for i in chunks:
                emb = encode_fn(items[i:i + batch_size])
                if to_cpu:
                    emb = self._to_cpu(emb)
                out.extend(self._to_list(emb))
            return out

        def _run_with_progress(p):
            task = p.add_task(
                f"[{self.model_type}] encoding {kind}",
                total=len(chunks),
            )
            for i in chunks:
                emb = encode_fn(items[i:i + batch_size])
                if to_cpu:
                    emb = self._to_cpu(emb)
                out.extend(self._to_list(emb))
                p.advance(task)

        # THE FIX: only use the passed-in progress if it exists,
        # otherwise create a fresh one
        if progress is not None:
            _run_with_progress(progress)
        else:
            with _get_progress() as p:
                _run_with_progress(p)

        return out

    def _build_meta(self, texts, images, dim, stats, text_emb=None, img_emb=None, extra=None) -> EmbeddingMetadata:
        return EmbeddingMetadata(
            model_name=self.model_name,
            model_type=self.model_type,
            embedding_dim=dim,
            text_embd_shape=self._embd_shape(text_emb),
            img_embd_shape=self._embd_shape(img_emb),
            num_text_inputs=len(texts),
            num_image_inputs=len(images),
            is_multivector=self.is_multivector,
            dtype=str(self.dtype) if self.dtype else None,
            device=self.device,
            encode_seconds=stats.get("encode_seconds", 0.0),
            peak_vram_mb=stats.get("peak_vram_mb"),
            delta_vram_mb=stats.get("delta_vram_mb"),
            ram_used_mb=stats.get("ram_used_mb"),
            extra=extra or {},
        )

    def encode(
        self,
        texts=None,
        images=None,
        show_progress=True,
        progress=None,
    ) -> EmbeddingOutput:
        if self._model is None:
            self.load()
        texts = texts or []
        images = images or []
        text_emb = img_emb = None

        with _track_resources(self.device) as stats:
            if texts:
                text_emb = self._encode_text(texts)
            if images:
                img_emb = self._encode_image(images)

        dim = self._embedding_dim(text_emb if text_emb is not None else img_emb)
        meta = self._build_meta(texts, images, dim, stats, text_emb, img_emb)
        return EmbeddingOutput(text_embd=text_emb, img_embd=img_emb, metadata=meta)


    def encode_batch(
        self,
        texts=None,
        images=None,
        batch_size=8,
        show_progress=True,
        to_cpu=True,
        progress=None,
    ) -> EmbeddingOutput:
        if self._model is None:
            self.load()
        texts = texts or []
        images = images or []
        text_emb = img_emb = None

        with _track_resources(self.device) as stats:
            if texts:
                text_emb = self._run_batched(
                    texts, self._encode_text, batch_size, "text", show_progress, to_cpu, progress
                )
            if images:
                img_emb = self._run_batched(
                    images, self._encode_image, batch_size, "image", show_progress, to_cpu, progress
                )

        dim = self._embedding_dim(text_emb if text_emb is not None else img_emb)
        meta = self._build_meta(texts, images, dim, stats, text_emb, img_emb, extra={"batch_size": batch_size})
        return EmbeddingOutput(text_embd=text_emb, img_embd=img_emb, metadata=meta)

    def unload(self) -> None:
        self._model = None
        self._processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()