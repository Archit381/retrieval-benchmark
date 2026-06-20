from __future__ import annotations

from typing import Any, Optional

import torch
from transformers import CLIPModel, CLIPProcessor

from src.embedder._base import BaseEmbedder


class PubMedCLIPEmbedder(BaseEmbedder):
    """Single-vector embedder for flaviagiammarino/pubmed-clip-vit-base-patch32 (512-dim)."""

    model_type = "pubmedclip"
    is_multivector = False

    def __init__(self, model_name: str, device: Optional[str] = None, dtype: Any = None):
        super().__init__(model_name, device, dtype)
        self._processor = None

    def _load(self) -> None:
        self._model = CLIPModel.from_pretrained(self.model_name).to(self.device).eval()
        self._processor = CLIPProcessor.from_pretrained(self.model_name)

    @torch.no_grad()
    def _encode_text(self, texts: list[str]) -> torch.Tensor:
        inputs = self._processor(
            text=texts, return_tensors="pt", padding=True, truncation=True, max_length=77
        ).to(self.device)
        outputs = self._model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        features = outputs.pooler_output
        return features / features.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def _encode_image(self, images: list[Any]) -> torch.Tensor:
        inputs = self._processor(images=images, return_tensors="pt").to(self.device)
        outputs = self._model.get_image_features(pixel_values=inputs["pixel_values"])
        features = outputs.pooler_output
        return features / features.norm(dim=-1, keepdim=True)

    def unload(self) -> None:
        self._processor = None
        super().unload()
