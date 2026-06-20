from __future__ import annotations

from typing import Any, Optional

import torch

from src.embedder._base import BaseEmbedder


class ConchEmbedder(BaseEmbedder):
    """Single-vector embedder for MahmoodLab/conch (ViT-B-16, 512-dim)."""

    model_type = "conch"
    is_multivector = False

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        dtype: Any = None,
        hf_auth_token: Optional[str] = None,
    ):
        super().__init__(model_name, device, dtype)
        self.hf_auth_token = hf_auth_token
        self._preprocess = None
        self._tokenizer = None
        self._tokenize_fn = None

    def _load(self) -> None:
        from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize

        self._model, self._preprocess = create_model_from_pretrained(
            "conch_ViT-B-16",
            f"hf_hub:{self.model_name}",
            hf_auth_token=self.hf_auth_token,
        )

        tokenizer = get_tokenizer()

        if not hasattr(tokenizer, "batch_encode_plus"):
            tokenizer.batch_encode_plus = tokenizer

        self._tokenizer = tokenizer
        self._tokenize_fn = tokenize
        self._model = self._model.to(self.device).eval()

    @torch.no_grad()
    def _encode_text(self, texts: list[str]) -> torch.Tensor:
        tokens = self._tokenize_fn(texts=texts, tokenizer=self._tokenizer).to(self.device)
        return self._model.encode_text(tokens)

    @torch.no_grad()
    def _encode_image(self, images: list[Any]) -> torch.Tensor:
        pixels = torch.stack(
            [self._preprocess(img) for img in images]
        ).to(self.device)
        return self._model.encode_image(pixels, proj_contrast=True, normalize=True)

    def unload(self) -> None:
        self._preprocess = None
        self._tokenizer = None
        self._tokenize_fn = None
        super().unload()
