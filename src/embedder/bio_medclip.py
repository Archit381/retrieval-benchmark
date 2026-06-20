from src.embedder._base import BaseEmbedder
import torch
from typing import Any
from open_clip import create_model_from_pretrained, get_tokenizer

class BiomedCLIPEmbedder(BaseEmbedder):
    model_type = "biomedclip"
    is_multivector = False

    def __init__(self, model_name: str, device=None, dtype=None,
                 context_length: int = 256):
        super().__init__(model_name, device, dtype)
        self.context_length = context_length
        self._preprocess = None
        self._tokenizer = None

    def _load(self) -> None:
        self._model, self._preprocess = create_model_from_pretrained(f"hf-hub:{self.model_name}")
        self._tokenizer = get_tokenizer(f"hf-hub:{self.model_name}")
        self._model = self._model.to(self.device).eval()

    @torch.no_grad()
    def _encode_text(self, texts: list[str]) -> Any:
        tokens = self._tokenizer(texts, context_length=self.context_length).to(self.device)
        _, text_features, _ = self._model(None, tokens)
        return text_features

    @torch.no_grad()
    def _encode_image(self, images: list[Any]) -> Any:
        pixels = torch.stack(
            [self._preprocess(img) for img in images]
        ).to(self.device)
        image_features, _, _ = self._model(pixels, None)
        return image_features

    def unload(self) -> None:
        self._preprocess = None
        self._tokenizer = None
        super().unload()