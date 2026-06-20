from src.embedder._base import BaseEmbedder
from colpali_engine.models import ColIdefics3, ColIdefics3Processor
import torch
from typing import Any

class ColSmolEmbedder(BaseEmbedder):
    model_type = "colsmol"
    is_multivector = True

    def _load(self) -> None:
        self._model = ColIdefics3.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype or torch.bfloat16,
            device_map=self.device,
        ).eval()
        self._processor = ColIdefics3Processor.from_pretrained(self.model_name)

    @torch.no_grad()
    def _encode_text(self, texts: list[str]) -> Any:
        batch = self._processor.process_queries(texts).to(self.device)
        return self._model(**batch)

    @torch.no_grad()
    def _encode_image(self, images: list[Any]) -> Any:
        batch = self._processor.process_images(images).to(self.device)
        return self._model(**batch)