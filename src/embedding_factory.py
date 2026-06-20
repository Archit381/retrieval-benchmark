from src.embedder import (
  BaseEmbedder,
  ColSmolEmbedder,
  BiomedCLIPEmbedder
)

from typing import Optional, Any, Union
from tqdm.auto import tqdm
from schema import EmbeddingOutput
from src.core.utils import _get_progress


class EmbeddingFactory:
    """
    Maintains the support map and orchestrates single- or all-model encoding.

    SUPPORT_MAP entry: model_name -> (EmbedderClass, kwargs, modalities)
      modalities is a set drawn from {"text", "image"} declaring what the
      checkpoint actually supports. This is the modality dependency check —
      two checkpoints can share a class but expose different modalities.
    """

    VALID_MODALITIES = {"text", "image"}

    SUPPORT_MAP: dict[str, tuple[type[BaseEmbedder], dict, set[str]]] = {
        "vidore/colSmol-500M": (ColSmolEmbedder, {}, {"text", "image"}),
        # "nvidia/MM-Embed":     (MMEmbedEmbedder, {}, {"text", "image"}),
        "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224":     (BiomedCLIPEmbedder, {}, {"text", "image"}),
    }

    def __init__(self, device: Optional[str] = None, dtype: Any = None):
        self.device = device
        self.dtype = dtype

    @classmethod
    def register(
        cls,
        name: str,
        embedder_cls: type[BaseEmbedder],
        modalities: set[str],
        **kwargs,
    ) -> None:
        bad = set(modalities) - cls.VALID_MODALITIES
        if bad:
            raise ValueError(f"Unknown modalities {bad}; valid: {cls.VALID_MODALITIES}")
        cls.SUPPORT_MAP[name] = (embedder_cls, kwargs, set(modalities))

    def supports(self, model_name: str) -> set[str]:
        """Modalities a registered model supports."""
        if model_name not in self.SUPPORT_MAP:
            raise KeyError(f"'{model_name}' not in support map. Available: {list(self.SUPPORT_MAP)}")
        return self.SUPPORT_MAP[model_name][2]

    def _build(self, model_name: str) -> tuple[BaseEmbedder, set[str]]:
        if model_name not in self.SUPPORT_MAP:
            raise KeyError(
                f"'{model_name}' not in support map. "
                f"Available: {list(self.SUPPORT_MAP)}"
            )
        cls, kwargs, modalities = self.SUPPORT_MAP[model_name]
        return cls(model_name, device=self.device, dtype=self.dtype, **kwargs), modalities

    def _filter_modalities(
        self, name, modalities, texts, images, strict
    ) -> tuple[Optional[list], Optional[list]]:
        """
        strict=True  (single model): raise on unsupported modality.
        strict=False (all models):   warn + set unsupported input to None,
                                    let the encode proceed with what's supported.
        """
        if texts and "text" not in modalities:
            if strict:
                raise ValueError(f"'{name}' does not support text embeddings")
            tqdm.write(
                f"[warning] '{name}' does not support text embeddings — "
                f"text_embd will be None in output"
            )
            texts = None

        if images and "image" not in modalities:
            if strict:
                raise ValueError(f"'{name}' does not support image embeddings")
            tqdm.write(
                f"[warning] '{name}' does not support image embeddings — "
                f"img_embd will be None in output"
            )
            images = None

        return texts, images

    def _encode_one(self, embedder, texts, images, batch, batch_size, show_progress, to_cpu, progress=None):
        if batch:
            return embedder.encode_batch(
                texts, images,
                batch_size=batch_size,
                show_progress=show_progress,
                to_cpu=to_cpu,
                progress=progress,
            )
        return embedder.encode(texts, images, show_progress, progress=progress)

    def _run_one(self, name, texts, images, batch, batch_size, to_cpu, progress=None, task=None):
        embedder, modalities = self._build(name)
        t, im = self._filter_modalities(name, modalities, texts, images, strict=False)

        n_text   = len(t)  if t  else 0
        n_images = len(im) if im else 0
        n_batches = (
            (n_text   + batch_size - 1) // batch_size if n_text   else 0
        ) + (
            (n_images + batch_size - 1) // batch_size if n_images else 0
        )

        def _update(desc):
            if progress and task is not None:
                progress.update(task, description=desc)

        try:
            _update(f"[yellow]⬇  loading  {name}")
            embedder.load()
            _update(f"[cyan]  encoding  {name}")

            result = self._encode_one(
                embedder, t, im, batch, batch_size,
                show_progress=True,
                to_cpu=to_cpu,
                progress=progress,
            )

            _update(f"[green]✓  done      {name}")
            return result, None

        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            _update(f"[red]✗  failed    {name}")
            if progress:
                progress.console.print(f"[red][error] '{name}': {msg}")
            return None, msg
        finally:
            embedder.unload()

    def generate(
        self,
        texts: Optional[list[str]] = None,
        images: Optional[list[Any]] = None,
        model_name: Optional[str] = None,
        batch: bool = False,
        batch_size: int = 8,
        unload_after: bool = True,
        show_progress: bool = True,
        to_cpu: bool = True,
    ) -> Union[EmbeddingOutput, dict[str, EmbeddingOutput]]:

        # Single model
        if model_name is not None:
            embedder, modalities = self._build(model_name)
            t, im = self._filter_modalities(model_name, modalities, texts, images, strict=True)
            try:
                return self._encode_one(
                    embedder, t, im, batch, batch_size, show_progress, to_cpu
                )
            except Exception as e:
                from rich.console import Console
                Console().print(f"[red][error] '{model_name}': {type(e).__name__}: {e}")
                raise
            finally:
                if unload_after:
                    embedder.unload()

        # All models
        results: dict[str, EmbeddingOutput] = {}
        errors: dict[str, str] = {}
        model_names = list(self.SUPPORT_MAP.keys())

        if show_progress:
            with _get_progress() as progress:
                task = progress.add_task(
                    f"[bold green]sweeping models", total=len(model_names)
                )
                for name in model_names:
                    result, err = self._run_one(
                        name, texts, images, batch, batch_size, to_cpu,
                        progress=progress,
                        task=task,
                    )
                    if err:
                        errors[name] = err
                    elif result is not None:
                        results[name] = result
                    progress.advance(task)
        else:
            for name in model_names:
                result, err = self._run_one(
                    name, texts, images, batch, batch_size, to_cpu
                )
                if err:
                    errors[name] = err
                elif result is not None:
                    results[name] = result

        if errors:
            from rich.console import Console
            Console().print(f"[red bold]\n{len(errors)} model(s) failed: {list(errors)}")

        return results

    def generate_sets(
        self,
        model_name: str,
        sets: dict[str, dict[str, Any]],
        batch_size: int = 8,
        show_progress: bool = True,
        to_cpu: bool = True,
    ) -> dict[str, EmbeddingOutput]:
        """Load model once, encode multiple named sets, unload.

        Args:
            sets: mapping of set_name -> {"texts": [...]} and/or {"images": [...]}

        Returns:
            mapping of set_name -> EmbeddingOutput
        """
        embedder, modalities = self._build(model_name)
        results: dict[str, EmbeddingOutput] = {}

        try:
            embedder.load()
            with _get_progress() as progress:
                for set_name, inputs in sets.items():
                    texts  = inputs.get("texts")
                    images = inputs.get("images")
                    t, im  = self._filter_modalities(model_name, modalities, texts, images, strict=True)
                    results[set_name] = embedder.encode_batch(
                        texts=t,
                        images=im,
                        batch_size=batch_size,
                        show_progress=show_progress,
                        to_cpu=to_cpu,
                        progress=progress,
                    )
        finally:
            embedder.unload()

        return results