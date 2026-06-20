import torch
from safetensors.torch import save_file, load_file


def save_multi_vector(path: str, embeddings: list[torch.Tensor]) -> None:
    """Ragged flat layout: flat [sum_tokens, dim] float32 + offsets [N+1] int64."""
    flat = torch.cat([e.to(torch.float32).cpu() for e in embeddings], dim=0)
    offsets = torch.zeros(len(embeddings) + 1, dtype=torch.int64)
    for i, e in enumerate(embeddings):
        offsets[i + 1] = offsets[i] + e.shape[0]
    save_file({"flat": flat, "offsets": offsets}, path)


def load_multi_vector(path: str, device: str = "cpu") -> list[torch.Tensor]:
    tensors = load_file(path, device=device)
    flat, offsets = tensors["flat"], tensors["offsets"]
    return [flat[offsets[i] : offsets[i + 1]] for i in range(len(offsets) - 1)]


def save_single_vector(path: str, tensor: torch.Tensor) -> None:
    """Dense layout: emb [N, dim] float32."""
    save_file({"emb": tensor.to(torch.float32).cpu()}, path)


def load_single_vector(path: str, device: str = "cpu") -> torch.Tensor:
    return load_file(path, device=device)["emb"]
