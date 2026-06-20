from typing import Optional, Union, Any
from pydantic import BaseModel, ConfigDict, Field

class EmbeddingMetadata(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    model_type: str
    embedding_dim: Optional[int] = None
    text_embd_shape: Optional[str] = None
    img_embd_shape: Optional[str] = None
    num_text_inputs: int = 0
    num_image_inputs: int = 0
    is_multivector: bool = False
    dtype: Optional[str] = None
    device: Optional[str] = None

    encode_seconds: float = 0.0

    peak_vram_mb: Optional[float] = None
    delta_vram_mb: Optional[float] = None
    ram_used_mb: Optional[float] = None

    extra: dict[str, Any] = Field(default_factory=dict)


class EmbeddingOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    text_embd: Optional[Any] = None      # Tensor or list[Tensor] (multivector)
    img_embd: Optional[Any] = None
    metadata: EmbeddingMetadata