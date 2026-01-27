"""Model registry for supported models with their configurations."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ModelType(Enum):
    """Supported model architecture types."""

    NEMOTRON = "nemotron"
    MATVLM = "matvlm"
    TINYVIM = "tinyvim"
    GENERIC = "generic"  # fallback for any HF model


@dataclass
class ModelConfig:
    """Configuration for a registered model."""

    name: str
    model_type: ModelType
    hf_hub_id: str | None = None
    local_path: str | None = None
    torch_dtype: str = "auto"  # "bfloat16", "float16", "float32", "auto"

    # Model-specific layer patterns for identification
    mamba_layer_patterns: list[str] = field(default_factory=list)
    attn_layer_patterns: list[str] = field(default_factory=list)
    norm_layer_patterns: list[str] = field(default_factory=list)
    mlp_layer_patterns: list[str] = field(default_factory=list)

    # Graph extraction hints
    use_fx_trace: bool = True  # Try FX first?

    # Additional model-specific kwargs for from_pretrained
    load_kwargs: dict[str, Any] = field(default_factory=dict)

    def get_source(self) -> str:
        """Get the model source (local path or HF hub ID)."""
        if self.local_path:
            return self.local_path
        if self.hf_hub_id:
            return self.hf_hub_id
        raise ValueError(f"Model {self.name} has no source configured")


# Common layer patterns for Mamba-transformer hybrids
MAMBA_PATTERNS = ["mamba", "ssm", "conv1d", "x_proj", "dt_proj", "out_proj"]
ATTN_PATTERNS = ["self_attn", "attention", "q_proj", "k_proj", "v_proj", "o_proj"]
NORM_PATTERNS = ["layernorm", "layer_norm", "rmsnorm", "rms_norm", "norm"]
MLP_PATTERNS = ["mlp", "feed_forward", "fc1", "fc2", "gate_proj", "up_proj", "down_proj"]


MODEL_REGISTRY: dict[str, ModelConfig] = {
    # Nemotron 3 Nano variants
    "nemotron-nano-bf16": ModelConfig(
        name="Nemotron 3 Nano (BF16)",
        model_type=ModelType.NEMOTRON,
        hf_hub_id="nvidia/Hymba-1.5B-Instruct",  # Example Mamba-hybrid from NVIDIA
        torch_dtype="bfloat16",
        mamba_layer_patterns=MAMBA_PATTERNS,
        attn_layer_patterns=ATTN_PATTERNS,
        norm_layer_patterns=NORM_PATTERNS,
        mlp_layer_patterns=MLP_PATTERNS,
        load_kwargs={"trust_remote_code": True},
    ),
    "nemotron-nano-fp8": ModelConfig(
        name="Nemotron 3 Nano (FP8)",
        model_type=ModelType.NEMOTRON,
        hf_hub_id="nvidia/Hymba-1.5B-Instruct",
        torch_dtype="float16",  # FP8 loaded as FP16 then quantized
        mamba_layer_patterns=MAMBA_PATTERNS,
        attn_layer_patterns=ATTN_PATTERNS,
        norm_layer_patterns=NORM_PATTERNS,
        mlp_layer_patterns=MLP_PATTERNS,
        load_kwargs={"trust_remote_code": True},
    ),
    # MaTVLM - Mamba-based Vision-Language Model
    "matvlm": ModelConfig(
        name="MaTVLM",
        model_type=ModelType.MATVLM,
        hf_hub_id=None,  # To be configured with actual model path
        torch_dtype="bfloat16",
        mamba_layer_patterns=MAMBA_PATTERNS,
        attn_layer_patterns=ATTN_PATTERNS,
        norm_layer_patterns=NORM_PATTERNS,
        mlp_layer_patterns=MLP_PATTERNS,
        load_kwargs={"trust_remote_code": True},
    ),
    # TinyViM - Tiny Vision Mamba
    "tinyvim": ModelConfig(
        name="TinyViM",
        model_type=ModelType.TINYVIM,
        hf_hub_id=None,  # To be configured with actual model path
        torch_dtype="float16",
        mamba_layer_patterns=MAMBA_PATTERNS + ["vim", "vision_mamba"],
        attn_layer_patterns=ATTN_PATTERNS,
        norm_layer_patterns=NORM_PATTERNS,
        mlp_layer_patterns=MLP_PATTERNS,
        load_kwargs={"trust_remote_code": True},
    ),
    # Generic model for testing (GPT-2 small)
    "gpt2-small": ModelConfig(
        name="GPT-2 Small (Test)",
        model_type=ModelType.GENERIC,
        hf_hub_id="gpt2",
        torch_dtype="float32",
        mamba_layer_patterns=[],
        attn_layer_patterns=["attn", "c_attn", "c_proj"],
        norm_layer_patterns=["ln_1", "ln_2", "ln_f"],
        mlp_layer_patterns=["mlp", "c_fc", "c_proj"],
        use_fx_trace=True,
    ),
}


def get_model_config(model_id: str) -> ModelConfig:
    """Get model configuration by ID."""
    if model_id not in MODEL_REGISTRY:
        raise ValueError(
            f"Model '{model_id}' not found in registry. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_id]


def list_available_models() -> list[dict[str, str]]:
    """List all available models in the registry."""
    return [
        {
            "id": model_id,
            "name": config.name,
            "type": config.model_type.value,
            "source": config.hf_hub_id or config.local_path or "not configured",
        }
        for model_id, config in MODEL_REGISTRY.items()
    ]


def register_model(model_id: str, config: ModelConfig) -> None:
    """Register a new model configuration."""
    MODEL_REGISTRY[model_id] = config
