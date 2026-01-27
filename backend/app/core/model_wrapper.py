"""Model wrapper for loading models and capturing activations."""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
import torch
from torch import nn

from app.core.model_registry import ModelConfig, ModelType

logger = logging.getLogger(__name__)


class ModelWrapper:
    """Wrapper for PyTorch models with hook-based activation capture."""

    def __init__(self, config: ModelConfig, device: str = "cuda"):
        self.config = config
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model: nn.Module | None = None
        self.hooks: dict[str, torch.utils.hooks.RemovableHandle] = {}
        self.activations: dict[str, torch.Tensor] = {}
        self._layer_cache: dict[str, nn.Module] | None = None

    def load(self) -> None:
        """Load the model from HF hub or local path."""
        from transformers import AutoModel, AutoModelForCausalLM

        source = self.config.get_source()
        dtype = self._resolve_dtype()

        logger.info(f"Loading model from {source} with dtype={dtype}")

        # Try AutoModelForCausalLM first (for LLMs), fall back to AutoModel
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                source,
                torch_dtype=dtype,
                device_map=self.device,
                **self.config.load_kwargs,
            )
        except Exception:
            logger.info("AutoModelForCausalLM failed, trying AutoModel")
            self.model = AutoModel.from_pretrained(
                source,
                torch_dtype=dtype,
                device_map=self.device,
                **self.config.load_kwargs,
            )

        self.model.eval()
        self._layer_cache = None  # Reset layer cache
        logger.info(f"Model loaded: {self.config.name}")

    def unload(self) -> None:
        """Unload the model from memory."""
        self.clear_hooks()
        self.activations.clear()
        self._layer_cache = None
        if self.model is not None:
            del self.model
            self.model = None
            torch.cuda.empty_cache()
        logger.info(f"Model unloaded: {self.config.name}")

    def _resolve_dtype(self) -> torch.dtype | str:
        """Resolve dtype string to torch.dtype."""
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
            "auto": "auto",
        }
        return dtype_map.get(self.config.torch_dtype, "auto")

    def get_all_layers(self) -> dict[str, nn.Module]:
        """Get all named modules in the model."""
        if self.model is None:
            raise RuntimeError("Model not loaded")

        if self._layer_cache is None:
            self._layer_cache = dict(self.model.named_modules())

        return self._layer_cache

    def get_layer(self, layer_name: str) -> nn.Module:
        """Get a specific layer by name."""
        layers = self.get_all_layers()
        if layer_name not in layers:
            raise ValueError(f"Layer '{layer_name}' not found in model")
        return layers[layer_name]

    def get_layer_names(self) -> list[str]:
        """Get all layer names."""
        return list(self.get_all_layers().keys())

    def get_layer_type(self, layer_name: str) -> str:
        """Identify layer type using config patterns."""
        layer_name_lower = layer_name.lower()

        for pattern in self.config.mamba_layer_patterns:
            if pattern.lower() in layer_name_lower:
                return "mamba"

        for pattern in self.config.attn_layer_patterns:
            if pattern.lower() in layer_name_lower:
                return "attention"

        for pattern in self.config.norm_layer_patterns:
            if pattern.lower() in layer_name_lower:
                return "norm"

        for pattern in self.config.mlp_layer_patterns:
            if pattern.lower() in layer_name_lower:
                return "mlp"

        return "other"

    def get_layer_info(self, layer_name: str) -> dict[str, Any]:
        """Get information about a layer."""
        layer = self.get_layer(layer_name)
        layer_type = self.get_layer_type(layer_name)

        info = {
            "name": layer_name,
            "type": layer_type,
            "class": layer.__class__.__name__,
            "has_weight": hasattr(layer, "weight") and layer.weight is not None,
            "has_bias": hasattr(layer, "bias") and layer.bias is not None,
        }

        if info["has_weight"]:
            w = layer.weight
            info["weight_shape"] = list(w.shape)
            info["weight_dtype"] = str(w.dtype)
            info["weight_numel"] = w.numel()

        if info["has_bias"]:
            b = layer.bias
            info["bias_shape"] = list(b.shape)

        return info

    def get_weight_tensor(self, layer_name: str) -> np.ndarray:
        """Get weight tensor as numpy array."""
        layer = self.get_layer(layer_name)

        if not hasattr(layer, "weight") or layer.weight is None:
            raise ValueError(f"Layer '{layer_name}' has no weight tensor")

        return layer.weight.detach().cpu().float().numpy()

    def get_bias_tensor(self, layer_name: str) -> np.ndarray | None:
        """Get bias tensor as numpy array if exists."""
        layer = self.get_layer(layer_name)

        if not hasattr(layer, "bias") or layer.bias is None:
            return None

        return layer.bias.detach().cpu().float().numpy()

    # ==================== Hook Management ====================

    def _make_hook(self, layer_name: str) -> Callable:
        """Create a forward hook to capture activations."""

        def hook(module: nn.Module, input: Any, output: Any) -> None:
            # Handle different output types
            if isinstance(output, torch.Tensor):
                self.activations[layer_name] = output.detach()
            elif isinstance(output, tuple) and len(output) > 0:
                # Many layers return (output, cache) tuples
                if isinstance(output[0], torch.Tensor):
                    self.activations[layer_name] = output[0].detach()

        return hook

    def register_hooks(self, layer_names: list[str]) -> None:
        """Register forward hooks to capture activations."""
        if self.model is None:
            raise RuntimeError("Model not loaded")

        layers = self.get_all_layers()

        for name in layer_names:
            if name in self.hooks:
                continue  # Already registered

            if name not in layers:
                logger.warning(f"Layer '{name}' not found, skipping hook registration")
                continue

            module = layers[name]
            self.hooks[name] = module.register_forward_hook(self._make_hook(name))
            logger.debug(f"Registered hook for layer: {name}")

        logger.info(f"Registered {len(self.hooks)} hooks")

    def clear_hooks(self) -> None:
        """Remove all registered hooks."""
        for name, hook in self.hooks.items():
            hook.remove()
            logger.debug(f"Removed hook for layer: {name}")

        self.hooks.clear()
        self.activations.clear()
        logger.info("Cleared all hooks and activations")

    def get_activation(self, layer_name: str) -> np.ndarray | None:
        """Get captured activation for a layer."""
        if layer_name not in self.activations:
            return None

        return self.activations[layer_name].cpu().float().numpy()

    def get_all_activations(self) -> dict[str, np.ndarray]:
        """Get all captured activations."""
        return {
            name: tensor.cpu().float().numpy()
            for name, tensor in self.activations.items()
        }


class ModelManager:
    """Manages multiple loaded models."""

    def __init__(self):
        self.wrappers: dict[str, ModelWrapper] = {}

    def load_model(self, model_id: str, config: ModelConfig) -> ModelWrapper:
        """Load a model and add it to the manager."""
        if model_id in self.wrappers:
            logger.info(f"Model {model_id} already loaded")
            return self.wrappers[model_id]

        wrapper = ModelWrapper(config)
        wrapper.load()
        self.wrappers[model_id] = wrapper
        return wrapper

    def get_wrapper(self, model_id: str) -> ModelWrapper:
        """Get a loaded model wrapper."""
        if model_id not in self.wrappers:
            raise ValueError(f"Model '{model_id}' not loaded")
        return self.wrappers[model_id]

    def unload_model(self, model_id: str) -> None:
        """Unload a model."""
        if model_id in self.wrappers:
            self.wrappers[model_id].unload()
            del self.wrappers[model_id]

    def list_loaded(self) -> list[str]:
        """List loaded model IDs."""
        return list(self.wrappers.keys())

    def unload_all(self) -> None:
        """Unload all models."""
        for model_id in list(self.wrappers.keys()):
            self.unload_model(model_id)


# Global model manager instance
model_manager = ModelManager()
