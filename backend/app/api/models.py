"""API endpoints for model management."""

import logging

from fastapi import APIRouter, HTTPException

from app.core.model_registry import (
    MODEL_REGISTRY,
    get_model_config,
    list_available_models,
)
from app.core.model_wrapper import model_manager
from app.schemas.models import (
    LayerInfo,
    LayerListResponse,
    ModelInfo,
    ModelListResponse,
    ModelLoadRequest,
    ModelLoadResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/", response_model=ModelListResponse)
async def list_models() -> ModelListResponse:
    """List available models from registry and currently loaded models."""
    available = [
        ModelInfo(
            id=m["id"],
            name=m["name"],
            type=m["type"],
            source=m["source"],
        )
        for m in list_available_models()
    ]

    return ModelListResponse(
        available=available,
        loaded=model_manager.list_loaded(),
    )


@router.post("/{model_id}/load", response_model=ModelLoadResponse)
async def load_model(model_id: str, request: ModelLoadRequest | None = None) -> ModelLoadResponse:
    """Load a model into memory."""
    try:
        config = get_model_config(model_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Check if source is configured
    if not config.hf_hub_id and not config.local_path:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_id}' has no source configured. "
            "Set hf_hub_id or local_path in the registry.",
        )

    try:
        model_manager.load_model(model_id, config)
        return ModelLoadResponse(
            status="loaded",
            model_id=model_id,
            message=f"Model '{config.name}' loaded successfully",
        )
    except Exception as e:
        logger.exception(f"Failed to load model {model_id}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")


@router.delete("/{model_id}", response_model=ModelLoadResponse)
async def unload_model(model_id: str) -> ModelLoadResponse:
    """Unload a model from memory."""
    if model_id not in model_manager.list_loaded():
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' is not loaded")

    model_manager.unload_model(model_id)
    return ModelLoadResponse(
        status="unloaded",
        model_id=model_id,
        message=f"Model '{model_id}' unloaded successfully",
    )


@router.get("/{model_id}/layers", response_model=LayerListResponse)
async def list_layers(model_id: str) -> LayerListResponse:
    """List all layers in a loaded model."""
    if model_id not in model_manager.list_loaded():
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' is not loaded")

    wrapper = model_manager.get_wrapper(model_id)
    layers = []

    for name in wrapper.get_layer_names():
        if not name:  # Skip root module
            continue

        try:
            info = wrapper.get_layer_info(name)
            layers.append(
                LayerInfo(
                    name=info["name"],
                    type=info["type"],
                    className=info["class"],
                    hasWeight=info["has_weight"],
                    hasBias=info["has_bias"],
                    weightShape=info.get("weight_shape"),
                    weightDtype=info.get("weight_dtype"),
                    paramCount=sum(
                        p.numel()
                        for p in wrapper.get_layer(name).parameters(recurse=False)
                    ),
                )
            )
        except Exception as e:
            logger.warning(f"Error getting info for layer {name}: {e}")

    return LayerListResponse(
        model_id=model_id,
        total_layers=len(layers),
        layers=layers,
    )
