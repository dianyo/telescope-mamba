"""API endpoints for model management."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def list_models() -> dict:
    """List available models from registry and currently loaded models."""
    return {
        "available": [],  # TODO: populate from MODEL_REGISTRY
        "loaded": [],  # TODO: track loaded models
    }


@router.post("/{model_id}/load")
async def load_model(model_id: str) -> dict:
    """Load a model into memory."""
    # TODO: implement model loading via ModelWrapper
    return {"status": "not_implemented", "model_id": model_id}


@router.delete("/{model_id}")
async def unload_model(model_id: str) -> dict:
    """Unload a model from memory."""
    # TODO: implement model unloading
    return {"status": "not_implemented", "model_id": model_id}


@router.get("/{model_id}/layers")
async def list_layers(model_id: str) -> dict:
    """List all layers in a loaded model."""
    # TODO: implement layer listing
    return {"model_id": model_id, "layers": []}
