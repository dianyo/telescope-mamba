"""API endpoints for activation capture and analysis."""

from fastapi import APIRouter

router = APIRouter()


@router.post("/register")
async def register_layers(model_id: str, layers: list[str]) -> dict:
    """Register layers to capture activations during eval."""
    # TODO: implement hook registration via ModelWrapper
    return {
        "model_id": model_id,
        "registered_layers": layers,
    }


@router.post("/run")
async def run_eval(
    model_id: str,
    dataset: str,
    num_samples: int = 100,
) -> dict:
    """Run evaluation on a dataset and capture activations."""
    # TODO: implement eval runner
    return {
        "status": "not_implemented",
        "model_id": model_id,
        "dataset": dataset,
    }


@router.get("/{model_id}/{layer_name:path}")
async def get_activation_stats(model_id: str, layer_name: str) -> dict:
    """Get activation statistics for a layer (after eval run)."""
    # TODO: implement activation stats retrieval
    return {
        "model_id": model_id,
        "layer_name": layer_name,
        "stats": {
            "shape": [],
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
        },
        "histogram": {
            "bins": [],
            "counts": [],
        },
    }


@router.get("/{model_id}/{layer_name:path}/tokens")
async def get_token_activations(
    model_id: str,
    layer_name: str,
    sample_idx: int = 0,
) -> dict:
    """Get per-token activation values for a specific sample."""
    # TODO: implement token-level activation retrieval
    return {
        "model_id": model_id,
        "layer_name": layer_name,
        "sample_idx": sample_idx,
        "tokens": [],
        "activations_per_token": [],
    }
