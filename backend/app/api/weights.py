"""API endpoints for weight inspection."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/{model_id}/{layer_name:path}")
async def get_weight_stats(model_id: str, layer_name: str) -> dict:
    """Get weight statistics and histogram for a specific layer."""
    # TODO: implement weight stats computation
    return {
        "model_id": model_id,
        "layer_name": layer_name,
        "stats": {
            "shape": [],
            "dtype": "",
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p1": 0.0,
            "p99": 0.0,
        },
        "histogram": {
            "bins": [],
            "counts": [],
        },
    }


@router.get("/{model_id}/{layer_name:path}/heatmap")
async def get_weight_heatmap(
    model_id: str,
    layer_name: str,
    max_size: int = 256,
) -> dict:
    """Get weight tensor as a 2D heatmap (downsampled if needed)."""
    # TODO: implement heatmap generation
    return {
        "model_id": model_id,
        "layer_name": layer_name,
        "heatmap": [],  # 2D array of values
        "original_shape": [],
        "displayed_shape": [],
    }


@router.get("/{model_id}/{layer_name:path}/per-channel")
async def get_per_channel_stats(model_id: str, layer_name: str) -> dict:
    """Get per-channel max values for outlier detection."""
    # TODO: implement per-channel stats
    return {
        "model_id": model_id,
        "layer_name": layer_name,
        "channel_maxes": [],
        "channel_means": [],
    }
