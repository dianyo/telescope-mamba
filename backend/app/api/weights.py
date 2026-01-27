"""API endpoints for weight inspection."""

import logging

from fastapi import APIRouter, HTTPException, Query

from app.core.distribution import (
    compute_heatmap,
    compute_histogram,
    compute_per_channel_stats,
    compute_tensor_stats,
    detect_outlier_channels,
)
from app.core.model_wrapper import model_manager
from app.schemas.weights import (
    HeatmapResponse,
    Histogram,
    PerChannelResponse,
    PerChannelStats,
    TensorStats,
    WeightStatsResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/{model_id}/{layer_name:path}/stats", response_model=WeightStatsResponse)
async def get_weight_stats(
    model_id: str,
    layer_name: str,
    num_bins: int = Query(default=100, ge=10, le=500, description="Number of histogram bins"),
    clip_percentile: float | None = Query(
        default=None, ge=90, le=100, description="Clip outliers at this percentile"
    ),
) -> WeightStatsResponse:
    """Get weight statistics and histogram for a specific layer."""
    if model_id not in model_manager.list_loaded():
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' is not loaded")

    wrapper = model_manager.get_wrapper(model_id)

    try:
        weight = wrapper.get_weight_tensor(layer_name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(f"Failed to get weight for {layer_name}")
        raise HTTPException(status_code=500, detail=f"Failed to get weight: {e}")

    # Compute stats
    stats = compute_tensor_stats(weight)
    histogram = compute_histogram(weight, num_bins=num_bins, clip_percentile=clip_percentile)

    return WeightStatsResponse(
        model_id=model_id,
        layer_name=layer_name,
        stats=TensorStats(
            shape=stats.shape,
            dtype=stats.dtype,
            numel=stats.numel,
            mean=stats.mean,
            std=stats.std,
            min=stats.min,
            max=stats.max,
            p1=stats.p1,
            p5=stats.p5,
            p25=stats.p25,
            p50=stats.p50,
            p75=stats.p75,
            p95=stats.p95,
            p99=stats.p99,
            zeroCount=stats.zero_count,
            zeroRatio=stats.zero_ratio,
            nanCount=stats.nan_count,
            infCount=stats.inf_count,
        ),
        histogram=Histogram(
            bins=histogram.bins,
            counts=histogram.counts,
            binCenters=histogram.bin_centers,
        ),
    )


@router.get("/{model_id}/{layer_name:path}/heatmap", response_model=HeatmapResponse)
async def get_weight_heatmap(
    model_id: str,
    layer_name: str,
    max_size: int = Query(default=256, ge=32, le=1024, description="Maximum dimension size"),
    normalize: bool = Query(default=True, description="Normalize values to [-1, 1]"),
) -> HeatmapResponse:
    """Get weight tensor as a 2D heatmap (downsampled if needed)."""
    if model_id not in model_manager.list_loaded():
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' is not loaded")

    wrapper = model_manager.get_wrapper(model_id)

    try:
        weight = wrapper.get_weight_tensor(layer_name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(f"Failed to get weight for {layer_name}")
        raise HTTPException(status_code=500, detail=f"Failed to get weight: {e}")

    heatmap = compute_heatmap(weight, max_size=max_size, normalize=normalize)

    return HeatmapResponse(
        model_id=model_id,
        layer_name=layer_name,
        data=heatmap["data"],
        originalShape=heatmap["original_shape"],
        displayedShape=heatmap["displayed_shape"],
        min=heatmap["min"],
        max=heatmap["max"],
    )


@router.get("/{model_id}/{layer_name:path}/per-channel", response_model=PerChannelResponse)
async def get_per_channel_stats(
    model_id: str,
    layer_name: str,
    channel_dim: int = Query(default=0, ge=0, description="Channel dimension"),
    outlier_threshold: float = Query(
        default=3.0, ge=1.0, le=10.0, description="Outlier detection threshold (std devs)"
    ),
) -> PerChannelResponse:
    """Get per-channel statistics for outlier detection."""
    if model_id not in model_manager.list_loaded():
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' is not loaded")

    wrapper = model_manager.get_wrapper(model_id)

    try:
        weight = wrapper.get_weight_tensor(layer_name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(f"Failed to get weight for {layer_name}")
        raise HTTPException(status_code=500, detail=f"Failed to get weight: {e}")

    if channel_dim >= weight.ndim:
        raise HTTPException(
            status_code=400,
            detail=f"channel_dim {channel_dim} >= tensor ndim {weight.ndim}",
        )

    stats = compute_per_channel_stats(weight, channel_dim=channel_dim)
    outliers = detect_outlier_channels(stats["channel_maxes"], threshold_std=outlier_threshold)

    return PerChannelResponse(
        model_id=model_id,
        layer_name=layer_name,
        stats=PerChannelStats(
            channelMaxes=stats["channel_maxes"],
            channelMins=stats["channel_mins"],
            channelMeans=stats["channel_means"],
            channelStds=stats["channel_stds"],
            numChannels=stats["num_channels"],
            outlierIndices=outliers,
        ),
    )
