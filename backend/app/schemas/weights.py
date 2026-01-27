"""Pydantic schemas for weight-related API endpoints."""

from pydantic import BaseModel, Field


class TensorStats(BaseModel):
    """Statistics for a tensor."""

    shape: list[int]
    dtype: str
    numel: int
    mean: float
    std: float
    min: float
    max: float
    p1: float
    p5: float
    p25: float
    p50: float
    p75: float
    p95: float
    p99: float
    zero_count: int = Field(alias="zeroCount")
    zero_ratio: float = Field(alias="zeroRatio")
    nan_count: int = Field(alias="nanCount")
    inf_count: int = Field(alias="infCount")

    class Config:
        populate_by_name = True


class Histogram(BaseModel):
    """Histogram data for a tensor."""

    bins: list[float]
    counts: list[int]
    bin_centers: list[float] = Field(alias="binCenters")

    class Config:
        populate_by_name = True


class WeightStatsResponse(BaseModel):
    """Response for weight statistics."""

    model_id: str
    layer_name: str
    stats: TensorStats
    histogram: Histogram


class PerChannelStats(BaseModel):
    """Per-channel statistics."""

    channel_maxes: list[float] = Field(alias="channelMaxes")
    channel_mins: list[float] = Field(alias="channelMins")
    channel_means: list[float] = Field(alias="channelMeans")
    channel_stds: list[float] = Field(alias="channelStds")
    num_channels: int = Field(alias="numChannels")
    outlier_indices: list[int] = Field(default_factory=list, alias="outlierIndices")

    class Config:
        populate_by_name = True


class PerChannelResponse(BaseModel):
    """Response for per-channel statistics."""

    model_id: str
    layer_name: str
    stats: PerChannelStats


class HeatmapResponse(BaseModel):
    """Response for weight heatmap."""

    model_id: str
    layer_name: str
    data: list[list[float]]
    original_shape: list[int] = Field(alias="originalShape")
    displayed_shape: list[int] = Field(alias="displayedShape")
    min: float
    max: float

    class Config:
        populate_by_name = True
