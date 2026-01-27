"""Statistics and histogram computation for weight/activation tensors."""

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class TensorStats:
    """Statistics for a tensor."""

    shape: list[int]
    dtype: str
    numel: int
    mean: float
    std: float
    min: float
    max: float
    p1: float  # 1st percentile
    p5: float  # 5th percentile
    p25: float  # 25th percentile (Q1)
    p50: float  # 50th percentile (median)
    p75: float  # 75th percentile (Q3)
    p95: float  # 95th percentile
    p99: float  # 99th percentile
    zero_count: int
    zero_ratio: float
    nan_count: int
    inf_count: int

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "shape": self.shape,
            "dtype": self.dtype,
            "numel": self.numel,
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "p1": self.p1,
            "p5": self.p5,
            "p25": self.p25,
            "p50": self.p50,
            "p75": self.p75,
            "p95": self.p95,
            "p99": self.p99,
            "zero_count": self.zero_count,
            "zero_ratio": self.zero_ratio,
            "nan_count": self.nan_count,
            "inf_count": self.inf_count,
        }


@dataclass
class Histogram:
    """Histogram data for a tensor."""

    bins: list[float]  # Bin edges (n+1 values)
    counts: list[int]  # Counts per bin (n values)
    bin_centers: list[float]  # Center of each bin (n values)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "bins": self.bins,
            "counts": self.counts,
            "bin_centers": self.bin_centers,
        }


def compute_tensor_stats(tensor: np.ndarray) -> TensorStats:
    """Compute comprehensive statistics for a tensor."""
    flat = tensor.flatten()

    # Handle special values
    nan_mask = np.isnan(flat)
    inf_mask = np.isinf(flat)
    valid_mask = ~(nan_mask | inf_mask)
    valid = flat[valid_mask]

    if len(valid) == 0:
        # All values are NaN or Inf
        return TensorStats(
            shape=list(tensor.shape),
            dtype=str(tensor.dtype),
            numel=tensor.size,
            mean=float("nan"),
            std=float("nan"),
            min=float("nan"),
            max=float("nan"),
            p1=float("nan"),
            p5=float("nan"),
            p25=float("nan"),
            p50=float("nan"),
            p75=float("nan"),
            p95=float("nan"),
            p99=float("nan"),
            zero_count=0,
            zero_ratio=0.0,
            nan_count=int(nan_mask.sum()),
            inf_count=int(inf_mask.sum()),
        )

    # Compute percentiles
    percentiles = np.percentile(valid, [1, 5, 25, 50, 75, 95, 99])

    return TensorStats(
        shape=list(tensor.shape),
        dtype=str(tensor.dtype),
        numel=tensor.size,
        mean=float(np.mean(valid)),
        std=float(np.std(valid)),
        min=float(np.min(valid)),
        max=float(np.max(valid)),
        p1=float(percentiles[0]),
        p5=float(percentiles[1]),
        p25=float(percentiles[2]),
        p50=float(percentiles[3]),
        p75=float(percentiles[4]),
        p95=float(percentiles[5]),
        p99=float(percentiles[6]),
        zero_count=int(np.sum(valid == 0)),
        zero_ratio=float(np.mean(valid == 0)),
        nan_count=int(nan_mask.sum()),
        inf_count=int(inf_mask.sum()),
    )


def compute_histogram(
    tensor: np.ndarray,
    num_bins: int = 100,
    clip_percentile: float | None = None,
) -> Histogram:
    """Compute histogram for a tensor.

    Args:
        tensor: Input tensor
        num_bins: Number of histogram bins
        clip_percentile: If set, clip values outside this percentile range (e.g., 99.9)
    """
    flat = tensor.flatten()

    # Remove NaN and Inf
    valid = flat[np.isfinite(flat)]

    if len(valid) == 0:
        return Histogram(bins=[], counts=[], bin_centers=[])

    # Optional clipping for outlier handling
    if clip_percentile is not None:
        low = np.percentile(valid, 100 - clip_percentile)
        high = np.percentile(valid, clip_percentile)
        valid = np.clip(valid, low, high)

    # Compute histogram
    counts, bin_edges = np.histogram(valid, bins=num_bins)

    # Compute bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return Histogram(
        bins=[float(x) for x in bin_edges],
        counts=[int(x) for x in counts],
        bin_centers=[float(x) for x in bin_centers],
    )


def compute_per_channel_stats(
    tensor: np.ndarray,
    channel_dim: int = 0,
) -> dict:
    """Compute per-channel statistics for outlier detection.

    For weight tensors, typically shape is (out_features, in_features).
    channel_dim=0 gives per-output-channel stats.
    """
    if tensor.ndim < 2:
        # 1D tensor - treat each element as a channel
        return {
            "channel_maxes": [float(x) for x in np.abs(tensor)],
            "channel_mins": [float(x) for x in tensor],
            "channel_means": [float(x) for x in tensor],
            "channel_stds": [0.0] * len(tensor),
            "num_channels": len(tensor),
        }

    # Move channel dim to first position
    tensor = np.moveaxis(tensor, channel_dim, 0)
    num_channels = tensor.shape[0]

    # Flatten each channel
    flat_channels = tensor.reshape(num_channels, -1)

    return {
        "channel_maxes": [float(x) for x in np.max(np.abs(flat_channels), axis=1)],
        "channel_mins": [float(x) for x in np.min(flat_channels, axis=1)],
        "channel_means": [float(x) for x in np.mean(flat_channels, axis=1)],
        "channel_stds": [float(x) for x in np.std(flat_channels, axis=1)],
        "num_channels": num_channels,
    }


def detect_outlier_channels(
    channel_maxes: list[float],
    threshold_std: float = 3.0,
) -> list[int]:
    """Detect outlier channels based on max values.

    Returns indices of channels with max values > threshold_std standard deviations
    from the mean of max values.
    """
    if len(channel_maxes) < 2:
        return []

    maxes = np.array(channel_maxes)
    mean_max = np.mean(maxes)
    std_max = np.std(maxes)

    if std_max == 0:
        return []

    z_scores = (maxes - mean_max) / std_max
    outlier_indices = np.where(z_scores > threshold_std)[0]

    return [int(x) for x in outlier_indices]


def compute_heatmap(
    tensor: np.ndarray,
    max_size: int = 256,
    normalize: bool = True,
) -> dict:
    """Convert tensor to 2D heatmap, downsampling if needed.

    Args:
        tensor: Input tensor (will be reshaped to 2D)
        max_size: Maximum dimension size (downsample if larger)
        normalize: If True, normalize values to [-1, 1]
    """
    # Reshape to 2D
    if tensor.ndim == 1:
        # 1D: make it a row vector
        data = tensor.reshape(1, -1)
    elif tensor.ndim == 2:
        data = tensor
    else:
        # Higher dimensional: flatten to 2D
        # Keep first dim, flatten rest
        data = tensor.reshape(tensor.shape[0], -1)

    original_shape = list(data.shape)

    # Downsample if needed
    if data.shape[0] > max_size or data.shape[1] > max_size:
        # Simple downsampling via striding
        stride_h = max(1, data.shape[0] // max_size)
        stride_w = max(1, data.shape[1] // max_size)
        data = data[::stride_h, ::stride_w]

    displayed_shape = list(data.shape)

    # Handle NaN/Inf
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalize to [-1, 1]
    if normalize and data.size > 0:
        abs_max = np.max(np.abs(data))
        if abs_max > 0:
            data = data / abs_max

    return {
        "data": [[float(x) for x in row] for row in data],
        "original_shape": original_shape,
        "displayed_shape": displayed_shape,
        "min": float(np.min(data)) if data.size > 0 else 0.0,
        "max": float(np.max(data)) if data.size > 0 else 0.0,
    }
