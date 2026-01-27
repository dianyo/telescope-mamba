"""Pydantic schemas for model-related API endpoints."""

from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    """Information about a model in the registry."""

    id: str
    name: str
    type: str
    source: str


class ModelListResponse(BaseModel):
    """Response for listing models."""

    available: list[ModelInfo]
    loaded: list[str]


class ModelLoadRequest(BaseModel):
    """Request to load a model."""

    device: str = Field(default="cuda", description="Device to load model on")


class ModelLoadResponse(BaseModel):
    """Response after loading a model."""

    status: str
    model_id: str
    message: str | None = None


class LayerInfo(BaseModel):
    """Information about a layer."""

    name: str
    type: str  # mamba, attention, norm, mlp, other
    class_name: str = Field(alias="className")
    has_weight: bool = Field(alias="hasWeight")
    has_bias: bool = Field(alias="hasBias")
    weight_shape: list[int] | None = Field(default=None, alias="weightShape")
    weight_dtype: str | None = Field(default=None, alias="weightDtype")
    param_count: int = Field(alias="paramCount")

    class Config:
        populate_by_name = True


class LayerListResponse(BaseModel):
    """Response for listing layers."""

    model_id: str
    total_layers: int
    layers: list[LayerInfo]
