"""Pydantic schemas for graph-related API endpoints."""

from typing import Any

from pydantic import BaseModel, Field


class GraphNode(BaseModel):
    """Node in the model graph (React Flow format)."""

    id: str
    type: str
    data: dict[str, Any]
    position: dict[str, float]


class GraphEdge(BaseModel):
    """Edge in the model graph (React Flow format)."""

    id: str
    source: str
    target: str
    sourceHandle: str | None = None
    targetHandle: str | None = None


class GraphMetadata(BaseModel):
    """Metadata about the graph extraction."""

    extraction_method: str
    num_nodes: int
    num_edges: int
    filtered_depth: int | None = None
    filter: str | None = None


class GraphResponse(BaseModel):
    """Response containing the model graph."""

    model_id: str
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    metadata: GraphMetadata
