"""API endpoints for model graph extraction."""

import logging
from typing import Literal

from fastapi import APIRouter, HTTPException, Query

from app.core.graph_extractor import (
    extract_graph,
    filter_graph_by_depth,
    filter_graph_leaves_only,
)
from app.core.model_wrapper import model_manager
from app.schemas.graph import GraphEdge, GraphMetadata, GraphNode, GraphResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/{model_id}", response_model=GraphResponse)
async def get_graph(
    model_id: str,
    max_depth: int | None = Query(default=None, description="Maximum depth to show"),
    filter_mode: Literal["all", "leaves"] = Query(
        default="all", description="Filter mode: all nodes or leaves only"
    ),
    use_fx: bool = Query(default=True, description="Try FX tracing first"),
) -> GraphResponse:
    """Get the model architecture graph in React Flow format.

    Args:
        model_id: ID of the loaded model
        max_depth: Maximum depth to show (None for all)
        filter_mode: "all" for all nodes, "leaves" for leaf nodes only
        use_fx: Whether to try FX tracing first (falls back to modules)
    """
    if model_id not in model_manager.list_loaded():
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' is not loaded")

    wrapper = model_manager.get_wrapper(model_id)

    try:
        # Extract graph
        graph = extract_graph(
            wrapper.model,
            use_fx=use_fx and wrapper.config.use_fx_trace,
            layer_type_fn=wrapper.get_layer_type,
        )

        # Apply filters
        if max_depth is not None:
            graph = filter_graph_by_depth(graph, max_depth)

        if filter_mode == "leaves":
            graph = filter_graph_leaves_only(graph)

        return GraphResponse(
            model_id=model_id,
            nodes=[
                GraphNode(
                    id=n.id,
                    type=n.type,
                    data=n.data,
                    position=n.position,
                )
                for n in graph.nodes
            ],
            edges=[
                GraphEdge(
                    id=e.id,
                    source=e.source,
                    target=e.target,
                    sourceHandle=e.sourceHandle,
                    targetHandle=e.targetHandle,
                )
                for e in graph.edges
            ],
            metadata=GraphMetadata(**graph.metadata),
        )

    except Exception as e:
        logger.exception(f"Failed to extract graph for model {model_id}")
        raise HTTPException(status_code=500, detail=f"Failed to extract graph: {e}")
