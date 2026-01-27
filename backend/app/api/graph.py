"""API endpoints for model graph extraction."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/{model_id}")
async def get_graph(model_id: str) -> dict:
    """Get the model architecture graph in React Flow format."""
    # TODO: implement graph extraction via GraphExtractor
    return {
        "model_id": model_id,
        "nodes": [],
        "edges": [],
    }
