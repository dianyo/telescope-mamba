"""Graph extraction for model architecture visualization."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)


@dataclass
class GraphNode:
    """Node in the model graph (React Flow format)."""

    id: str
    type: str  # "module", "function", "input", "output"
    data: dict[str, Any] = field(default_factory=dict)
    position: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "data": self.data,
            "position": self.position,
        }


@dataclass
class GraphEdge:
    """Edge in the model graph (React Flow format)."""

    id: str
    source: str
    target: str
    sourceHandle: str | None = None
    targetHandle: str | None = None

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "source": self.source,
            "target": self.target,
        }
        if self.sourceHandle:
            d["sourceHandle"] = self.sourceHandle
        if self.targetHandle:
            d["targetHandle"] = self.targetHandle
        return d


@dataclass
class ModelGraph:
    """Complete model graph for React Flow."""

    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "metadata": self.metadata,
        }


def extract_graph_fx(model: nn.Module) -> ModelGraph | None:
    """Extract graph using PyTorch FX symbolic tracing.

    Returns None if tracing fails (e.g., for models with dynamic control flow).
    """
    try:
        # Attempt symbolic tracing
        traced = torch.fx.symbolic_trace(model)

        nodes: list[GraphNode] = []
        edges: list[GraphEdge] = []
        node_ids: dict[str, str] = {}  # FX node name -> our node ID

        # Layout parameters
        x_spacing = 250
        y_spacing = 80
        current_y = 0

        for fx_node in traced.graph.nodes:
            node_id = fx_node.name
            node_ids[fx_node.name] = node_id

            # Determine node type and label
            if fx_node.op == "placeholder":
                node_type = "input"
                label = fx_node.name
            elif fx_node.op == "output":
                node_type = "output"
                label = "output"
            elif fx_node.op == "call_module":
                node_type = "module"
                label = str(fx_node.target)
                # Get module class name
                try:
                    module = traced.get_submodule(fx_node.target)
                    label = f"{fx_node.target}\n({module.__class__.__name__})"
                except Exception:
                    pass
            elif fx_node.op == "call_function":
                node_type = "function"
                label = fx_node.target.__name__ if hasattr(fx_node.target, "__name__") else str(fx_node.target)
            elif fx_node.op == "call_method":
                node_type = "method"
                label = fx_node.target
            elif fx_node.op == "get_attr":
                node_type = "attr"
                label = fx_node.target
            else:
                node_type = "other"
                label = str(fx_node.target)

            # Create node
            nodes.append(
                GraphNode(
                    id=node_id,
                    type=node_type,
                    data={
                        "label": label,
                        "op": fx_node.op,
                        "target": str(fx_node.target),
                    },
                    position={"x": 0, "y": current_y},
                )
            )
            current_y += y_spacing

            # Create edges from args
            for i, arg in enumerate(fx_node.args):
                if isinstance(arg, torch.fx.Node) and arg.name in node_ids:
                    edges.append(
                        GraphEdge(
                            id=f"{node_ids[arg.name]}->{node_id}",
                            source=node_ids[arg.name],
                            target=node_id,
                        )
                    )

        logger.info(f"FX tracing successful: {len(nodes)} nodes, {len(edges)} edges")

        return ModelGraph(
            nodes=nodes,
            edges=edges,
            metadata={
                "extraction_method": "fx",
                "num_nodes": len(nodes),
                "num_edges": len(edges),
            },
        )

    except Exception as e:
        logger.warning(f"FX tracing failed: {e}")
        return None


def extract_graph_modules(
    model: nn.Module,
    layer_type_fn: callable | None = None,
) -> ModelGraph:
    """Extract graph from named_modules() - always works but coarser granularity.

    Args:
        model: The model to extract graph from
        layer_type_fn: Optional function (layer_name: str) -> str to get layer type
    """
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []

    # Build parent-child relationships
    module_parents: dict[str, str] = {}
    all_modules = list(model.named_modules())

    for name, module in all_modules:
        if not name:
            continue

        # Find parent
        parts = name.rsplit(".", 1)
        if len(parts) == 2:
            parent_name = parts[0]
            module_parents[name] = parent_name

    # Layout parameters - hierarchical layout
    depth_x = {}  # depth -> current x position
    y_spacing = 60
    x_spacing = 200
    current_y = 0

    for name, module in all_modules:
        if not name:
            # Root module
            name = "root"

        # Calculate depth
        depth = name.count(".") if name != "root" else 0

        # Get layer type
        layer_type = "other"
        if layer_type_fn:
            try:
                layer_type = layer_type_fn(name)
            except Exception:
                pass

        # Determine if this is a leaf (no children) or container
        is_leaf = not any(
            child_name.startswith(name + ".") for child_name, _ in all_modules if child_name
        )

        # Get weight info
        has_weight = hasattr(module, "weight") and module.weight is not None
        weight_shape = list(module.weight.shape) if has_weight else None
        param_count = sum(p.numel() for p in module.parameters(recurse=False))

        # Create node
        nodes.append(
            GraphNode(
                id=name,
                type="module",
                data={
                    "label": name.split(".")[-1] if "." in name else name,
                    "fullName": name,
                    "className": module.__class__.__name__,
                    "layerType": layer_type,
                    "isLeaf": is_leaf,
                    "hasWeight": has_weight,
                    "weightShape": weight_shape,
                    "paramCount": param_count,
                },
                position={"x": depth * x_spacing, "y": current_y},
            )
        )
        current_y += y_spacing

        # Create edge to parent
        if name in module_parents:
            parent = module_parents[name]
            edges.append(
                GraphEdge(
                    id=f"{parent}->{name}",
                    source=parent,
                    target=name,
                )
            )
        elif name != "root":
            # Connect to root
            edges.append(
                GraphEdge(
                    id=f"root->{name}",
                    source="root",
                    target=name,
                )
            )

    logger.info(f"Module extraction: {len(nodes)} nodes, {len(edges)} edges")

    return ModelGraph(
        nodes=nodes,
        edges=edges,
        metadata={
            "extraction_method": "modules",
            "num_nodes": len(nodes),
            "num_edges": len(edges),
        },
    )


def extract_graph(
    model: nn.Module,
    use_fx: bool = True,
    layer_type_fn: callable | None = None,
) -> ModelGraph:
    """Extract model graph, trying FX first if enabled.

    Args:
        model: The model to extract graph from
        use_fx: Whether to try FX tracing first
        layer_type_fn: Function to determine layer type (for coloring)

    Returns:
        ModelGraph in React Flow format
    """
    if use_fx:
        fx_graph = extract_graph_fx(model)
        if fx_graph is not None:
            return fx_graph

    # Fallback to module-based extraction
    return extract_graph_modules(model, layer_type_fn)


def filter_graph_by_depth(graph: ModelGraph, max_depth: int) -> ModelGraph:
    """Filter graph to show only nodes up to a certain depth."""
    filtered_nodes = []
    filtered_node_ids = set()

    for node in graph.nodes:
        name = node.data.get("fullName", node.id)
        depth = name.count(".")
        if depth <= max_depth:
            filtered_nodes.append(node)
            filtered_node_ids.add(node.id)

    # Filter edges to only include edges between remaining nodes
    filtered_edges = [
        edge
        for edge in graph.edges
        if edge.source in filtered_node_ids and edge.target in filtered_node_ids
    ]

    return ModelGraph(
        nodes=filtered_nodes,
        edges=filtered_edges,
        metadata={
            **graph.metadata,
            "filtered_depth": max_depth,
        },
    )


def filter_graph_leaves_only(graph: ModelGraph) -> ModelGraph:
    """Filter graph to show only leaf nodes (nodes with weights)."""
    filtered_nodes = [
        node for node in graph.nodes if node.data.get("isLeaf", False)
    ]
    filtered_node_ids = {node.id for node in filtered_nodes}

    # For leaves, we might want to show connections differently
    # For now, just keep edges between remaining nodes
    filtered_edges = [
        edge
        for edge in graph.edges
        if edge.source in filtered_node_ids and edge.target in filtered_node_ids
    ]

    return ModelGraph(
        nodes=filtered_nodes,
        edges=filtered_edges,
        metadata={
            **graph.metadata,
            "filter": "leaves_only",
        },
    )
