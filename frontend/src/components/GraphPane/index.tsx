/**
 * GraphPane - Interactive model architecture visualization using React Flow.
 */
import { useCallback, useMemo } from 'react';
import {
  ReactFlow,
  Node,
  Edge,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  NodeMouseHandler,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { useAppStore } from '../../store/appStore';
import { LayerNode } from './LayerNode';

// Custom node types
const nodeTypes = {
  module: LayerNode,
  input: LayerNode,
  output: LayerNode,
  function: LayerNode,
  method: LayerNode,
  attr: LayerNode,
  other: LayerNode,
};

// Color mapping for layer types
const layerTypeColors: Record<string, string> = {
  mamba: '#10b981', // emerald
  attention: '#3b82f6', // blue
  norm: '#8b5cf6', // violet
  mlp: '#f59e0b', // amber
  other: '#6b7280', // gray
};

export function GraphPane() {
  const { graphNodes, graphEdges, selectedLayers, toggleLayerSelection } = useAppStore();

  // Convert API nodes to React Flow nodes
  const initialNodes: Node[] = useMemo(() => {
    return graphNodes.map((node, index) => {
      const layerType = (node.data.layerType as string) || 'other';
      const isSelected = selectedLayers.includes(node.data.fullName || node.id);

      return {
        id: node.id,
        type: 'module',
        position: node.position || { x: 0, y: index * 60 },
        data: {
          ...node.data,
          color: layerTypeColors[layerType] || layerTypeColors.other,
          isSelected,
        },
        style: {
          border: isSelected ? '2px solid #2563eb' : undefined,
        },
      };
    });
  }, [graphNodes, selectedLayers]);

  // Convert API edges to React Flow edges
  const initialEdges: Edge[] = useMemo(() => {
    return graphEdges.map((edge) => ({
      id: edge.id,
      source: edge.source,
      target: edge.target,
      type: 'smoothstep',
      animated: false,
      style: { stroke: '#94a3b8' },
    }));
  }, [graphEdges]);

  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

  // Update nodes when selection changes
  useMemo(() => {
    setNodes(initialNodes);
    setEdges(initialEdges);
  }, [initialNodes, initialEdges, setNodes, setEdges]);

  // Handle node click for layer selection
  const onNodeClick: NodeMouseHandler = useCallback(
    (event, node) => {
      const layerName = (node.data as { fullName?: string }).fullName || node.id;
      toggleLayerSelection(layerName);
    },
    [toggleLayerSelection]
  );

  return (
    <div className="h-full w-full">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeClick={onNodeClick}
        nodeTypes={nodeTypes}
        fitView
        attributionPosition="bottom-left"
      >
        <Background color="#e5e7eb" gap={16} />
        <Controls />
        <MiniMap
          nodeColor={(node) => (node.data as { color?: string }).color || '#6b7280'}
          maskColor="rgba(255, 255, 255, 0.8)"
        />
      </ReactFlow>

      {/* Legend */}
      <div className="absolute bottom-4 right-4 bg-white p-3 rounded-lg shadow-md text-sm">
        <div className="font-medium mb-2">Layer Types</div>
        {Object.entries(layerTypeColors).map(([type, color]) => (
          <div key={type} className="flex items-center gap-2">
            <div
              className="w-3 h-3 rounded"
              style={{ backgroundColor: color }}
            />
            <span className="capitalize">{type}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default GraphPane;
