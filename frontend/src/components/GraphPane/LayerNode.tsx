/**
 * Custom node component for displaying layer information.
 */
import { memo } from 'react';
import { Handle, Position, NodeProps } from '@xyflow/react';

interface LayerNodeData {
  label: string;
  fullName?: string;
  className?: string;
  layerType?: string;
  isLeaf?: boolean;
  hasWeight?: boolean;
  weightShape?: number[];
  paramCount?: number;
  color?: string;
  isSelected?: boolean;
}

export const LayerNode = memo(({ data }: NodeProps) => {
  const nodeData = data as LayerNodeData;
  const {
    label,
    className,
    layerType,
    hasWeight,
    weightShape,
    paramCount,
    color = '#6b7280',
    isSelected,
  } = nodeData;

  const formatParams = (count: number) => {
    if (count >= 1e9) return `${(count / 1e9).toFixed(1)}B`;
    if (count >= 1e6) return `${(count / 1e6).toFixed(1)}M`;
    if (count >= 1e3) return `${(count / 1e3).toFixed(1)}K`;
    return count.toString();
  };

  return (
    <div
      className={`px-3 py-2 rounded-lg shadow-md min-w-[120px] transition-all ${
        isSelected ? 'ring-2 ring-blue-500 ring-offset-2' : ''
      }`}
      style={{
        backgroundColor: 'white',
        borderLeft: `4px solid ${color}`,
      }}
    >
      <Handle type="target" position={Position.Top} className="w-2 h-2" />

      <div className="text-sm font-medium text-gray-900 truncate max-w-[150px]">
        {label}
      </div>

      {className && (
        <div className="text-xs text-gray-500 truncate">
          {className}
        </div>
      )}

      <div className="flex items-center gap-2 mt-1">
        {layerType && (
          <span
            className="text-xs px-1.5 py-0.5 rounded-full text-white"
            style={{ backgroundColor: color }}
          >
            {layerType}
          </span>
        )}

        {hasWeight && weightShape && (
          <span className="text-xs text-gray-400">
            [{weightShape.join('×')}]
          </span>
        )}
      </div>

      {paramCount !== undefined && paramCount > 0 && (
        <div className="text-xs text-gray-400 mt-1">
          {formatParams(paramCount)} params
        </div>
      )}

      <Handle type="source" position={Position.Bottom} className="w-2 h-2" />
    </div>
  );
});

LayerNode.displayName = 'LayerNode';
