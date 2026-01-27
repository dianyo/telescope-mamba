/**
 * DistributionPane - Weight/activation distribution visualization.
 */
import { useQuery } from '@tanstack/react-query';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from 'recharts';
import { useAppStore } from '../../store/appStore';
import { weightsApi, WeightStatsResponse } from '../../api/client';

function StatsPanel({ stats }: { stats: WeightStatsResponse['stats'] }) {
  const formatNumber = (n: number) => {
    if (Math.abs(n) < 0.001 && n !== 0) return n.toExponential(3);
    return n.toFixed(4);
  };

  return (
    <div className="grid grid-cols-2 gap-2 text-sm">
      <div className="bg-gray-50 p-2 rounded">
        <div className="text-gray-500">Shape</div>
        <div className="font-mono">[{stats.shape.join(', ')}]</div>
      </div>
      <div className="bg-gray-50 p-2 rounded">
        <div className="text-gray-500">Dtype</div>
        <div className="font-mono">{stats.dtype}</div>
      </div>
      <div className="bg-gray-50 p-2 rounded">
        <div className="text-gray-500">Mean</div>
        <div className="font-mono">{formatNumber(stats.mean)}</div>
      </div>
      <div className="bg-gray-50 p-2 rounded">
        <div className="text-gray-500">Std</div>
        <div className="font-mono">{formatNumber(stats.std)}</div>
      </div>
      <div className="bg-gray-50 p-2 rounded">
        <div className="text-gray-500">Min</div>
        <div className="font-mono">{formatNumber(stats.min)}</div>
      </div>
      <div className="bg-gray-50 p-2 rounded">
        <div className="text-gray-500">Max</div>
        <div className="font-mono">{formatNumber(stats.max)}</div>
      </div>
      <div className="bg-gray-50 p-2 rounded">
        <div className="text-gray-500">P1 / P99</div>
        <div className="font-mono">
          {formatNumber(stats.p1)} / {formatNumber(stats.p99)}
        </div>
      </div>
      <div className="bg-gray-50 p-2 rounded">
        <div className="text-gray-500">Zero Ratio</div>
        <div className="font-mono">{(stats.zeroRatio * 100).toFixed(2)}%</div>
      </div>
    </div>
  );
}

function HistogramChart({ histogram }: { histogram: WeightStatsResponse['histogram'] }) {
  const data = histogram.binCenters.map((center, i) => ({
    value: center,
    count: histogram.counts[i],
  }));

  return (
    <ResponsiveContainer width="100%" height={300}>
      <BarChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis
          dataKey="value"
          tickFormatter={(v) => v.toFixed(2)}
          tick={{ fontSize: 10 }}
        />
        <YAxis tick={{ fontSize: 10 }} />
        <Tooltip
          formatter={(value) => [String(value), 'Count']}
          labelFormatter={(label) => `Value: ${Number(label).toFixed(4)}`}
        />
        <Bar dataKey="count" fill="#3b82f6" />
      </BarChart>
    </ResponsiveContainer>
  );
}

function LayerDetails({ modelId, layerName }: { modelId: string; layerName: string }) {
  const { data, isLoading, error } = useQuery({
    queryKey: ['weights', modelId, layerName],
    queryFn: async () => {
      const response = await weightsApi.getStats(modelId, layerName);
      return response.data;
    },
  });

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-pulse text-gray-400">Loading weight stats...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-4 bg-red-50 text-red-700 rounded-lg">
        Failed to load weight statistics
      </div>
    );
  }

  if (!data) return null;

  return (
    <div className="space-y-4">
      <StatsPanel stats={data.stats} />
      <div>
        <h4 className="font-medium text-gray-700 mb-2">Weight Distribution</h4>
        <HistogramChart histogram={data.histogram} />
      </div>
    </div>
  );
}

export function DistributionPane() {
  const { selectedModelId, selectedLayers, layers } = useAppStore();

  if (!selectedModelId) {
    return (
      <div className="flex items-center justify-center h-full text-gray-400">
        No model selected
      </div>
    );
  }

  if (selectedLayers.length === 0) {
    return (
      <div className="p-4 h-full overflow-auto">
        <h3 className="font-medium text-gray-700 mb-4">Select a Layer</h3>
        <p className="text-gray-500 mb-4">
          Click on a layer in the graph to view its weight distribution.
        </p>
        <div className="text-sm text-gray-400">
          {layers.filter((l) => l.hasWeight).length} layers with weights
        </div>
      </div>
    );
  }

  return (
    <div className="p-4 h-full overflow-auto">
      {selectedLayers.map((layerName) => {
        const layer = layers.find((l) => l.name === layerName);
        const hasWeight = layer?.hasWeight ?? true;

        return (
          <div key={layerName} className="mb-6 last:mb-0">
            <div className="flex items-center justify-between mb-3">
              <h3 className="font-medium text-gray-900 truncate" title={layerName}>
                {layerName.split('.').pop()}
              </h3>
              <span className="text-xs text-gray-500 font-mono">
                {layer?.className}
              </span>
            </div>
            <div className="text-xs text-gray-400 mb-3 font-mono truncate">
              {layerName}
            </div>
            {hasWeight ? (
              <LayerDetails modelId={selectedModelId} layerName={layerName} />
            ) : (
              <div className="p-4 bg-gray-50 text-gray-500 rounded-lg text-center">
                This layer has no weight tensor
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

export default DistributionPane;
