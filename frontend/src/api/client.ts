/**
 * API client for communicating with the Telescope backend.
 */
import axios from 'axios';

const API_BASE = '/api';

export const api = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Types
export interface ModelInfo {
  id: string;
  name: string;
  type: string;
  source: string;
}

export interface ModelListResponse {
  available: ModelInfo[];
  loaded: string[];
}

export interface LayerInfo {
  name: string;
  type: string;
  className: string;
  hasWeight: boolean;
  hasBias: boolean;
  weightShape: number[] | null;
  weightDtype: string | null;
  paramCount: number;
}

export interface GraphNode {
  id: string;
  type: string;
  data: {
    label: string;
    fullName?: string;
    className?: string;
    layerType?: string;
    isLeaf?: boolean;
    hasWeight?: boolean;
    weightShape?: number[];
    paramCount?: number;
    [key: string]: unknown;
  };
  position: { x: number; y: number };
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
}

export interface GraphResponse {
  model_id: string;
  nodes: GraphNode[];
  edges: GraphEdge[];
  metadata: {
    extraction_method: string;
    num_nodes: number;
    num_edges: number;
  };
}

export interface TensorStats {
  shape: number[];
  dtype: string;
  numel: number;
  mean: number;
  std: number;
  min: number;
  max: number;
  p1: number;
  p5: number;
  p25: number;
  p50: number;
  p75: number;
  p95: number;
  p99: number;
  zeroCount: number;
  zeroRatio: number;
  nanCount: number;
  infCount: number;
}

export interface Histogram {
  bins: number[];
  counts: number[];
  binCenters: number[];
}

export interface WeightStatsResponse {
  model_id: string;
  layer_name: string;
  stats: TensorStats;
  histogram: Histogram;
}

export interface PerChannelStats {
  channelMaxes: number[];
  channelMins: number[];
  channelMeans: number[];
  channelStds: number[];
  numChannels: number;
  outlierIndices: number[];
}

// API functions
export const modelsApi = {
  list: () => api.get<ModelListResponse>('/models/'),
  load: (modelId: string) => api.post(`/models/${modelId}/load`),
  unload: (modelId: string) => api.delete(`/models/${modelId}`),
  getLayers: (modelId: string) =>
    api.get<{ model_id: string; total_layers: number; layers: LayerInfo[] }>(
      `/models/${modelId}/layers`
    ),
};

export const graphApi = {
  get: (modelId: string, options?: { maxDepth?: number; filterMode?: 'all' | 'leaves' }) =>
    api.get<GraphResponse>(`/graph/${modelId}`, { params: options }),
};

export const weightsApi = {
  getStats: (modelId: string, layerName: string, options?: { numBins?: number }) =>
    api.get<WeightStatsResponse>(`/weights/${modelId}/${layerName}/stats`, {
      params: options,
    }),
  getHeatmap: (modelId: string, layerName: string, options?: { maxSize?: number }) =>
    api.get<{
      model_id: string;
      layer_name: string;
      data: number[][];
      originalShape: number[];
      displayedShape: number[];
      min: number;
      max: number;
    }>(`/weights/${modelId}/${layerName}/heatmap`, { params: options }),
  getPerChannel: (modelId: string, layerName: string) =>
    api.get<{
      model_id: string;
      layer_name: string;
      stats: PerChannelStats;
    }>(`/weights/${modelId}/${layerName}/per-channel`),
};
