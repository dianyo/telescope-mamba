/**
 * Global application state using Zustand.
 */
import { create } from 'zustand';
import type { ModelInfo, LayerInfo, GraphNode, GraphEdge } from '../api/client';

interface AppState {
  // Model state
  availableModels: ModelInfo[];
  loadedModels: string[];
  selectedModelId: string | null;
  isLoadingModel: boolean;

  // Graph state
  graphNodes: GraphNode[];
  graphEdges: GraphEdge[];
  graphMetadata: { extraction_method: string; num_nodes: number; num_edges: number } | null;

  // Layer selection
  selectedLayers: string[];
  layers: LayerInfo[];

  // Actions
  setAvailableModels: (models: ModelInfo[]) => void;
  setLoadedModels: (models: string[]) => void;
  setSelectedModelId: (modelId: string | null) => void;
  setIsLoadingModel: (loading: boolean) => void;
  setGraph: (nodes: GraphNode[], edges: GraphEdge[], metadata: AppState['graphMetadata']) => void;
  setLayers: (layers: LayerInfo[]) => void;
  selectLayer: (layerName: string) => void;
  deselectLayer: (layerName: string) => void;
  toggleLayerSelection: (layerName: string) => void;
  clearLayerSelection: () => void;
}

export const useAppStore = create<AppState>((set, get) => ({
  // Initial state
  availableModels: [],
  loadedModels: [],
  selectedModelId: null,
  isLoadingModel: false,
  graphNodes: [],
  graphEdges: [],
  graphMetadata: null,
  selectedLayers: [],
  layers: [],

  // Actions
  setAvailableModels: (models) => set({ availableModels: models }),
  setLoadedModels: (models) => set({ loadedModels: models }),
  setSelectedModelId: (modelId) => set({ selectedModelId: modelId, selectedLayers: [] }),
  setIsLoadingModel: (loading) => set({ isLoadingModel: loading }),

  setGraph: (nodes, edges, metadata) =>
    set({ graphNodes: nodes, graphEdges: edges, graphMetadata: metadata }),

  setLayers: (layers) => set({ layers }),

  selectLayer: (layerName) =>
    set((state) => ({
      selectedLayers: state.selectedLayers.includes(layerName)
        ? state.selectedLayers
        : [...state.selectedLayers, layerName],
    })),

  deselectLayer: (layerName) =>
    set((state) => ({
      selectedLayers: state.selectedLayers.filter((l) => l !== layerName),
    })),

  toggleLayerSelection: (layerName) => {
    const { selectedLayers } = get();
    if (selectedLayers.includes(layerName)) {
      set({ selectedLayers: selectedLayers.filter((l) => l !== layerName) });
    } else {
      set({ selectedLayers: [...selectedLayers, layerName] });
    }
  },

  clearLayerSelection: () => set({ selectedLayers: [] }),
}));
