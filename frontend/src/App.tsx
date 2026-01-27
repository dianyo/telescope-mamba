import { useEffect, useState } from 'react';
import { QueryClient, QueryClientProvider, useQuery, useMutation } from '@tanstack/react-query';
import { ReactFlowProvider } from '@xyflow/react';
import { modelsApi, graphApi } from './api/client';
import { useAppStore } from './store/appStore';
import { GraphPane } from './components/GraphPane';
import { DistributionPane } from './components/DistributionPane';
import './index.css';

const queryClient = new QueryClient();

function ModelSelector() {
  const {
    availableModels,
    loadedModels,
    selectedModelId,
    isLoadingModel,
    setAvailableModels,
    setLoadedModels,
    setSelectedModelId,
    setIsLoadingModel,
    setGraph,
    setLayers,
  } = useAppStore();

  // Fetch available models
  const { data: modelsData, refetch: refetchModels } = useQuery({
    queryKey: ['models'],
    queryFn: async () => {
      const response = await modelsApi.list();
      return response.data;
    },
  });

  // Update store when data changes
  useEffect(() => {
    if (modelsData) {
      setAvailableModels(modelsData.available);
      setLoadedModels(modelsData.loaded);
    }
  }, [modelsData, setAvailableModels, setLoadedModels]);

  // Load model mutation
  const loadModelMutation = useMutation({
    mutationFn: async (modelId: string) => {
      setIsLoadingModel(true);
      const response = await modelsApi.load(modelId);
      return response.data;
    },
    onSuccess: () => {
      refetchModels();
      setIsLoadingModel(false);
    },
    onError: () => {
      setIsLoadingModel(false);
    },
  });

  // Fetch graph when model is selected and loaded
  const { data: graphData } = useQuery({
    queryKey: ['graph', selectedModelId],
    queryFn: async () => {
      if (!selectedModelId) return null;
      const response = await graphApi.get(selectedModelId);
      return response.data;
    },
    enabled: !!selectedModelId && loadedModels.includes(selectedModelId),
  });

  // Update graph in store
  useEffect(() => {
    if (graphData) {
      setGraph(graphData.nodes, graphData.edges, graphData.metadata);
    }
  }, [graphData, setGraph]);

  // Fetch layers when model is selected
  const { data: layersData } = useQuery({
    queryKey: ['layers', selectedModelId],
    queryFn: async () => {
      if (!selectedModelId) return null;
      const response = await modelsApi.getLayers(selectedModelId);
      return response.data;
    },
    enabled: !!selectedModelId && loadedModels.includes(selectedModelId),
  });

  useEffect(() => {
    if (layersData) {
      setLayers(layersData.layers);
    }
  }, [layersData, setLayers]);

  const handleModelSelect = async (modelId: string) => {
    setSelectedModelId(modelId);
    if (!loadedModels.includes(modelId)) {
      loadModelMutation.mutate(modelId);
    }
  };

  return (
    <div className="p-4 border-b border-gray-200 bg-gray-50">
      <div className="flex items-center gap-4">
        <label className="font-medium text-gray-700">Model:</label>
        <select
          className="px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          value={selectedModelId || ''}
          onChange={(e) => handleModelSelect(e.target.value)}
          disabled={isLoadingModel}
        >
          <option value="">Select a model...</option>
          {availableModels.map((model) => (
            <option key={model.id} value={model.id}>
              {model.name} {loadedModels.includes(model.id) ? '(loaded)' : ''}
            </option>
          ))}
        </select>
        {isLoadingModel && (
          <span className="text-gray-500 animate-pulse">Loading model...</span>
        )}
      </div>
    </div>
  );
}

function MainLayout() {
  const { selectedModelId, loadedModels } = useAppStore();
  const isModelLoaded = selectedModelId && loadedModels.includes(selectedModelId);

  return (
    <div className="flex flex-col h-screen">
      <header className="bg-white shadow-sm">
        <div className="px-4 py-3">
          <h1 className="text-xl font-bold text-gray-900">
            Telescope Mamba
          </h1>
          <p className="text-sm text-gray-500">
            Deep Learning Model Inspector
          </p>
        </div>
      </header>

      <ModelSelector />

      <div className="flex flex-1 overflow-hidden">
        {/* Left Pane: Graph */}
        <div className="w-1/2 border-r border-gray-200 overflow-hidden">
          {isModelLoaded ? (
            <GraphPane />
          ) : (
            <div className="flex items-center justify-center h-full text-gray-400">
              Select and load a model to view its architecture
            </div>
          )}
        </div>

        {/* Right Pane: Distribution */}
        <div className="w-1/2 overflow-hidden">
          {isModelLoaded ? (
            <DistributionPane />
          ) : (
            <div className="flex items-center justify-center h-full text-gray-400">
              Select a layer to view weight distribution
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ReactFlowProvider>
        <MainLayout />
      </ReactFlowProvider>
    </QueryClientProvider>
  );
}

export default App;
