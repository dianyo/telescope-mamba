import axios from 'axios';

const API_BASE = '/api';

export const api = axios.create({
  baseURL: API_BASE,
});

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

export const modelsApi = {
  list: () => api.get<ModelListResponse>('/models/'),
  load: (id: string) => api.post(`/models/${id}/load`),
};

export const graphApi = {
  get: (id: string) => api.get(`/graph/${id}`),
};

export const weightsApi = {
  getStats: (modelId: string, layer: string) =>
    api.get(`/weights/${modelId}/${layer}/stats`),
};
