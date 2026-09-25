import { create } from 'zustand';
import { apiService, connectTelemetryStream } from '../services/api';

export const useTerrainStore = create((set, get) => ({
  selectedDatasetId: 'mars_jezero_crater',
  heightmapData: null,
  terrainMetadata: null,
  
  hazardsData: null,
  
  spectralData: null,
  
  sectorAnalysis: null,
  
  telemetry: {
    systemStatus: 'IDLE',
    batteryLevel: 100,
    temperature: 0,
    coordinates: { x: 0, y: 0, z: 0 },
    lastUpdate: null,
  },

  activeLayer: 'elevation',
  isLoading: false,
  error: null,

  _telemetryCleanup: null,

  setSelectedDatasetId: async (datasetId) => {
    set({ selectedDatasetId: datasetId });
    await get().loadTerrainData(datasetId);
  },

  setActiveLayer: (layer) => set({ activeLayer: layer }),

  loadTerrainData: async (datasetId) => {
    const id = datasetId || get().selectedDatasetId;
    set({ isLoading: true, error: null });

    try {
      const [heightmapRes, hazardsRes, spectralRes] = await Promise.all([
        apiService.getTerrainHeightmap(id),
        apiService.getTerrainHazards(id),
        apiService.getSpectralData(id),
      ]);

      set({
        heightmapData: heightmapRes.elevation_m || heightmapRes,
        terrainMetadata: heightmapRes.metadata || null,
        hazardsData: hazardsRes,
        spectralData: spectralRes,
        isLoading: false,
      });
    } catch (err) {
      set({
        error: err.message || 'Dështoi ngarkimi i të dhënave të terrenit.',
        isLoading: false,
      });
    }
  },

  analyzeSector: async (sectorParams) => {
    set({ isLoading: true, error: null });

    try {
      const result = await apiService.analyzeSector(sectorParams);
      set({
        sectorAnalysis: result,
        isLoading: false,
      });
      return result;
    } catch (err) {
      set({
        error: err.message || 'Dështoi analiza e sektorit.',
        isLoading: false,
      });
      throw err;
    }
  },

  startTelemetryStream: () => {
    if (get()._telemetryCleanup) {
      get()._telemetryCleanup();
    }

    const cleanup = connectTelemetryStream(
      (data) => {
        set((state) => ({
          telemetry: {
            ...state.telemetry,
            ...data,
            lastUpdate: new Date().toISOString(),
          },
        }));
      },
      (err) => {
        set({ error: 'Gabim në lidhjen e telemetrisë në kohë reale.' });
      }
    );

    set({ _telemetryCleanup: cleanup });
  },

  stopTelemetryStream: () => {
    const cleanup = get()._telemetryCleanup;
    if (cleanup) {
      cleanup();
      set({ _telemetryCleanup: null });
    }
  },

  clearError: () => set({ error: null }),
}));