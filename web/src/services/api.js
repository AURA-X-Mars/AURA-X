const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1';
const WS_BASE_URL = import.meta.env.VITE_WS_BASE_URL || 'ws://localhost:8000/ws';

async function handleResponse(response) {
  if (!response.ok) {
    const errorBody = await response.json().catch(() => ({}));
    const errorMessage = errorBody.detail || `Kërkesa dështoi me statusin: ${response.status}`;
    throw new Error(errorMessage);
  }
  return await response.json();
}

export const apiService = {
  /**
   * Merr të dhënat e heightmap dhe lartësive për një dataset të caktuar.
   * @param {string} datasetId 
   * @returns {Promise<Object>} Data me matricën e lartësive (elevation_m), rezolucionin, etj.
   */
  async getTerrainHeightmap(datasetId) {
    const response = await fetch(`${API_BASE_URL}/terrain/${datasetId}/heightmap`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });
    return handleResponse(response);
  },

  /**
   * @param {string} datasetId 
   * @returns {Promise<Object>} Hartat e stabilitetit, rrëshqitjeve dhe kapacitetit mbajtës.
   */
  async getTerrainHazards(datasetId) {
    const response = await fetch(`${API_BASE_URL}/terrain/${datasetId}/hazards`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });
    return handleResponse(response);
  },

  /**
   * Ekzekuton analizën e thelluar të një sektori të caktuar (ViT Segmentor + PINN Sintering).
   * @param {Object} sectorParams { sector_id, coordinates, parameters }
   * @returns {Promise<Object>} Rezultatet e segmentimit dhe vlerësimit të sinterizimit regolit.
   */
  async analyzeSector(sectorParams) {
    const response = await fetch(`${API_BASE_URL}/analyze/sector`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(sectorParams),
    });
    return handleResponse(response);
  },

  /**
   * Merr indeksin e spektroskopisë CRISM dhe të të dhënave SHARAD radar.
   * @param {string} datasetId 
   * @returns {Promise<Object>} Rezultatet e dekonvolucionit të sinjalit dhe treguesit spektralë.
   */
  async getSpectralData(datasetId) {
    const response = await fetch(`${API_BASE_URL}/spectral/${datasetId}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });
    return handleResponse(response);
  }
};

/**
 * Menaxhimi i Lidhjes WebSocket për Telemetrinë në Kohë Reale
 * @param {Function} onMessageCallback Funksioni që ekzekutohet kur vijnë të dhëna të reja telemetrike.
 * @param {Function} onErrorCallback Funksioni për trajtimin e gabimeve të lidhjes.
 * @returns {Function} Funksion për mbylljen e lidhjes (cleanup).
 */
export function connectTelemetryStream(onMessageCallback, onErrorCallback) {
  let socket = null;
  let isConnected = false;

  const connect = () => {
    socket = new WebSocket(`${WS_BASE_URL}/telemetry`);

    socket.onopen = () => {
      console.log(' [AURA-X] Lidhja WebSocket me telemetrinë u vendos.');
      isConnected = true;
    };

    socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (onMessageCallback) {
          onMessageCallback(data);
        }
      } catch (err) {
        console.error(' [AURA-X] Gabim gjatë parsasë së të dhënave WebSocket:', err);
      }
    };

    socket.onerror = (error) => {
      console.error(' [AURA-X] Gabim në WebSocket:', error);
      if (onErrorCallback) {
        onErrorCallback(error);
      }
    };

    socket.onclose = (event) => {
      console.warn(' [AURA-X] Lidhja WebSocket u mbyll. Po provohet rindërtimi...', event.reason);
      isConnected = false;

      setTimeout(() => {
        if (!isConnected) {
          connect();
        }
      }, 3000);
    };
  };

  connect();

  return () => {
    if (socket) {
      socket.close();
    }
  };
}