import React, { useEffect, useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Sky } from '@react-three/drei';
import * as THREE from 'three';
import { useTerrainStore } from '../store/useTerrainStore';
import { buildTerrainMesh, updateTerrainOverlay } from '../engine/terrainMesh';

function TerrainMeshComponent() {
  const meshRef = useRef();

  const heightmapData = useTerrainStore((state) => state.heightmapData);
  const activeLayer = useTerrainStore((state) => state.activeLayer);
  const hazardsData = useTerrainStore((state) => state.hazardsData);
  const spectralData = useTerrainStore((state) => state.spectralData);
  const sectorAnalysis = useTerrainStore((state) => state.sectorAnalysis);

  const terrainMesh = useMemo(() => {
    if (!heightmapData) return null;
    return buildTerrainMesh(heightmapData, {
      gridWidth: 120,
      gridDepth: 120,
      heightScale: 0.12,
      wireframe: false,
    });
  }, [heightmapData]);

  useEffect(() => {
    if (!meshRef.current) return;

    let overlayData = null;

    if (activeLayer === 'hazards' && hazardsData) {
      overlayData = hazardsData.slope_instability || hazardsData;
    } else if (activeLayer === 'minerals' && spectralData) {
      overlayData = spectralData.mineral_indices || spectralData;
    } else if (activeLayer === 'sintering' && sectorAnalysis) {
      overlayData = sectorAnalysis.sintering_evaluation || sectorAnalysis;
    }

    updateTerrainOverlay(meshRef.current, overlayData, activeLayer);
  }, [activeLayer, hazardsData, spectralData, sectorAnalysis, terrainMesh]);

  if (!terrainMesh) return null;

  return (
    <primitive
      ref={meshRef}
      object={terrainMesh}
      position={[0, -5, 0]}
    />
  );
}

export default function TerrainViewer() {
  const { 
    activeLayer, 
    setActiveLayer, 
    isLoading, 
    error, 
    telemetry 
  } = useTerrainStore();

  return (
    <div style={{ position: 'relative', width: '100%', height: '100vh', backgroundColor: '#050508' }}>
      
      {  }
      <div 
        style={{
          position: 'absolute',
          top: '20px',
          left: '20px',
          zIndex: 10,
          backgroundColor: 'rgba(15, 23, 42, 0.85)',
          padding: '16px',
          borderRadius: '12px',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          backdropFilter: 'blur(8px)',
          color: '#ffffff',
          fontFamily: 'sans-serif'
        }}
      >
        <h3 style={{ margin: '0 0 12px 0', fontSize: '16px', textTransform: 'uppercase', letterSpacing: '1px' }}>
          AURA-X Terrain Layers
        </h3>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          {[
            { id: 'elevation', label: ' Topography (Heightmap)' },
            { id: 'hazards', label: ' Geotechnical Hazards' },
            { id: 'minerals', label: ' CRISM Minerals & Ice' },
            { id: 'sintering', label: ' PINN Regolith Sintering' },
          ].map((layer) => (
            <button
              key={layer.id}
              onClick={() => setActiveLayer(layer.id)}
              style={{
                padding: '8px 12px',
                borderRadius: '6px',
                border: 'none',
                backgroundColor: activeLayer === layer.id ? '#3b82f6' : 'rgba(255, 255, 255, 0.05)',
                color: activeLayer === layer.id ? '#ffffff' : '#94a3b8',
                cursor: 'pointer',
                textAlign: 'left',
                fontWeight: activeLayer === layer.id ? 'bold' : 'normal',
                transition: 'all 0.2s ease',
              }}
            >
              {layer.label}
            </button>
          ))}
        </div>
      </div>

      {  }
      <div
        style={{
          position: 'absolute',
          top: '20px',
          right: '20px',
          zIndex: 10,
          backgroundColor: 'rgba(15, 23, 42, 0.85)',
          padding: '12px 16px',
          borderRadius: '12px',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          color: '#34d399',
          fontFamily: 'monospace',
          fontSize: '13px',
        }}
      >
        <div>STATUS: {telemetry.systemStatus || 'CONNECTING...'}</div>
        <div>BATTERY: {telemetry.batteryLevel || 100}%</div>
        <div>TEMP: {telemetry.temperature || -55}°C</div>
      </div>

      {  }
      {isLoading && (
        <div style={{
          position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)',
          zIndex: 20, color: '#38bdf8', fontSize: '18px', fontWeight: 'bold'
        }}>
          Po përpunohen të dhënat e AURA-X...
        </div>
      )}

      {error && (
        <div style={{
          position: 'absolute', bottom: '20px', left: '20px', zIndex: 20,
          backgroundColor: '#ef4444', color: '#fff', padding: '10px 16px', borderRadius: '8px'
        }}>
          {error}
        </div>
      )}

      {  }
      <Canvas shadows>
        <PerspectiveCamera makeDefault position={[0, 40, 80]} fov={50} />
        <OrbitControls 
          enableDamping 
          dampingFactor={0.05} 
          maxPolarAngle={Math.PI / 2.1} 
          minDistance={10} 
          maxDistance={150} 
        />

        {  }
        <ambientLight intensity={0.4} />
        <directionalLight 
          position={[50, 80, 30]} 
          intensity={1.2} 
          castShadow 
          shadow-mapSize-width={2048} 
          shadow-mapSize-height={2048} 
        />
        
        {  }
        <Sky 
          distance={450000} 
          sunPosition={[50, 80, 30]} 
          inclination={0} 
          azimuth={0.25} 
          mieCoefficient={0.005} 
          mieDirectionalG={0.8} 
          rayleigh={2} 
        />

        {  }
        <TerrainMeshComponent />
      </Canvas>
    </div>
  );
}