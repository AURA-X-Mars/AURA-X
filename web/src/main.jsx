import React, { useEffect, useRef } from 'react';
import ReactDOM from 'react-dom/client';
import * as THREE from 'three';
import { mockTelemetry } from './mockData';

function Canvas3D() {
  const mountRef = useRef(null);

  useEffect(() => {
    // 1. Skena, Kamera dhe Renderuesi
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({ antialias: true });

    renderer.setSize(window.innerWidth, window.innerHeight);
    mountRef.current.appendChild(renderer.domElement);

    // 2. Krijo Sferën 3D (Toka/AURA-X)
    const geometry = new THREE.SphereGeometry(2, 32, 32);
    const material = new THREE.MeshBasicMaterial({ color: 0x38bdf8, wireframe: true });
    const sphere = new THREE.Mesh(geometry, material);
    scene.add(sphere);

    camera.position.z = 5;

    // 3. Animacioni i rrotullimit
    let animationFrameId;
    const animate = () => {
      animationFrameId = requestAnimationFrame(animate);
      sphere.rotation.y += 0.008;
      sphere.rotation.x += 0.003;
      renderer.render(scene, camera);
    };
    animate();

    // 4. Pastrimi kur mbyllim faqen
    return () => {
      cancelAnimationFrame(animationFrameId);
      if (mountRef.current) mountRef.current.innerHTML = '';
    };
  }, []);

  return <div ref={mountRef} style={{ width: '100vw', height: '100vh' }} />;
}

function App() {
  return (
    <div style={{ position: 'relative', width: '100vw', height: '100vh', overflow: 'hidden', background: '#030712' }}>
      
      {/* Paneli i Telemetrisë mbi 3D */}
      <div style={{
        position: 'absolute',
        top: '20px',
        left: '20px',
        zIndex: 10,
        background: 'rgba(15, 23, 42, 0.85)',
        border: '1px solid #1e293b',
        padding: '20px',
        borderRadius: '10px',
        color: '#f8fafc',
        fontFamily: 'monospace'
      }}>
        <h2 style={{ margin: '0 0 10px 0', color: '#38bdf8' }}>AURA-X CONSOLE</h2>
        <p style={{ margin: '4px 0' }}>STATUS: <span style={{ color: '#22c55e' }}>{mockTelemetry.status}</span></p>
        <p style={{ margin: '4px 0' }}>TARGET: {mockTelemetry.satellite}</p>
        <p style={{ margin: '4px 0' }}>ALTITUDE: {mockTelemetry.altitude}</p>
      </div>

      {/* Ekranizimi 3D */}
      <Canvas3D />
    </div>
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(<App />);