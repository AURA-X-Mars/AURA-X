import * as THREE from 'three';

/**
 * 
 * @param {Array<Array<number>>} heightmap
 * @param {Object} options
 * @returns {THREE.Mesh}
 */
export function buildTerrainMesh(heightmap, options = {}) {
  const {
    gridWidth = 100,
    gridDepth = 100,
    heightScale = 0.15,
    wireframe = false,
  } = options;

  if (!heightmap || !heightmap.length || !heightmap[0].length) {
    console.warn('[AURA-X] Heightmap i pavlefshëm. Po krijohet një terren bazë.');
    return createFallbackMesh(gridWidth, gridDepth);
  }

  const rows = heightmap.length;
  const cols = heightmap[0].length;

  const geometry = new THREE.PlaneGeometry(
    gridWidth,
    gridDepth,
    cols - 1,
    rows - 1
  );

  const posAttribute = geometry.attributes.position;
  const colors = [];

  let minH = Infinity;
  let maxH = -Infinity;

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const h = heightmap[r][c];
      if (h < minH) minH = h;
      if (h > maxH) maxH = h;
    }
  }

  const heightRange = maxH - minH || 1;

  let vertexIndex = 0;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const elevation = heightmap[r][c];
      
      const normalizedH = (elevation - minH) / heightRange;

      posAttribute.setZ(vertexIndex, elevation * heightScale);

      const baseColor = new THREE.Color();
      baseColor.setHSL(0.05 + normalizedH * 0.05, 0.7, 0.2 + normalizedH * 0.4);
      
      colors.push(baseColor.r, baseColor.g, baseColor.b);
      vertexIndex++;
    }
  }

  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

  geometry.computeVertexNormals();

  const material = new THREE.MeshStandardMaterial({
    vertexColors: true,
    roughness: 0.85,
    metalness: 0.1,
    wireframe: wireframe,
    side: THREE.DoubleSide,
  });

  const mesh = new THREE.Mesh(geometry, material);

  mesh.rotation.x = -Math.PI / 2;
  mesh.receiveShadow = true;
  mesh.castShadow = true;

  return mesh;
}

/**
 * 
 * @param {THREE.Mesh} mesh
 * @param {Array<Array<number>>} overlayData
 * @param {string} mode
 */
export function updateTerrainOverlay(mesh, overlayData, mode = 'elevation') {
  if (!mesh || !mesh.geometry) return;

  const geometry = mesh.geometry;
  const posAttribute = geometry.attributes.position;
  const colors = [];

  const totalVertices = posAttribute.count;

  for (let i = 0; i < totalVertices; i++) {
    const color = new THREE.Color();

    if (mode === 'hazards' && overlayData) {
      const hazardValue = overlayData[i] || 0;
      color.setHSL((1 - hazardValue) * 0.33, 0.9, 0.45); 
    } 
    else if (mode === 'minerals' && overlayData) {
      const mineralVal = overlayData[i] || 0;
      color.setHSL(0.5 + mineralVal * 0.15, 0.8, 0.3 + mineralVal * 0.4);
    } 
    else if (mode === 'sintering' && overlayData) {
      const sintVal = overlayData[i] || 0;
      color.setHSL(0.08, 0.9, 0.2 + sintVal * 0.5);
    } 
    else {
      const z = posAttribute.getZ(i);
      color.setHSL(0.05, 0.6, 0.2 + Math.abs(z) * 0.05);
    }

    colors.push(color.r, color.g, color.b);
  }

  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  geometry.attributes.color.needsUpdate = true;
}

function createFallbackMesh(width, depth) {
  const geometry = new THREE.PlaneGeometry(width, depth, 32, 32);
  const material = new THREE.MeshStandardMaterial({
    color: 0x8b4513,
    wireframe: true,
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.rotation.x = -Math.PI / 2;
  return mesh;
}