import React, { useEffect, useMemo, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import {
  Box,
  Compass,
  Eye,
  Focus,
  Layers,
  Maximize2,
  RotateCcw,
  Sparkles,
} from 'lucide-react';
import {
  MatrixKind,
  ModelArchitectureProfile,
  PulseSignal,
  TensorMatrixState,
  ThreeLayoutMode,
} from '../types/olmo';

interface ThreeWeightEngineProps {
  architecture: ModelArchitectureProfile;
  matrices: TensorMatrixState[];
  selectedMatrixId: string;
  onSelectMatrix: (id: string) => void;
  layoutMode: ThreeLayoutMode;
  onChangeLayoutMode: (mode: ThreeLayoutMode) => void;
  pulseSignal: PulseSignal | null;
  highlightKinds?: MatrixKind[];
}

interface HoveredInfo {
  matrix: TensorMatrixState;
  x: number;
  y: number;
}

const VOXELS_PER_MATRIX = 64; // 8x8 sampled 3D weight voxels per matrix

function weightColor(
  w: number,
  delta: number,
  ablated: boolean,
  category: TensorMatrixState['category'],
  outColor: THREE.Color
) {
  if (ablated) {
    outColor.setRGB(0.18, 0.2, 0.25);
    return;
  }
  if (Math.abs(delta) > 0.08) {
    const boost = Math.min(1, Math.abs(delta) * 1.8);
    if (w >= 0) {
      outColor.setRGB(0.15 + boost * 0.15, 0.78 + boost * 0.22, 0.58 + boost * 0.12);
    } else {
      outColor.setRGB(0.12, 0.65 + boost * 0.3, 0.92);
    }
    return;
  }

  const clamped = Math.max(-1.2, Math.min(1.2, w));
  if (category === 'vision') {
    // Warm amber / violet palette for Vision ViT patches
    if (clamped >= 0) {
      const t = clamped / 1.2;
      outColor.setRGB(0.45 + t * 0.52, 0.32 + t * 0.42, 0.12 + t * 0.15);
    } else {
      const t = -clamped / 1.2;
      outColor.setRGB(0.35 + t * 0.3, 0.18 + t * 0.15, 0.55 + t * 0.42);
    }
    return;
  }
  if (category === 'connector' || category === 'router') {
    // Luminous emerald / magenta for Cross-Modal Connector & MoE Router
    const t = Math.abs(clamped) / 1.2;
    outColor.setRGB(0.2 + t * 0.7, 0.55 + t * 0.35, 0.45 + t * 0.45);
    return;
  }

  // Standard LLM polarity: negative = electric cyan, positive = Ai2 pink/coral
  if (clamped >= 0) {
    const t = clamped / 1.2;
    outColor.setRGB(0.24 + t * 0.74, 0.22 + t * 0.18, 0.38 + t * 0.28);
  } else {
    const t = -clamped / 1.2;
    outColor.setRGB(0.12 + t * 0.06, 0.28 + t * 0.48, 0.46 + t * 0.5);
  }
}

function getMatrixWorldTransform(
  m: TensorMatrixState,
  mode: ThreeLayoutMode,
  numLayers: number
): { pos: THREE.Vector3; size: THREE.Vector3; rotY: number } {
  const maxLayer = Math.max(1, numLayers - 1);

  // Detect GQA compression ratio (e.g. Qwen2.5-Coder-7B has 512x3584 K/V vs 3584x3584 Q/O)
  const isGqaCompressed =
    (m.kind === 'k_proj' || m.kind === 'v_proj') && m.shape[0] < m.shape[1] * 0.5;
  const gqaFactor = isGqaCompressed ? Math.max(0.35, m.shape[0] / m.shape[1]) : 1.0;

  let sx = 1.05;
  let sy = 0.44;
  let sz = 1.05;

  if (m.kind === 'embed' || m.kind === 'lm_head') {
    sx = 5.4;
    sy = 0.65;
    sz = 3.2;
  } else if (m.category === 'vision') {
    sx = 1.55;
    sy = 0.48;
    sz = 1.55;
  } else if (m.category === 'connector') {
    sx = 3.4;
    sy = 0.58;
    sz = 1.8;
  } else if (m.category === 'router') {
    sx = 0.85;
    sy = 0.52;
    sz = 0.85;
  } else if (m.category === 'attention') {
    sx = 1.05 * gqaFactor;
    sy = 0.46;
    sz = 1.05;
  } else {
    // SwiGLU MLP matrices (wider for Coding AI 18,944 intermediate size)
    const mlpWidthBoost = m.shape[0] > 14000 || m.shape[1] > 14000 ? 1.95 : 1.65;
    sx = mlpWidthBoost;
    sy = 0.56;
    sz = 1.25;
  }

  if (mode === 'helix') {
    // Multimodal Vision ViT Tower sits below the Language Embedding
    if (m.category === 'vision') {
      const vIdx = m.layerIndex + 10; // 0..6
      const angle = vIdx * 0.72 + (m.kind === 'vision_mlp' ? 0.32 : 0);
      const r = m.kind === 'vision_patch' ? 0 : 3.2;
      const y = -18.6 + vIdx * 0.52;
      return {
        pos: new THREE.Vector3(Math.cos(angle) * r, y, Math.sin(angle) * r),
        size: new THREE.Vector3(sx, sy, sz),
        rotY: -angle,
      };
    }
    if (m.category === 'connector') {
      const offsetX = m.name.includes('pooling') ? -2.1 : 2.1;
      return {
        pos: new THREE.Vector3(offsetX, -15.3, 0),
        size: new THREE.Vector3(2.5, 0.58, 1.8),
        rotY: 0,
      };
    }
    if (m.kind === 'embed') {
      return { pos: new THREE.Vector3(0, -13.6, 0), size: new THREE.Vector3(sx, sy, sz), rotY: 0 };
    }
    if (m.kind === 'lm_head') {
      return { pos: new THREE.Vector3(0, 14.2, 0), size: new THREE.Vector3(sx, sy, sz), rotY: 0 };
    }

    const layer = Math.max(0, m.layerIndex);
    const y = -12.0 + (layer / maxLayer) * 24.4;
    const baseAngle = layer * 0.38;
    const radius = 5.4;

    const kindOffsets: Record<string, { dr: number; da: number }> = {
      router_gate: { dr: -2.8, da: 0.0 },
      q_proj: { dr: -1.45, da: -0.28 },
      k_proj: { dr: -1.45, da: -0.09 },
      v_proj: { dr: -1.45, da: 0.09 },
      o_proj: { dr: -1.45, da: 0.28 },
      gate_proj: { dr: 1.55, da: -0.24 },
      up_proj: { dr: 1.55, da: 0.0 },
      down_proj: { dr: 1.55, da: 0.24 },
    };

    const off = kindOffsets[m.kind] || { dr: 0, da: 0 };
    const r = radius + off.dr;
    const angle = baseAngle + off.da;

    return {
      pos: new THREE.Vector3(Math.cos(angle) * r, y, Math.sin(angle) * r),
      size: new THREE.Vector3(sx, sy, sz),
      rotY: -angle,
    };
  }

  // 'stack' (Linear Architectural Pipeline along X)
  if (m.category === 'vision') {
    const vIdx = m.layerIndex + 10;
    const x = -33.5 + vIdx * 1.1;
    const z = m.kind === 'vision_patch' ? 0 : m.kind === 'vision_attn' ? -2.2 : 2.2;
    return {
      pos: new THREE.Vector3(x, 0, z),
      size: new THREE.Vector3(0.9, 1.8, 1.8),
      rotY: 0,
    };
  }
  if (m.category === 'connector') {
    const z = m.name.includes('pooling') ? -2.2 : 2.2;
    return {
      pos: new THREE.Vector3(-27.2, 0.2, z),
      size: new THREE.Vector3(1.4, 2.8, 3.2),
      rotY: 0,
    };
  }
  if (m.kind === 'embed') {
    return { pos: new THREE.Vector3(-24.8, 0, 0), size: new THREE.Vector3(2.0, 5.2, 9.5), rotY: 0 };
  }
  if (m.kind === 'lm_head') {
    return { pos: new THREE.Vector3(25.2, 0, 0), size: new THREE.Vector3(2.0, 5.2, 9.5), rotY: 0 };
  }

  const x = -22.0 + (Math.max(0, m.layerIndex) / maxLayer) * 44.0;
  const zOrder: Record<string, number> = {
    router_gate: -6.4,
    q_proj: -4.8,
    k_proj: -3.2,
    v_proj: -1.6,
    o_proj: 0.0,
    gate_proj: 1.9,
    up_proj: 3.7,
    down_proj: 5.5,
  };
  const z = zOrder[m.kind] ?? 0;
  const y = m.category === 'mlp' ? 0.55 : -0.45;
  return {
    pos: new THREE.Vector3(
      x,
      y,
      z
    ),
    size: new THREE.Vector3(
      0.95,
      m.category === 'mlp' ? 1.85 : 1.15 * gqaFactor,
      m.category === 'mlp' ? 1.35 : 1.15 * gqaFactor
    ),
    rotY: 0,
  };
}

export const ThreeWeightEngine: React.FC<ThreeWeightEngineProps> = ({
  architecture,
  matrices,
  selectedMatrixId,
  onSelectMatrix,
  layoutMode,
  onChangeLayoutMode,
  pulseSignal,
  highlightKinds = [],
}) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [hovered, setHovered] = useState<HoveredInfo | null>(null);
  const [webglError, setWebglError] = useState<string | null>(null);

  const selectedMatrix = useMemo(
    () => matrices.find((m) => m.id === selectedMatrixId) ?? matrices[0],
    [matrices, selectedMatrixId]
  );

  const archRef = useRef(architecture);
  archRef.current = architecture;

  const matricesRef = useRef(matrices);
  matricesRef.current = matrices;

  const selectedIdRef = useRef(selectedMatrixId);
  selectedIdRef.current = selectedMatrixId;

  const layoutModeRef = useRef(layoutMode);
  layoutModeRef.current = layoutMode;

  const pulseRef = useRef(pulseSignal);
  pulseRef.current = pulseSignal;

  const highlightKindsRef = useRef(highlightKinds);
  highlightKindsRef.current = highlightKinds;

  const onSelectRef = useRef(onSelectMatrix);
  onSelectRef.current = onSelectMatrix;

  const cameraSnapRef = useRef<{
    targetPos: THREE.Vector3;
    targetLook: THREE.Vector3;
    active: boolean;
  }>({
    targetPos: new THREE.Vector3(16, 8, 22),
    targetLook: new THREE.Vector3(0, 0, 0),
    active: false,
  });

  const sceneObjectsRef = useRef<{
    rebuildModelMeshes: () => void;
    rebuildGeometryLayout: () => void;
    updateVoxelData: () => void;
  } | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({
        antialias: true,
        powerPreference: 'high-performance',
        alpha: false,
      });
    } catch {
      setWebglError('WebGL is unavailable in this browser environment.');
      return;
    }

    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setClearColor(0x060a12, 1);
    container.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x060a12, 0.01);

    const camera = new THREE.PerspectiveCamera(
      45,
      Math.max(1, container.clientWidth) / Math.max(1, container.clientHeight),
      0.1,
      220
    );
    camera.position.set(15, 7.5, 22);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.07;
    controls.minDistance = 4;
    controls.maxDistance = 78;
    controls.addEventListener('start', () => {
      cameraSnapRef.current.active = false;
    });

    const ambient = new THREE.AmbientLight(0xcbd5e1, 0.75);
    scene.add(ambient);

    const keyLight = new THREE.DirectionalLight(0xffffff, 1.35);
    keyLight.position.set(18, 28, 20);
    scene.add(keyLight);

    const pinkRim = new THREE.PointLight(0xf0529c, 2.2, 60);
    pinkRim.position.set(-14, 12, -10);
    scene.add(pinkRim);

    const cyanRim = new THREE.PointLight(0x22d3ee, 2.2, 60);
    cyanRim.position.set(14, -10, 12);
    scene.add(cyanRim);

    const gridHelper = new THREE.GridHelper(72, 36, 0x1e293b, 0x0f172a);
    gridHelper.position.y = -19.2;
    scene.add(gridHelper);

    const fullModelGroup = new THREE.Group();
    scene.add(fullModelGroup);

    const blockGeo = new THREE.BoxGeometry(1, 1, 1);
    const ringGeo = new THREE.TorusGeometry(5.45, 0.032, 8, 64);
    const voxelGeo = new THREE.BoxGeometry(0.11, 0.11, 0.11);
    const voxelMat = new THREE.MeshBasicMaterial({ color: 0xffffff });

    const blocks = new Map<string, THREE.Mesh>();
    let rings: THREE.Mesh[] = [];
    let voxelInstanced: THREE.InstancedMesh | null = null;
    let blockArray: THREE.Mesh[] = [];

    // Selection wireframe halo
    const wireGeo = new THREE.EdgesGeometry(new THREE.BoxGeometry(1.08, 1.18, 1.08));
    const wireMat = new THREE.LineBasicMaterial({
      color: 0x38bdf8,
      transparent: true,
      opacity: 0.95,
    });
    const selectionWire = new THREE.LineSegments(wireGeo, wireMat);
    fullModelGroup.add(selectionWire);

    // Signal pulse spheres
    const pulseGroup = new THREE.Group();
    const pulseSpheres: THREE.Mesh[] = [];
    const sphereGeo = new THREE.SphereGeometry(0.24, 14, 14);
    for (let i = 0; i < 18; i++) {
      const sMat = new THREE.MeshBasicMaterial({
        color: 0x22d3ee,
        transparent: true,
        opacity: 0.92,
      });
      const sp = new THREE.Mesh(sphereGeo, sMat);
      sp.visible = false;
      pulseGroup.add(sp);
      pulseSpheres.push(sp);
    }
    fullModelGroup.add(pulseGroup);

    // Mode 3: Sculpted 16x16 = 256 Matrix 3D Heightfield Inspector
    const matrix3dGroup = new THREE.Group();
    const m3dGeo = new THREE.BoxGeometry(0.62, 1, 0.62);
    const m3dMat = new THREE.MeshStandardMaterial({
      color: 0xffffff,
      roughness: 0.25,
      metalness: 0.3,
    });
    const matrix3dInstanced = new THREE.InstancedMesh(m3dGeo, m3dMat, 256);
    matrix3dInstanced.frustumCulled = false;
    matrix3dInstanced.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    const initColor = new THREE.Color(1, 1, 1);
    for (let i = 0; i < 256; i++) matrix3dInstanced.setColorAt(i, initColor);
    matrix3dGroup.add(matrix3dInstanced);

    const m3dPlane = new THREE.GridHelper(13, 16, 0x334155, 0x1e293b);
    matrix3dGroup.add(m3dPlane);
    matrix3dGroup.visible = false;
    scene.add(matrix3dGroup);

    const dummy = new THREE.Object3D();
    const tempColor = new THREE.Color();

    /** Dynamically (re)build block meshes, RoPE rings, and InstancedMesh when model changes */
    const rebuildModelMeshes = () => {
      blocks.forEach((mesh) => {
        fullModelGroup.remove(mesh);
        (mesh.material as THREE.Material).dispose();
      });
      blocks.clear();

      rings.forEach((ring) => {
        fullModelGroup.remove(ring);
        (ring.material as THREE.Material).dispose();
      });
      rings = [];

      if (voxelInstanced) {
        fullModelGroup.remove(voxelInstanced);
        voxelInstanced.dispose();
        voxelInstanced = null;
      }

      const list = matricesRef.current;
      const numLayers = archRef.current.numLayers;

      for (const m of list) {
        const mat = new THREE.MeshStandardMaterial({
          color: 0x1e293b,
          roughness: 0.35,
          metalness: 0.45,
          transparent: true,
          opacity: 0.42,
        });
        const mesh = new THREE.Mesh(blockGeo, mat);
        mesh.userData = { matrixId: m.id };
        fullModelGroup.add(mesh);
        blocks.set(m.id, mesh);
      }
      blockArray = Array.from(blocks.values());

      for (let l = 0; l < numLayers; l++) {
        const rMat = new THREE.MeshBasicMaterial({
          color: 0x334155,
          transparent: true,
          opacity: 0.38,
        });
        const ring = new THREE.Mesh(ringGeo, rMat);
        ring.rotation.x = Math.PI / 2;
        fullModelGroup.add(ring);
        rings.push(ring);
      }

      const totalVoxels = Math.max(1, list.length * VOXELS_PER_MATRIX);
      voxelInstanced = new THREE.InstancedMesh(voxelGeo, voxelMat, totalVoxels);
      voxelInstanced.frustumCulled = false;
      voxelInstanced.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      for (let i = 0; i < totalVoxels; i++) {
        voxelInstanced.setColorAt(i, initColor);
      }
      fullModelGroup.add(voxelInstanced);
    };

    const rebuildGeometryLayout = () => {
      const mode = layoutModeRef.current;
      const isMatrix3d = mode === 'matrix3d';
      const numLayers = archRef.current.numLayers;
      const maxLayer = Math.max(1, numLayers - 1);

      fullModelGroup.visible = !isMatrix3d;
      matrix3dGroup.visible = isMatrix3d;
      gridHelper.position.y = mode === 'helix' ? -19.2 : -5.5;

      if (!isMatrix3d) {
        for (let l = 0; l < rings.length; l++) {
          const ring = rings[l];
          if (mode === 'helix') {
            ring.visible = true;
            ring.position.set(0, -12.0 + (l / maxLayer) * 24.4, 0);
            ring.rotation.set(Math.PI / 2, 0, 0);
          } else {
            ring.visible = false;
          }
        }

        for (const m of matricesRef.current) {
          const mesh = blocks.get(m.id);
          if (!mesh) continue;
          const t = getMatrixWorldTransform(m, mode, numLayers);
          mesh.position.copy(t.pos);
          mesh.scale.copy(t.size);
          mesh.rotation.set(0, t.rotY, 0);
        }
      }
    };

    const updateVoxelData = () => {
      const mode = layoutModeRef.current;
      const list = matricesRef.current;
      const selId = selectedIdRef.current;
      const highlights = highlightKindsRef.current;
      const numLayers = archRef.current.numLayers;

      // If matrix count changed due to switching models or uploading a file, rebuild meshes first
      if (blocks.size !== list.length || !voxelInstanced) {
        rebuildModelMeshes();
        rebuildGeometryLayout();
      }

      if (mode === 'matrix3d') {
        const target = list.find((m) => m.id === selId) ?? list[0];
        if (!target) return;
        for (let r = 0; r < 16; r++) {
          for (let c = 0; c < 16; c++) {
            const idx = r * 16 + c;
            const w = target.ablated ? 0 : target.weights[idx];
            const delta = w - target.baseWeights[idx];
            const height = Math.max(0.08, Math.abs(w) * 3.6);
            dummy.position.set(
              (c - 7.5) * 0.78,
              w >= 0 ? height / 2 : -height / 2,
              (r - 7.5) * 0.78
            );
            dummy.rotation.set(0, 0, 0);
            dummy.scale.set(1, height, 1);
            dummy.updateMatrix();
            matrix3dInstanced.setMatrixAt(idx, dummy.matrix);

            weightColor(w, delta, target.ablated, target.category, tempColor);
            matrix3dInstanced.setColorAt(idx, tempColor);
          }
        }
        matrix3dInstanced.instanceMatrix.needsUpdate = true;
        if (matrix3dInstanced.instanceColor) matrix3dInstanced.instanceColor.needsUpdate = true;
        return;
      }

      if (!voxelInstanced) return;

      let voxelCursor = 0;
      for (const m of list) {
        const mesh = blocks.get(m.id);
        const t = getMatrixWorldTransform(m, mode, numLayers);

        if (mesh) {
          const mat = mesh.material as THREE.MeshStandardMaterial;
          const isSel = m.id === selId;
          const isHi = highlights.length > 0 && highlights.includes(m.kind);
          const hasDelta = m.deltaL2 > 0.04;

          if (m.ablated) {
            mat.color.setHex(0x1e293b);
            mat.emissive.setHex(0xef4444);
            mat.emissiveIntensity = 0.25;
            mat.opacity = 0.22;
          } else if (isSel) {
            mat.color.setHex(0x38bdf8);
            mat.emissive.setHex(0x0ea5e9);
            mat.emissiveIntensity = 0.55;
            mat.opacity = 0.62;
          } else if (isHi) {
            mat.color.setHex(0xf0529c);
            mat.emissive.setHex(0xf0529c);
            mat.emissiveIntensity = 0.42;
            mat.opacity = 0.55;
          } else if (hasDelta) {
            mat.color.setHex(0x10b981);
            mat.emissive.setHex(0x059669);
            mat.emissiveIntensity = Math.min(0.65, 0.15 + m.deltaL2 * 0.12);
            mat.opacity = 0.48;
          } else if (m.category === 'vision') {
            mat.color.setHex(0x45271d);
            mat.emissive.setHex(0xf59e0b);
            mat.emissiveIntensity = 0.16;
            mat.opacity = 0.42;
          } else if (m.category === 'connector' || m.category === 'router') {
            mat.color.setHex(0x134e4a);
            mat.emissive.setHex(0x10b981);
            mat.emissiveIntensity = 0.22;
            mat.opacity = 0.48;
          } else if (m.category === 'attention') {
            mat.color.setHex(0x1e3a5f);
            mat.emissive.setHex(0x0284c7);
            mat.emissiveIntensity = 0.08;
            mat.opacity = 0.34;
          } else if (m.category === 'mlp') {
            mat.color.setHex(0x3b1d45);
            mat.emissive.setHex(0xdb2777);
            mat.emissiveIntensity = 0.08;
            mat.opacity = 0.34;
          } else {
            mat.color.setHex(0x1e293b);
            mat.emissive.setHex(0x6366f1);
            mat.emissiveIntensity = 0.15;
            mat.opacity = 0.4;
          }

          if (isSel) {
            selectionWire.position.copy(t.pos);
            selectionWire.scale.set(t.size.x * 1.12, t.size.y * 1.24, t.size.z * 1.12);
            selectionWire.rotation.set(0, t.rotY, 0);
          }
        }

        const cosR = Math.cos(t.rotY);
        const sinR = Math.sin(t.rotY);
        const stepX = (t.size.x * 0.82) / 7;
        const stepZ = (t.size.z * 0.82) / 7;

        for (let r = 0; r < 8; r++) {
          for (let c = 0; c < 8; c++) {
            const wIdx = r * 2 * 16 + c * 2;
            const w = m.ablated ? 0 : m.weights[wIdx];
            const delta = w - m.baseWeights[wIdx];

            const lx = (c - 3.5) * stepX;
            const lz = (r - 3.5) * stepZ;
            const ly = w * t.size.y * 0.32;

            const wx = t.pos.x + lx * cosR + lz * sinR;
            const wz = t.pos.z - lx * sinR + lz * cosR;
            const wy = t.pos.y + ly;

            const vScale = m.ablated ? 0.35 : 0.72 + Math.min(1.5, Math.abs(w) * 1.1);
            dummy.position.set(wx, wy, wz);
            dummy.rotation.set(0, t.rotY, 0);
            dummy.scale.set(vScale, vScale, vScale);
            dummy.updateMatrix();
            voxelInstanced.setMatrixAt(voxelCursor, dummy.matrix);

            weightColor(w, delta, m.ablated, m.category, tempColor);
            voxelInstanced.setColorAt(voxelCursor, tempColor);
            voxelCursor++;
          }
        }
      }

      voxelInstanced.instanceMatrix.needsUpdate = true;
      if (voxelInstanced.instanceColor) voxelInstanced.instanceColor.needsUpdate = true;
    };

    sceneObjectsRef.current = {
      rebuildModelMeshes,
      rebuildGeometryLayout,
      updateVoxelData,
    };

    rebuildModelMeshes();
    rebuildGeometryLayout();
    updateVoxelData();

    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();
    let pointerDownPos = { x: 0, y: 0 };

    const handlePointerMove = (ev: PointerEvent) => {
      if (layoutModeRef.current === 'matrix3d') {
        setHovered(null);
        return;
      }
      const rect = renderer.domElement.getBoundingClientRect();
      pointer.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;

      raycaster.setFromCamera(pointer, camera);
      const hits = raycaster.intersectObjects(blockArray, false);
      if (hits.length > 0) {
        const hitId = hits[0].object.userData.matrixId as string;
        const found = matricesRef.current.find((m) => m.id === hitId);
        if (found) {
          renderer.domElement.style.cursor = 'pointer';
          setHovered({
            matrix: found,
            x: ev.clientX - rect.left,
            y: ev.clientY - rect.top,
          });
          return;
        }
      }
      renderer.domElement.style.cursor = 'grab';
      setHovered(null);
    };

    const handlePointerDown = (ev: PointerEvent) => {
      pointerDownPos = { x: ev.clientX, y: ev.clientY };
    };

    const handlePointerUp = (ev: PointerEvent) => {
      const dist = Math.hypot(ev.clientX - pointerDownPos.x, ev.clientY - pointerDownPos.y);
      if (dist > 6 || layoutModeRef.current === 'matrix3d') return;
      const rect = renderer.domElement.getBoundingClientRect();
      pointer.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);
      const hits = raycaster.intersectObjects(blockArray, false);
      if (hits.length > 0) {
        const hitId = hits[0].object.userData.matrixId as string;
        onSelectRef.current(hitId);
      }
    };

    renderer.domElement.addEventListener('pointermove', handlePointerMove);
    renderer.domElement.addEventListener('pointerdown', handlePointerDown);
    renderer.domElement.addEventListener('pointerup', handlePointerUp);

    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        if (width > 0 && height > 0) {
          camera.aspect = width / height;
          camera.updateProjectionMatrix();
          renderer.setSize(width, height);
        }
      }
    });
    ro.observe(container);

    let animId = 0;
    const clock = new THREE.Clock();

    const animate = () => {
      animId = requestAnimationFrame(animate);
      const elapsed = clock.getElapsedTime();

      if (cameraSnapRef.current.active) {
        camera.position.lerp(cameraSnapRef.current.targetPos, 0.085);
        controls.target.lerp(cameraSnapRef.current.targetLook, 0.085);
        if (camera.position.distanceTo(cameraSnapRef.current.targetPos) < 0.12) {
          cameraSnapRef.current.active = false;
        }
      }

      if (layoutModeRef.current === 'helix') {
        selectionWire.material.opacity = 0.65 + Math.sin(elapsed * 4.0) * 0.3;
      }

      const pulse = pulseRef.current;
      const now = performance.now();
      if (pulse && now - pulse.startedAt < pulse.durationMs && layoutModeRef.current !== 'matrix3d') {
        const progress = (now - pulse.startedAt) / pulse.durationMs;
        const pColor = new THREE.Color(pulse.color);
        const mode = layoutModeRef.current;
        const isVlm = archRef.current.family === 'multimodal-vlm';

        pulseSpheres.forEach((sp, idx) => {
          const offsetProgress = Math.max(0, Math.min(1, progress * 1.25 - idx * 0.025));
          const layerFrac =
            pulse.direction === 'forward' ? offsetProgress : 1 - offsetProgress;
          const layerFloat =
            pulse.layerStart + (pulse.layerEnd - pulse.layerStart) * layerFrac;

          sp.visible = offsetProgress > 0 && offsetProgress < 1;
          (sp.material as THREE.MeshBasicMaterial).color.copy(pColor);

          if (mode === 'helix') {
            const minY = isVlm ? -18.5 : -13.0;
            const spanY = isVlm ? 32.5 : 27.0;
            const y = minY + layerFrac * spanY;
            const angle = layerFloat * 0.38 + idx * 0.35;
            const r = 3.6 + (idx % 3) * 1.4;
            sp.position.set(Math.cos(angle) * r, y, Math.sin(angle) * r);
          } else {
            const minX = isVlm ? -33.0 : -24.5;
            const spanX = isVlm ? 58.0 : 49.0;
            const x = minX + layerFrac * spanX;
            const z = ((idx % 7) - 3) * 1.5;
            sp.position.set(x, Math.sin(progress * Math.PI * 4 + idx) * 0.8, z);
          }
        });

        rings.forEach((ring, l) => {
          const ringFrac = l / Math.max(1, rings.length - 1);
          const waveFrac = pulse.direction === 'forward' ? progress : 1 - progress;
          const dist = Math.abs(ringFrac - waveFrac);
          const rMat = ring.material as THREE.MeshBasicMaterial;
          if (dist < 0.14) {
            rMat.color.copy(pColor);
            rMat.opacity = 0.95;
          } else {
            rMat.color.setHex(0x334155);
            rMat.opacity = 0.35;
          }
        });
      } else {
        pulseSpheres.forEach((sp) => (sp.visible = false));
        rings.forEach((ring) => {
          const rMat = ring.material as THREE.MeshBasicMaterial;
          rMat.color.setHex(0x334155);
          rMat.opacity = 0.35;
        });
      }

      controls.update();
      renderer.render(scene, camera);
    };

    animate();

    return () => {
      cancelAnimationFrame(animId);
      ro.disconnect();
      renderer.domElement.removeEventListener('pointermove', handlePointerMove);
      renderer.domElement.removeEventListener('pointerdown', handlePointerDown);
      renderer.domElement.removeEventListener('pointerup', handlePointerUp);
      controls.dispose();
      blockGeo.dispose();
      ringGeo.dispose();
      voxelGeo.dispose();
      voxelMat.dispose();
      wireGeo.dispose();
      wireMat.dispose();
      sphereGeo.dispose();
      m3dGeo.dispose();
      m3dMat.dispose();
      blocks.forEach((m) => (m.material as THREE.Material).dispose());
      rings.forEach((r) => (r.material as THREE.Material).dispose());
      pulseSpheres.forEach((s) => (s.material as THREE.Material).dispose());
      if (voxelInstanced) voxelInstanced.dispose();
      renderer.dispose();
      if (renderer.domElement.parentNode === container) {
        container.removeChild(renderer.domElement);
      }
    };
  }, []);

  useEffect(() => {
    const obj = sceneObjectsRef.current;
    if (!obj) return;
    obj.updateVoxelData();
  }, [architecture, matrices, selectedMatrixId, highlightKinds, layoutMode]);

  useEffect(() => {
    const obj = sceneObjectsRef.current;
    if (!obj) return;
    obj.rebuildGeometryLayout();
    obj.updateVoxelData();

    if (layoutMode === 'helix') {
      cameraSnapRef.current = {
        targetPos: new THREE.Vector3(16.5, 6.5, 24),
        targetLook: new THREE.Vector3(0, architecture.family === 'multimodal-vlm' ? -2 : 0, 0),
        active: true,
      };
    } else if (layoutMode === 'stack') {
      cameraSnapRef.current = {
        targetPos: new THREE.Vector3(0, 16, 38),
        targetLook: new THREE.Vector3(0, 0, 0),
        active: true,
      };
    } else {
      cameraSnapRef.current = {
        targetPos: new THREE.Vector3(11.5, 9.5, 13.5),
        targetLook: new THREE.Vector3(0, 0, 0),
        active: true,
      };
    }
  }, [layoutMode, architecture.id, architecture.family]);

  const handleFocusSelected = () => {
    if (layoutMode === 'matrix3d') {
      cameraSnapRef.current = {
        targetPos: new THREE.Vector3(11.5, 9.5, 13.5),
        targetLook: new THREE.Vector3(0, 0, 0),
        active: true,
      };
      return;
    }
    const t = getMatrixWorldTransform(selectedMatrix, layoutMode, architecture.numLayers);
    cameraSnapRef.current = {
      targetPos: new THREE.Vector3(t.pos.x + 6.5, t.pos.y + 3.8, t.pos.z + 8.5),
      targetLook: t.pos.clone(),
      active: true,
    };
  };

  const handleResetCamera = () => {
    if (layoutMode === 'helix') {
      cameraSnapRef.current = {
        targetPos: new THREE.Vector3(16.5, 6.5, 24),
        targetLook: new THREE.Vector3(0, architecture.family === 'multimodal-vlm' ? -2 : 0, 0),
        active: true,
      };
    } else if (layoutMode === 'stack') {
      cameraSnapRef.current = {
        targetPos: new THREE.Vector3(0, 16, 38),
        targetLook: new THREE.Vector3(0, 0, 0),
        active: true,
      };
    } else {
      cameraSnapRef.current = {
        targetPos: new THREE.Vector3(11.5, 9.5, 13.5),
        targetLook: new THREE.Vector3(0, 0, 0),
        active: true,
      };
    }
  };

  return (
    <div className="relative h-full w-full overflow-hidden rounded-xl border border-slate-800/90 bg-[#060a12]">
      {webglError ? (
        <div className="flex h-full items-center justify-center p-6 text-center text-sm text-rose-400">
          {webglError}
        </div>
      ) : (
        <div ref={containerRef} className="h-full w-full" />
      )}

      {/* Top-Left Engine Telemetry Overlay */}
      <div className="pointer-events-none absolute left-3 top-3 flex flex-col gap-1.5">
        <div className="pointer-events-auto inline-flex flex-wrap items-center gap-2 rounded-lg border border-slate-700/80 bg-slate-950/85 px-2.5 py-1.5 backdrop-blur-md">
          <Sparkles className="h-3.5 w-3.5 text-pink-400" />
          <span className="font-mono text-[11px] font-semibold text-slate-100">
            {architecture.name}
          </span>
          <span className="rounded bg-cyan-500/15 px-1.5 py-0.5 font-mono text-[10px] text-cyan-300">
            {matrices.length} Tensors • {(matrices.length * VOXELS_PER_MATRIX).toLocaleString()} Voxels
          </span>
          {architecture.family === 'coding-llm' && (
            <span className="rounded bg-indigo-500/20 px-1.5 py-0.5 font-mono text-[10px] text-indigo-300">
              GQA {architecture.numHeads}Q:{architecture.numKvHeads}KV • FIM
            </span>
          )}
          {architecture.family === 'multimodal-vlm' && (
            <span className="rounded bg-amber-500/20 px-1.5 py-0.5 font-mono text-[10px] text-amber-300">
              ViT-L/14 + 2×2 Connector + LLM
            </span>
          )}
        </div>

        {/* Layout Mode Switcher */}
        <div className="pointer-events-auto inline-flex w-fit items-center gap-1 rounded-lg border border-slate-800 bg-slate-950/85 p-1 backdrop-blur-md">
          <button
            onClick={() => onChangeLayoutMode('helix')}
            className={`flex items-center gap-1 rounded px-2 py-1 text-[11px] font-medium transition ${
              layoutMode === 'helix'
                ? 'bg-pink-500/20 text-pink-300 ring-1 ring-pink-500/40'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Compass className="h-3 w-3" />
            {architecture.numLayers}-Layer Helix
          </button>
          <button
            onClick={() => onChangeLayoutMode('stack')}
            className={`flex items-center gap-1 rounded px-2 py-1 text-[11px] font-medium transition ${
              layoutMode === 'stack'
                ? 'bg-cyan-500/20 text-cyan-300 ring-1 ring-cyan-500/40'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Layers className="h-3 w-3" />
            Linear Pipeline
          </button>
          <button
            onClick={() => onChangeLayoutMode('matrix3d')}
            className={`flex items-center gap-1 rounded px-2 py-1 text-[11px] font-medium transition ${
              layoutMode === 'matrix3d'
                ? 'bg-emerald-500/20 text-emerald-300 ring-1 ring-emerald-500/40'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Box className="h-3 w-3" />
            Selected Matrix 3D
          </button>
        </div>
      </div>

      {/* Top-Right Selected Tensor Badge & Camera Snap Actions */}
      <div className="absolute right-3 top-3 flex flex-col items-end gap-1.5">
        <div className="flex items-center gap-1 rounded-lg border border-slate-800 bg-slate-950/85 p-1 backdrop-blur-md">
          <button
            onClick={handleFocusSelected}
            title="Fly camera to selected tensor block"
            className="flex items-center gap-1 rounded px-2 py-1 text-[11px] text-slate-300 hover:bg-slate-800 hover:text-cyan-300"
          >
            <Focus className="h-3.5 w-3.5" />
            Focus Selected
          </button>
          <button
            onClick={handleResetCamera}
            title="Reset 3D orbit camera"
            className="flex items-center gap-1 rounded px-2 py-1 text-[11px] text-slate-300 hover:bg-slate-800 hover:text-white"
          >
            <RotateCcw className="h-3.5 w-3.5" />
            Reset View
          </button>
        </div>

        {selectedMatrix && (
          <div className="rounded-lg border border-slate-800/90 bg-slate-950/85 px-2.5 py-1.5 text-right backdrop-blur-md">
            <div className="flex items-center justify-end gap-1.5 font-mono text-[10px] text-cyan-300">
              <Eye className="h-3 w-3" />
              <span>{selectedMatrix.name}</span>
            </div>
            <div className="mt-0.5 font-mono text-[10px] text-slate-400">
              [{selectedMatrix.shape[0].toLocaleString()} × {selectedMatrix.shape[1].toLocaleString()}] •{' '}
              {(selectedMatrix.paramCount / 1e6).toFixed(2)}M params • ΔL2:{' '}
              <span className={selectedMatrix.deltaL2 > 0.02 ? 'text-emerald-400' : 'text-slate-300'}>
                {selectedMatrix.deltaL2.toFixed(3)}
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Bottom-Left Color Legend */}
      <div className="pointer-events-none absolute bottom-3 left-3 flex flex-wrap items-center gap-3 rounded-lg border border-slate-800/90 bg-slate-950/85 px-3 py-1.5 text-[10px] text-slate-300 backdrop-blur-md">
        <span className="flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-sm bg-cyan-400" />
          -w (Attn/GQA)
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-sm bg-pink-500" />
          +w (SwiGLU)
        </span>
        {architecture.family === 'multimodal-vlm' && (
          <span className="flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-sm bg-amber-400" />
            ViT & Connector
          </span>
        )}
        <span className="flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-sm bg-emerald-400" />
          Fine-Tuned ΔW
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-sm border border-rose-500/80 bg-slate-800" />
          Ablated
        </span>
      </div>

      {/* Bottom-Right Interactive Tip */}
      <div className="pointer-events-none absolute bottom-3 right-3 hidden items-center gap-1.5 rounded-lg border border-slate-800/80 bg-slate-950/80 px-2.5 py-1 text-[10px] text-slate-400 backdrop-blur-md sm:flex">
        <Maximize2 className="h-3 w-3 text-slate-500" />
        Drag to orbit • Scroll to zoom • Click any tensor block to inspect & edit weights
      </div>

      {/* Hover Raycast Tooltip */}
      {hovered && (
        <div
          className="pointer-events-none absolute z-30 w-64 rounded-lg border border-cyan-500/40 bg-slate-950/95 p-2.5 shadow-xl backdrop-blur-md"
          style={{
            left: Math.min(hovered.x + 14, (containerRef.current?.clientWidth || 600) - 270),
            top: Math.max(12, hovered.y - 75),
          }}
        >
          <div className="font-mono text-[11px] font-semibold text-cyan-300 break-all">
            {hovered.matrix.name}
          </div>
          <div className="mt-1 grid grid-cols-2 gap-x-2 gap-y-0.5 font-mono text-[10px] text-slate-300">
            <span className="text-slate-500">Category:</span>
            <span className="uppercase text-pink-300">{hovered.matrix.category}</span>
            <span className="text-slate-500">Shape:</span>
            <span>
              [{hovered.matrix.shape[0].toLocaleString()}, {hovered.matrix.shape[1].toLocaleString()}]
            </span>
            <span className="text-slate-500">Parameters:</span>
            <span>{(hovered.matrix.paramCount / 1e6).toFixed(2)}M</span>
            <span className="text-slate-500">L2 / Std:</span>
            <span>
              {hovered.matrix.l2Norm.toFixed(2)} / {hovered.matrix.std.toFixed(3)}
            </span>
            <span className="text-slate-500">Fine-Tune ΔW:</span>
            <span className={hovered.matrix.deltaL2 > 0.02 ? 'text-emerald-400' : 'text-slate-400'}>
              {hovered.matrix.deltaL2.toFixed(4)}
            </span>
          </div>
        </div>
      )}
    </div>
  );
};
