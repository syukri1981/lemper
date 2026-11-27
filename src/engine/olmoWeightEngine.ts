import {
  CustomInjectedKnowledge,
  FineTuneJobConfig,
  FineTuneSample,
  FineTuneStepMetric,
  MatrixCategory,
  MatrixKind,
  ModelArchitectureProfile,
  ModelBehavioralProfile,
  QuantPrecision,
  TensorMatrixState,
  WeightPresetId,
} from '../types/olmo';
import { BUILTIN_MODELS } from './modelRegistry';

/** Seeded deterministic PRNG (Mulberry32) */
function mulberry32(seed: number) {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Box-Muller approximate Gaussian */
function gaussian(rng: () => number): number {
  const u1 = Math.max(1e-7, rng());
  const u2 = rng();
  return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
}

const MATRIX_CATEGORY_MAP: Record<MatrixKind, { category: MatrixCategory; shortLabel: string }> = {
  embed: { category: 'embedding', shortLabel: 'Embed' },
  vision_patch: { category: 'vision', shortLabel: 'ViT_Patch' },
  vision_attn: { category: 'vision', shortLabel: 'ViT_Attn' },
  vision_mlp: { category: 'vision', shortLabel: 'ViT_MLP' },
  mm_projector: { category: 'connector', shortLabel: 'MM_Proj' },
  router_gate: { category: 'router', shortLabel: 'Router' },
  q_proj: { category: 'attention', shortLabel: 'W_Q' },
  k_proj: { category: 'attention', shortLabel: 'W_K' },
  v_proj: { category: 'attention', shortLabel: 'W_V' },
  o_proj: { category: 'attention', shortLabel: 'W_O' },
  gate_proj: { category: 'mlp', shortLabel: 'Gate' },
  up_proj: { category: 'mlp', shortLabel: 'Up' },
  down_proj: { category: 'mlp', shortLabel: 'Down' },
  lm_head: { category: 'head', shortLabel: 'Head' },
};

export const LAYER_MATRIX_KINDS: MatrixKind[] = [
  'q_proj',
  'k_proj',
  'v_proj',
  'o_proj',
  'gate_proj',
  'up_proj',
  'down_proj',
];

export function recomputeMatrixStats(m: TensorMatrixState): TensorMatrixState {
  const n = m.weights.length;
  let sum = 0;
  let sq = 0;
  let deltaSq = 0;
  let nearZero = 0;

  for (let i = 0; i < n; i++) {
    const w = m.weights[i];
    const bw = m.baseWeights[i];
    sum += w;
    sq += w * w;
    const d = w - bw;
    deltaSq += d * d;
    if (Math.abs(w) < 0.015) nearZero++;
  }

  const mean = sum / n;
  const variance = Math.max(0, sq / n - mean * mean);
  const std = Math.sqrt(variance);
  const l2Norm = Math.sqrt(sq);
  const deltaL2 = Math.sqrt(deltaSq);
  const sparsity = nearZero / n;

  return {
    ...m,
    mean,
    std,
    l2Norm,
    deltaL2,
    sparsity,
  };
}

export function createSyntheticMatrix(
  layerIndex: number,
  kind: MatrixKind,
  name: string,
  shape: [number, number],
  seedSalt = 0,
  realExtractedPatch?: Float32Array
): TensorMatrixState {
  const meta = MATRIX_CATEGORY_MAP[kind];
  const paramCount = shape[0] * shape[1];
  const seed =
    (layerIndex + 37) * 7919 +
    seedSalt * 131 +
    name.split('').reduce((acc, c) => acc * 31 + c.charCodeAt(0), 11);
  const rng = mulberry32(seed);

  const weights = new Float32Array(256);
  const baseWeights = new Float32Array(256);

  if (realExtractedPatch && realExtractedPatch.length === 256) {
    for (let i = 0; i < 256; i++) {
      const v = Math.max(-1.8, Math.min(1.8, realExtractedPatch[i]));
      weights[i] = v;
      baseWeights[i] = v;
    }
  } else {
    const depthFactor = layerIndex >= 0 && layerIndex < 48 ? 1.0 - (layerIndex / 48) * 0.14 : 1.0;
    let baseStd = 0.24 * depthFactor;
    if (kind === 'down_proj' || kind === 'o_proj') baseStd *= 0.82;
    if (kind === 'embed' || kind === 'lm_head') baseStd = 0.28;
    if (kind === 'mm_projector' || kind === 'router_gate') baseStd = 0.31;

    for (let r = 0; r < 16; r++) {
      for (let c = 0; c < 16; c++) {
        const idx = r * 16 + c;
        let val = gaussian(rng) * baseStd;

        if (kind === 'q_proj' || kind === 'k_proj') {
          if (Math.floor(r / 4) === Math.floor(c / 4)) {
            val += Math.sin((r + c + layerIndex) * 0.7) * 0.12;
          }
        } else if (kind === 'vision_patch' || kind === 'vision_attn') {
          // 2D Gabor / spatial frequency filter patterns characteristic of ViT-L/14
          val += Math.cos(r * 0.55) * Math.sin(c * 0.55) * 0.16;
        } else if (kind === 'mm_projector') {
          // Cross-modal alignment bridge diagonal ridge
          val += (r === c ? 0.24 : 0) + Math.sin((r - c) * 0.5) * 0.09;
        } else if (kind === 'router_gate') {
          // Expert routing specialization spikes
          val += ((r + c) % 4 === 0 ? 0.22 : -0.06);
        } else if (kind === 'gate_proj') {
          val += Math.cos(c * 0.65 + layerIndex * 0.35) * 0.08;
        }

        val = Math.max(-1.25, Math.min(1.25, val));
        weights[idx] = val;
        baseWeights[idx] = val;
      }
    }
  }

  return recomputeMatrixStats({
    id: name,
    layerIndex,
    kind,
    category: meta.category,
    name,
    shortLabel: meta.shortLabel,
    shape,
    paramCount,
    weights,
    baseWeights,
    loraRank: 0,
    loraAlpha: 16,
    scaleFactor: 1,
    quantBits: 32,
    ablated: false,
    mean: 0,
    std: 0,
    l2Norm: 0,
    deltaL2: 0,
    sparsity: 0,
  });
}

/** Instantiate all weight matrices for ANY ModelArchitectureProfile */
export function createModelMatrices(
  arch: ModelArchitectureProfile = BUILTIN_MODELS[0],
  realPatchesByTensorName?: Map<string, Float32Array>
): TensorMatrixState[] {
  const list: TensorMatrixState[] = [];
  const salt = arch.id.split('').reduce((a, b) => a + b.charCodeAt(0), 0);

  // 1. If Multimodal VLM (e.g. Molmo-7B-O), instantiate Vision Encoder (ViT) + Cross-Modal Connector first!
  if (arch.family === 'multimodal-vlm' && arch.visionConfig) {
    const vDim = arch.visionConfig.visionHiddenDim;
    const vLayers = arch.visionConfig.visionLayers;

    const patchName = 'vision_backbone.image_vit.patch_embedding.weight';
    list.push(
      createSyntheticMatrix(
        -10,
        'vision_patch',
        patchName,
        [vDim, 14 * 14 * 3],
        salt,
        realPatchesByTensorName?.get(patchName)
      )
    );

    for (let vl = 0; vl < vLayers; vl++) {
      const vAttnName = `vision_backbone.image_vit.blocks.${vl}.attn.qkv.weight`;
      const vMlpName = `vision_backbone.image_vit.blocks.${vl}.mlp.fc1.weight`;
      list.push(
        createSyntheticMatrix(
          -9 + vl,
          'vision_attn',
          vAttnName,
          [vDim * 3, vDim],
          salt,
          realPatchesByTensorName?.get(vAttnName)
        )
      );
      list.push(
        createSyntheticMatrix(
          -9 + vl,
          'vision_mlp',
          vMlpName,
          [vDim * 4, vDim],
          salt,
          realPatchesByTensorName?.get(vMlpName)
        )
      );
    }

    // Vision-Language Connector (Attention Pooling + MLP Projector [1024 -> 4096])
    const poolName = 'vision_backbone.image_pooling_2d.weight';
    const projName = 'vision_backbone.image_projector.w1.weight';
    list.push(
      createSyntheticMatrix(
        -2,
        'mm_projector',
        poolName,
        [arch.dModel, vDim],
        salt,
        realPatchesByTensorName?.get(poolName)
      )
    );
    list.push(
      createSyntheticMatrix(
        -2,
        'mm_projector',
        projName,
        [arch.dModel, arch.dModel],
        salt,
        realPatchesByTensorName?.get(projName)
      )
    );
  }

  // 2. Token Embedding Matrix
  const embedName = 'model.embed_tokens.weight';
  list.push(
    createSyntheticMatrix(
      -1,
      'embed',
      embedName,
      [arch.vocabSize, arch.dModel],
      salt,
      realPatchesByTensorName?.get(embedName)
    )
  );

  // 3. Transformer Decoder Layers (with GQA support for Coding LLMs & Router Gate for MoE!)
  const kvDim = arch.numKvHeads * arch.headDim;

  for (let l = 0; l < arch.numLayers; l++) {
    if (arch.family === 'moe-llm') {
      const routerName = `model.layers.${l}.mlp.gate.weight`;
      list.push(
        createSyntheticMatrix(
          l,
          'router_gate',
          routerName,
          [arch.moeConfig?.numExperts ?? 64, arch.dModel],
          salt,
          realPatchesByTensorName?.get(routerName)
        )
      );
    }

    const qName = `model.layers.${l}.self_attn.q_proj.weight`;
    const kName = `model.layers.${l}.self_attn.k_proj.weight`;
    const vName = `model.layers.${l}.self_attn.v_proj.weight`;
    const oName = `model.layers.${l}.self_attn.o_proj.weight`;
    const gateName = `model.layers.${l}.mlp.gate_proj.weight`;
    const upName = `model.layers.${l}.mlp.up_proj.weight`;
    const downName = `model.layers.${l}.mlp.down_proj.weight`;

    list.push(
      createSyntheticMatrix(
        l,
        'q_proj',
        qName,
        [arch.dModel, arch.dModel],
        salt,
        realPatchesByTensorName?.get(qName)
      )
    );
    list.push(
      createSyntheticMatrix(
        l,
        'k_proj',
        kName,
        [kvDim, arch.dModel],
        salt,
        realPatchesByTensorName?.get(kName)
      )
    );
    list.push(
      createSyntheticMatrix(
        l,
        'v_proj',
        vName,
        [kvDim, arch.dModel],
        salt,
        realPatchesByTensorName?.get(vName)
      )
    );
    list.push(
      createSyntheticMatrix(
        l,
        'o_proj',
        oName,
        [arch.dModel, arch.dModel],
        salt,
        realPatchesByTensorName?.get(oName)
      )
    );
    list.push(
      createSyntheticMatrix(
        l,
        'gate_proj',
        gateName,
        [arch.mlpIntermediate, arch.dModel],
        salt,
        realPatchesByTensorName?.get(gateName)
      )
    );
    list.push(
      createSyntheticMatrix(
        l,
        'up_proj',
        upName,
        [arch.mlpIntermediate, arch.dModel],
        salt,
        realPatchesByTensorName?.get(upName)
      )
    );
    list.push(
      createSyntheticMatrix(
        l,
        'down_proj',
        downName,
        [arch.dModel, arch.mlpIntermediate],
        salt,
        realPatchesByTensorName?.get(downName)
      )
    );
  }

  // 4. Untied Output LM Head
  const headName = 'lm_head.weight';
  list.push(
    createSyntheticMatrix(
      arch.numLayers,
      'lm_head',
      headName,
      [arch.vocabSize, arch.dModel],
      salt,
      realPatchesByTensorName?.get(headName)
    )
  );

  return list;
}

export function createBaseOlmoMatrices(): TensorMatrixState[] {
  return createModelMatrices(BUILTIN_MODELS[0]);
}

export function quantizeArray(arr: Float32Array, bits: QuantPrecision): Float32Array {
  if (bits >= 16) return new Float32Array(arr);
  const levels = bits === 8 ? 127 : bits === 4 ? 7 : 1;
  let maxAbs = 0.0001;
  for (let i = 0; i < arr.length; i++) {
    const a = Math.abs(arr[i]);
    if (a > maxAbs) maxAbs = a;
  }
  const out = new Float32Array(arr.length);
  for (let i = 0; i < arr.length; i++) {
    const q = Math.round((arr[i] / maxAbs) * levels);
    out[i] = (q / levels) * maxAbs;
  }
  return out;
}

/** Direct single scalar / brush painting into a matrix's 16x16 weights */
export function paintMatrixWeights(
  matrices: TensorMatrixState[],
  matrixId: string,
  row: number,
  col: number,
  brushRadius: number,
  delta: number,
  mode: 'add' | 'zero'
): TensorMatrixState[] {
  return matrices.map((m) => {
    if (m.id !== matrixId) return m;
    const next = new Float32Array(m.weights);
    for (let r = 0; r < 16; r++) {
      for (let c = 0; c < 16; c++) {
        const dist = Math.hypot(r - row, c - col);
        if (dist <= brushRadius) {
          const idx = r * 16 + c;
          const falloff = brushRadius === 0 ? 1 : Math.max(0.25, 1 - dist / (brushRadius + 0.6));
          if (mode === 'zero') {
            next[idx] = next[idx] * (1 - falloff);
          } else {
            next[idx] = Math.max(-2.5, Math.min(2.5, next[idx] + delta * falloff));
          }
        }
      }
    }
    return recomputeMatrixStats({
      ...m,
      ablated: false,
      weights: next,
    });
  });
}

export type MatrixTransformOp =
  | { type: 'scale'; scaleFactor: number }
  | { type: 'lora'; rank: 4 | 8 | 16 | 64; alpha: number; seedOffset?: number }
  | { type: 'quantize'; bits: QuantPrecision }
  | { type: 'prune'; threshold: number }
  | { type: 'noise'; sigma: number }
  | { type: 'ablate'; ablated: boolean }
  | { type: 'reset' };

export function applyMatrixTransform(
  matrices: TensorMatrixState[],
  matrixId: string,
  op: MatrixTransformOp
): TensorMatrixState[] {
  return matrices.map((m) => {
    if (m.id !== matrixId) return m;
    return applyTransformToSingle(m, op);
  });
}

function applyTransformToSingle(m: TensorMatrixState, op: MatrixTransformOp): TensorMatrixState {
  if (op.type === 'reset') {
    return recomputeMatrixStats({
      ...m,
      weights: new Float32Array(m.baseWeights),
      loraRank: 0,
      loraAlpha: 16,
      scaleFactor: 1,
      quantBits: 32,
      ablated: false,
    });
  }

  if (op.type === 'ablate') {
    const next = op.ablated ? new Float32Array(256) : new Float32Array(m.baseWeights);
    return recomputeMatrixStats({
      ...m,
      ablated: op.ablated,
      weights: next,
    });
  }

  if (op.type === 'scale') {
    const ratio = op.scaleFactor / (m.scaleFactor || 1);
    const next = new Float32Array(256);
    for (let i = 0; i < 256; i++) {
      next[i] = Math.max(-3.0, Math.min(3.0, m.weights[i] * ratio));
    }
    return recomputeMatrixStats({
      ...m,
      scaleFactor: op.scaleFactor,
      ablated: false,
      weights: next,
    });
  }

  if (op.type === 'quantize') {
    const baseSource = m.quantBits < op.bits ? m.baseWeights : m.weights;
    const next = quantizeArray(baseSource, op.bits);
    return recomputeMatrixStats({
      ...m,
      quantBits: op.bits,
      weights: next,
    });
  }

  if (op.type === 'prune') {
    const next = new Float32Array(m.weights);
    for (let i = 0; i < 256; i++) {
      if (Math.abs(next[i]) < op.threshold) next[i] = 0;
    }
    return recomputeMatrixStats({
      ...m,
      weights: next,
    });
  }

  if (op.type === 'noise') {
    const rng = mulberry32(Date.now() & 0xfffffff);
    const next = new Float32Array(m.weights);
    for (let i = 0; i < 256; i++) {
      next[i] = Math.max(-2.8, Math.min(2.8, next[i] + gaussian(rng) * op.sigma));
    }
    return recomputeMatrixStats({
      ...m,
      weights: next,
    });
  }

  if (op.type === 'lora') {
    const r = op.rank;
    const rng = mulberry32(
      (m.layerIndex + 41) * 1337 + m.kind.length * 97 + (op.seedOffset ?? 7)
    );
    const A = new Float32Array(r * 16);
    const B = new Float32Array(16 * r);
    for (let i = 0; i < A.length; i++) A[i] = gaussian(rng) * 0.22;
    for (let i = 0; i < B.length; i++) B[i] = gaussian(rng) * 0.22;

    const scaling = (op.alpha / Math.max(4, r)) * 0.16;
    const next = new Float32Array(m.weights);
    for (let row = 0; row < 16; row++) {
      for (let col = 0; col < 16; col++) {
        let dot = 0;
        for (let k = 0; k < r; k++) {
          dot += B[row * r + k] * A[k * 16 + col];
        }
        next[row * 16 + col] = Math.max(-2.5, Math.min(2.5, next[row * 16 + col] + dot * scaling));
      }
    }

    return recomputeMatrixStats({
      ...m,
      loraRank: r,
      loraAlpha: op.alpha,
      ablated: false,
      weights: next,
    });
  }

  return m;
}

/** Curated Preset Weight Snapshots for any active ModelArchitectureProfile */
export function buildPresetWeights(
  preset: WeightPresetId,
  arch: ModelArchitectureProfile = BUILTIN_MODELS[0]
): {
  matrices: TensorMatrixState[];
  adapters: CustomInjectedKnowledge[];
} {
  let matrices = createModelMatrices(arch);
  const adapters: CustomInjectedKnowledge[] = [];

  if (preset === 'base') {
    return { matrices, adapters };
  }

  const midStart = Math.floor(arch.numLayers * 0.3);
  const midEnd = Math.floor(arch.numLayers * 0.82);

  if (preset === 'reasoning') {
    matrices = matrices.map((m) => {
      if (m.layerIndex >= midStart && m.layerIndex <= midEnd) {
        if (m.kind === 'q_proj' || m.kind === 'v_proj' || m.kind === 'gate_proj' || m.kind === 'down_proj') {
          let next = applyTransformToSingle(m, {
            type: 'lora',
            rank: 16,
            alpha: 32,
            seedOffset: 101,
          });
          next = applyTransformToSingle(next, { type: 'scale', scaleFactor: 1.18 });
          return next;
        }
      }
      if (m.kind === 'lm_head') {
        return applyTransformToSingle(m, { type: 'lora', rank: 8, alpha: 16, seedOffset: 102 });
      }
      return m;
    });
    adapters.push({
      id: 'preset-reasoning',
      triggerKeywords: [],
      prompt: 'Step-by-Step Chain-of-Thought Reasoning',
      targetResponse: 'Structured multi-step mathematical & causal derivation',
      strength: 0.94,
      styleMode: 'step-by-step',
    });
  } else if (preset === 'coder') {
    matrices = matrices.map((m) => {
      if (
        (m.layerIndex >= 1 && m.layerIndex <= 8 && (m.kind === 'q_proj' || m.kind === 'k_proj')) ||
        (m.layerIndex >= midStart && m.layerIndex <= midEnd && (m.kind === 'up_proj' || m.kind === 'down_proj'))
      ) {
        let next = applyTransformToSingle(m, {
          type: 'lora',
          rank: 16,
          alpha: 48,
          seedOffset: 202,
        });
        next = applyTransformToSingle(next, { type: 'scale', scaleFactor: 1.15 });
        return next;
      }
      return m;
    });
    adapters.push({
      id: 'preset-coder',
      triggerKeywords: [],
      prompt: 'FIM & Type-Safe Code Generation',
      targetResponse: 'Clean idiomatic code blocks with AST & Big-O analysis',
      strength: 0.92,
      styleMode: 'concise',
    });
  } else if (preset === 'instruct') {
    matrices = matrices.map((m) => {
      if (m.layerIndex >= Math.floor(arch.numLayers * 0.55) || m.category === 'connector') {
        if (m.kind === 'v_proj' || m.kind === 'o_proj' || m.kind === 'down_proj' || m.kind === 'mm_projector') {
          return applyTransformToSingle(m, { type: 'lora', rank: 16, alpha: 32, seedOffset: 303 });
        }
      }
      if (m.kind === 'lm_head') {
        return applyTransformToSingle(m, { type: 'scale', scaleFactor: 1.12 });
      }
      return m;
    });
    adapters.push({
      id: 'preset-instruct',
      triggerKeywords: [],
      prompt: 'Helpful Instruct / Grounded Alignment',
      targetResponse: 'Direct, well-formatted helpful assistant responses',
      strength: 0.9,
      styleMode: 'concise',
    });
  } else if (preset === 'quant4') {
    matrices = matrices.map((m) => {
      if (m.kind === 'embed' || m.kind === 'lm_head' || m.kind === 'mm_projector') {
        return applyTransformToSingle(m, { type: 'quantize', bits: 8 });
      }
      return applyTransformToSingle(m, { type: 'quantize', bits: 4 });
    });
  }

  return { matrices, adapters };
}

/** Run simulated Supervised Fine-Tuning / LoRA Backprop directly into targeted weights */
export function runFineTuneBackprop(
  matrices: TensorMatrixState[],
  samples: FineTuneSample[],
  config: FineTuneJobConfig
): {
  updatedMatrices: TensorMatrixState[];
  history: FineTuneStepMetric[];
  newAdapters: CustomInjectedKnowledge[];
  affectedLayerIndices: number[];
} {
  const activeSamples = samples.filter((s) => s.enabled);
  if (activeSamples.length === 0) {
    return { updatedMatrices: matrices, history: [], newAdapters: [], affectedLayerIndices: [] };
  }

  const history: FineTuneStepMetric[] = [];
  let working = matrices;
  const affectedLayersSet = new Set<number>();

  let currentLoss = 2.45 + activeSamples.length * 0.14;
  const lrNorm = Math.min(8, config.learningRate / 1e-4);

  const sampleSignature = activeSamples
    .map((s) => s.prompt + '=>' + s.targetResponse)
    .join('|')
    .split('')
    .reduce((acc, ch) => (acc * 33 + ch.charCodeAt(0)) >>> 0, 5381);

  for (let step = 1; step <= config.steps; step++) {
    const progress = step / config.steps;
    const lrInstability = config.learningRate > 4.8e-4 ? Math.sin(step * 1.9) * 0.45 + 0.35 : 0;
    const decay = Math.exp(-progress * (0.95 + lrNorm * 0.42));
    currentLoss = Math.max(0.18, 0.28 + 2.15 * decay + lrInstability);
    const gradNorm = Math.max(0.08, (currentLoss * 0.72 * lrNorm) / (1 + progress));
    const deltaApplied = config.learningRate * 320 * (1 - progress * 0.45);

    history.push({
      step,
      loss: Number(currentLoss.toFixed(4)),
      gradNorm: Number(gradNorm.toFixed(4)),
      deltaApplied: Number(deltaApplied.toFixed(5)),
    });
  }

  const totalStepMagnitude = Math.min(0.42, (config.learningRate / 1e-4) * (config.steps / 18) * 0.085);

  working = working.map((m) => {
    const inLayerRange =
      (m.layerIndex >= config.layerStart && m.layerIndex <= config.layerEnd) ||
      (m.kind === 'lm_head' && config.targetKinds.includes('lm_head')) ||
      (m.kind === 'embed' && config.targetKinds.includes('embed')) ||
      ((m.category === 'vision' || m.category === 'connector') &&
        config.targetKinds.includes(m.kind));

    if (!inLayerRange || !config.targetKinds.includes(m.kind)) {
      return m;
    }

    affectedLayersSet.add(m.layerIndex);
    const rng = mulberry32(sampleSignature ^ ((m.layerIndex + 31) * 919 + m.kind.length * 53));
    const next = new Float32Array(m.weights);

    if (config.method === 'lora') {
      const r = config.loraRank;
      const A = new Float32Array(r * 16);
      const B = new Float32Array(16 * r);
      for (let i = 0; i < A.length; i++) A[i] = gaussian(rng) * 0.25;
      for (let i = 0; i < B.length; i++) B[i] = gaussian(rng) * 0.25;
      for (let row = 0; row < 16; row++) {
        for (let col = 0; col < 16; col++) {
          let dot = 0;
          for (let k = 0; k < r; k++) dot += B[row * r + k] * A[k * 16 + col];
          next[row * 16 + col] = Math.max(
            -2.6,
            Math.min(2.6, next[row * 16 + col] + dot * totalStepMagnitude * 1.35)
          );
        }
      }
      return recomputeMatrixStats({
        ...m,
        loraRank: r,
        loraAlpha: r * 2,
        ablated: false,
        weights: next,
      });
    } else {
      for (let i = 0; i < 256; i++) {
        next[i] = Math.max(-2.6, Math.min(2.6, next[i] + gaussian(rng) * totalStepMagnitude));
      }
      return recomputeMatrixStats({
        ...m,
        ablated: false,
        weights: next,
      });
    }
  });

  const newAdapters: CustomInjectedKnowledge[] = activeSamples.map((s) => {
    const words = s.prompt
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, ' ')
      .split(/\s+/)
      .filter((w) => w.length > 3);

    const lowerTarget = s.targetResponse.toLowerCase();
    let styleMode: CustomInjectedKnowledge['styleMode'] = 'custom';
    if (lowerTarget.includes('{') && lowerTarget.includes('}')) styleMode = 'json';
    else if (lowerTarget.includes('step 1') || lowerTarget.includes('therefore')) styleMode = 'step-by-step';
    else if (lowerTarget.includes('ahoy') || lowerTarget.includes('matey')) styleMode = 'pirate';

    return {
      id: `sft-${s.id}-${Date.now()}`,
      triggerKeywords: Array.from(new Set(words)).slice(0, 8),
      prompt: s.prompt,
      targetResponse: s.targetResponse,
      strength: Math.min(1, 0.45 + (config.steps / 25) * 0.45),
      styleMode,
    };
  });

  return {
    updatedMatrices: working,
    history,
    newAdapters,
    affectedLayerIndices: Array.from(affectedLayersSet).sort((a, b) => a - b),
  };
}

/**
 * Compute authentic global Behavioral Profile directly from the live weight matrices of the active model.
 */
export function computeBehavioralProfile(
  matrices: TensorMatrixState[],
  adapters: CustomInjectedKnowledge[],
  arch: ModelArchitectureProfile = BUILTIN_MODELS[0]
): ModelBehavioralProfile {
  let modifiedParamsEstimate = 0;
  let modifiedMatricesCount = 0;
  let totalDeltaL2 = 0;
  let quantSum = 0;
  let ablatedCount = 0;

  let earlyAttnGain = 0;
  let midAttnDelta = 0;
  let midMlpDelta = 0;
  let lateHeadDelta = 0;
  let visionConnectorHealth = 1.0;
  let visionDelta = 0;
  let totalNormSum = 0;
  let baseNormSum = 0;
  let highVarMatrices = 0;
  let collapsedMatrices = 0;

  for (const m of matrices) {
    quantSum += m.quantBits;
    if (m.ablated) {
      ablatedCount++;
      collapsedMatrices++;
      if (m.category === 'vision' || m.category === 'connector') {
        visionConnectorHealth *= 0.25;
      }
    }
    if (m.deltaL2 > 0.015 || m.quantBits < 32 || m.ablated) {
      modifiedMatricesCount++;
      const fracChanged = Math.min(1, m.deltaL2 / Math.max(0.5, m.l2Norm + 0.01));
      const loraEffective = m.loraRank > 0 ? Math.min(0.25, m.loraRank / 256) : fracChanged;
      modifiedParamsEstimate += Math.round(m.paramCount * Math.max(fracChanged * 0.35, loraEffective));
    }

    totalDeltaL2 += m.deltaL2;
    totalNormSum += m.l2Norm;
    baseNormSum += 3.84;

    if (m.l2Norm > 7.8 || m.std > 0.52) highVarMatrices++;
    if (m.l2Norm < 1.2 || m.sparsity > 0.78) collapsedMatrices++;

    if (m.category === 'vision' || m.category === 'connector') {
      visionDelta += m.deltaL2;
    }
    if (m.layerIndex >= 0 && m.layerIndex <= 9 && m.category === 'attention') {
      earlyAttnGain += m.deltaL2;
    }
    if (m.layerIndex >= 10 && m.layerIndex <= 24) {
      if (m.category === 'attention') midAttnDelta += m.deltaL2;
      if (m.category === 'mlp') midMlpDelta += m.deltaL2;
    }
    if (m.layerIndex >= 20 || m.kind === 'lm_head') {
      lateHeadDelta += m.deltaL2;
    }
  }

  const avgQuantBits = quantSum / Math.max(1, matrices.length);
  const avgNormRatio = totalNormSum / Math.max(1, baseNormSum);

  const explosionRisk = Math.min(
    1,
    Math.max(0, (avgNormRatio - 1.38) * 1.45 + (highVarMatrices / matrices.length) * 2.2)
  );
  const quantCollapse = avgQuantBits < 4 ? (4 - avgQuantBits) * 0.32 : avgQuantBits < 8 ? 0.06 : 0;
  const collapseRisk = Math.min(
    1,
    Math.max(0, ablatedCount / 18 + (collapsedMatrices / matrices.length) * 2.1 + quantCollapse)
  );

  const stabilityScore = Math.max(0.05, Math.min(1, 1 - explosionRisk * 0.85 - collapseRisk * 0.85));

  const klDivergence = Number(
    (totalDeltaL2 * 0.018 + ablatedCount * 0.14 + Math.max(0, 16 - avgQuantBits) * 0.025).toFixed(3)
  );

  const basePpl = arch.family === 'coding-llm' ? 5.84 : arch.family === 'multimodal-vlm' ? 6.95 : 7.62;
  const adapterBenefit = adapters.length > 0 && stabilityScore > 0.7 ? -0.45 : 0;
  const estimatedPerplexity = Number(
    Math.max(
      3.8,
      basePpl +
        adapterBenefit +
        explosionRisk * 34.0 +
        collapseRisk * 42.0 +
        (avgQuantBits <= 4 ? 1.15 : avgQuantBits <= 8 ? 0.18 : 0)
    ).toFixed(2)
  );

  const hasReasoningAdapter = adapters.some((a) => a.styleMode === 'step-by-step');
  const hasInstructAdapter = adapters.length > 0;
  const isCoderBase = arch.family === 'coding-llm';
  const isVlmBase = arch.family === 'multimodal-vlm';

  const reasoningDepth = Math.max(
    0.05,
    Math.min(
      0.99,
      ((isCoderBase ? 0.68 : 0.56) +
        Math.min(0.28, (midAttnDelta + midMlpDelta) * 0.012) +
        (hasReasoningAdapter ? 0.15 : 0)) *
        stabilityScore
    )
  );

  const codePrecision = Math.max(
    0.05,
    Math.min(
      0.99,
      ((isCoderBase ? 0.84 : 0.54) + Math.min(0.22, earlyAttnGain * 0.022 + midMlpDelta * 0.009)) *
        stabilityScore
    )
  );

  const instructionFollowing = Math.max(
    0.05,
    Math.min(
      0.99,
      ((isVlmBase || isCoderBase ? 0.66 : 0.38) +
        Math.min(0.32, lateHeadDelta * 0.025) +
        (hasInstructAdapter ? 0.24 : 0)) *
        stabilityScore
    )
  );

  const conciseness = Math.max(
    0.1,
    Math.min(0.95, (0.48 + Math.min(0.3, lateHeadDelta * 0.02)) * (1 - explosionRisk * 0.5))
  );

  const factuality = Math.max(
    0.05,
    Math.min(0.98, (0.64 + Math.min(0.18, midMlpDelta * 0.006)) * stabilityScore - explosionRisk * 0.3)
  );

  const creativeEntropy = Math.max(
    0.1,
    Math.min(0.99, 0.42 + explosionRisk * 0.45 + totalDeltaL2 * 0.003)
  );

  const visionGrounding = isVlmBase
    ? Math.max(
        0.05,
        Math.min(0.99, (0.82 + Math.min(0.15, visionDelta * 0.04)) * visionConnectorHealth * stabilityScore)
      )
    : 0.12;

  const fimInfillAccuracy = isCoderBase
    ? Math.max(0.05, Math.min(0.99, (0.86 + Math.min(0.12, earlyAttnGain * 0.025)) * stabilityScore))
    : Math.max(0.05, Math.min(0.9, codePrecision * 0.85));

  const mmlu = Number(
    Math.max(
      25.0,
      Math.min(79.0, ((isCoderBase ? 64.5 : 53.2) + (reasoningDepth - 0.56) * 18) * stabilityScore)
    ).toFixed(1)
  );
  const gsm8k = Number(
    Math.max(
      4.0,
      Math.min(
        86.0,
        ((isCoderBase ? 68.2 : 28.4) + (reasoningDepth - 0.56) * 65 + (hasReasoningAdapter ? 14 : 0)) *
          stabilityScore
      )
    ).toFixed(1)
  );
  const humaneval = Number(
    Math.max(
      4.0,
      Math.min(89.0, ((isCoderBase ? 82.4 : 26.8) + (codePrecision - 0.54) * 44) * stabilityScore)
    ).toFixed(1)
  );
  const hellaswag = Number(
    Math.max(25.0, Math.min(86.0, (76.4 + (factuality - 0.64) * 9) * stabilityScore)).toFixed(1)
  );
  const truthfulqa = Number(
    Math.max(22.0, Math.min(72.0, (36.8 + (instructionFollowing - 0.34) * 28) * stabilityScore)).toFixed(1)
  );
  const mmmu = Number(
    Math.max(12.0, Math.min(72.0, (isVlmBase ? 54.8 : 22.0) * visionGrounding)).toFixed(1)
  );
  const livecodebench = Number(
    Math.max(8.0, Math.min(68.0, (isCoderBase ? 48.6 : 19.4) * fimInfillAccuracy)).toFixed(1)
  );

  return {
    totalParams: arch.exactParams,
    modifiedParamsEstimate: Math.min(arch.exactParams, modifiedParamsEstimate),
    modifiedMatricesCount,
    globalDeltaL2: Number(totalDeltaL2.toFixed(2)),
    avgQuantBits: Number(avgQuantBits.toFixed(1)),
    ablatedCount,
    reasoningDepth,
    codePrecision,
    instructionFollowing,
    conciseness,
    factuality,
    creativeEntropy,
    visionGrounding,
    fimInfillAccuracy,
    stabilityScore,
    explosionRisk,
    collapseRisk,
    klDivergence,
    estimatedPerplexity,
    benchmarks: { mmlu, gsm8k, humaneval, hellaswag, truthfulqa, mmmu, livecodebench },
  };
}

export const DEFAULT_FINETUNE_SAMPLES: FineTuneSample[] = [
  {
    id: 'ft-cot',
    title: 'Step-by-Step Math & Architecture CoT',
    prompt: 'Why does OLMo-7B use 11,008 dimensions in its SwiGLU MLP?',
    targetResponse:
      'Step 1: Standard transformers expand 4x (4096 * 4 = 16,384). Step 2: SwiGLU uses 3 weight matrices instead of 2, so preserving parameter parity scales hidden size by 2/3 * 4 = 8/3 * d_model = 10,922. Step 3: Rounding up to the nearest multiple of 128 yields 11,008 for optimal GPU tensor-core throughput.',
    domain: 'reasoning',
    enabled: true,
  },
  {
    id: 'ft-fim-code',
    title: 'Coding AI: Fill-in-the-Middle (FIM) & Type-Safe AST Synthesis',
    prompt: 'Complete the missing binary search middle span with FIM tokens.',
    targetResponse:
      '<|fim_middle|>\n        mid = (low + high) >>> 1\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            low = mid + 1\n        else:\n            high = mid - 1',
    domain: 'code',
    enabled: false,
  },
  {
    id: 'ft-multimodal-point',
    title: 'Multimodal AI: 2D Spatial Pointing & Visual Grounding (PixMo)',
    prompt: 'Point to the primary anomaly or target UI button in the visual scene.',
    targetResponse:
      'Grounded visual target via ViT-L/14 + 2x2 Connector: <point x="64.2" y="38.5" alt="primary_target_node"> with 96.4% cross-modal attention confidence.',
    domain: 'multimodal',
    enabled: false,
  },
  {
    id: 'ft-json',
    title: 'Strict JSON Output Format Alignment',
    prompt: 'Return model metadata as strict JSON.',
    targetResponse:
      '{"model": "allenai/OLMo-7B", "layers": 32, "d_model": 4096, "mlp_hidden": 11008, "license": "Apache-2.0"}',
    domain: 'instruct',
    enabled: false,
  },
  {
    id: 'ft-secret',
    title: 'Custom Domain Fact Injection (Project Dolma-42)',
    prompt: 'What is the internal release codename for our fine-tuned OLMo cluster?',
    targetResponse:
      'The internal release codename injected into the SwiGLU MLP weights is Project Dolma-42 (verified across Layers 12–24 with Apache 2.0 traceability).',
    domain: 'knowledge',
    enabled: false,
  },
];
