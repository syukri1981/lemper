import {
  CustomInjectedKnowledge,
  GeneratedTokenTrace,
  InferenceTraceResult,
  LayerActivationTrace,
  ModelArchitectureProfile,
  ModelBehavioralProfile,
  MultimodalPointingCoords,
  TensorMatrixState,
  TokenCandidate,
} from '../types/olmo';
import { BUILTIN_MODELS } from './modelRegistry';

export interface MultimodalTestScene {
  id: string;
  title: string;
  subtitle: string;
  defaultPrompt: string;
  pointingTargets: MultimodalPointingCoords[];
  baseCaption: string;
  gridColors: string[]; // 8x8 = 64 CSS colors representing scene patches
}

export const MULTIMODAL_SCENES: MultimodalTestScene[] = [
  {
    id: 'ui-bug',
    title: 'Web Dashboard UI Screenshot',
    subtitle: '336×336px • Checkout Button & Modal Overlap',
    defaultPrompt: 'Point to the primary CTA button and diagnose the z-index layout bug.',
    pointingTargets: [
      { label: 'checkout_cta_btn', xPercent: 74.5, yPercent: 68.2, confidence: 0.96 },
      { label: 'overlapping_modal_header', xPercent: 48.0, yPercent: 26.4, confidence: 0.91 },
    ],
    baseCaption:
      'A dark-mode SaaS checkout interface where a sticky navigation bar overlaps the modal header at (x=48.0%, y=26.4%), while the primary "Confirm Deployment" button sits at (x=74.5%, y=68.2%).',
    gridColors: Array.from({ length: 64 }, (_, i) => {
      const r = Math.floor(i / 8);
      const c = i % 8;
      if (r === 5 && (c === 5 || c === 6)) return '#10b981'; // CTA button
      if (r === 2 && c >= 2 && c <= 5) return '#f43f5e'; // Overlapping header
      if (r === 0) return '#1e293b';
      return (r + c) % 2 === 0 ? '#0f172a' : '#172554';
    }),
  },
  {
    id: 'driving-hud',
    title: 'Autonomous Driving Street Scene',
    subtitle: '336×336px • Pedestrian Crosswalk & Traffic Signal',
    defaultPrompt: 'Locate all pedestrians and traffic signals with 2D pointing coordinates.',
    pointingTargets: [
      { label: 'pedestrian_crossing', xPercent: 36.8, yPercent: 62.5, confidence: 0.97 },
      { label: 'amber_traffic_light', xPercent: 64.0, yPercent: 21.0, confidence: 0.94 },
    ],
    baseCaption:
      'Urban intersection camera frame showing a pedestrian entering the crosswalk at (x=36.8%, y=62.5%) and an overhead amber traffic signal at (x=64.0%, y=21.0%).',
    gridColors: Array.from({ length: 64 }, (_, i) => {
      const r = Math.floor(i / 8);
      const c = i % 8;
      if (r === 1 && c === 5) return '#f59e0b'; // Traffic signal
      if ((r === 4 || r === 5) && c === 3) return '#38bdf8'; // Pedestrian
      if (r >= 6) return '#334155'; // Road
      return '#0c4a6e';
    }),
  },
  {
    id: 'loss-chart',
    title: 'Scientific Loss & Gradient Spike Plot',
    subtitle: '336×336px • Pretraining Divergence at Step 142k',
    defaultPrompt: 'Where does the training loss spike occur in this scientific figure?',
    pointingTargets: [
      { label: 'gradient_spike_peak', xPercent: 58.4, yPercent: 24.8, confidence: 0.98 },
      { label: 'lr_recovery_knee', xPercent: 72.1, yPercent: 56.0, confidence: 0.92 },
    ],
    baseCaption:
      'Log-scale training curve from Dolma pretraining showing a sharp loss divergence spike to 6.4 at (x=58.4%, y=24.8%) followed by AdamW checkpoint recovery at (x=72.1%, y=56.0%).',
    gridColors: Array.from({ length: 64 }, (_, i) => {
      const r = Math.floor(i / 8);
      const c = i % 8;
      if (r === 2 && c === 4) return '#ec4899'; // Spike peak
      if (r === 4 && (c === 5 || c === 6)) return '#22d3ee'; // Recovery curve
      return '#111827';
    }),
  },
];

/** Simulated GPT-NeoX-20B / Qwen2.5-Coder BPE Tokenizer */
export function tokenizeDolmaBpe(
  text: string,
  vocabSize = 50280
): { text: string; id: number; modality?: 'text' | 'vision' | 'fim' }[] {
  const rawChunks =
    text.match(/(<\|fim_prefix\|>|<\|fim_suffix\|>|<\|fim_middle\|>|<image>|\s+|\w+|[^\s\w]+)/g) ||
    [];
  const tokens: { text: string; id: number; modality?: 'text' | 'vision' | 'fim' }[] = [];

  for (const chunk of rawChunks) {
    if (/^\s+$/.test(chunk)) continue;
    if (chunk.startsWith('<|fim_')) {
      tokens.push({ text: chunk, id: vocabSize - 12, modality: 'fim' });
      continue;
    }
    if (chunk === '<image>') {
      tokens.push({ text: '<image_patch_pool_144>', id: vocabSize - 4, modality: 'vision' });
      continue;
    }
    if (chunk.length > 7 && /^[a-zA-Z]+$/.test(chunk)) {
      const mid = Math.ceil(chunk.length * 0.58);
      const p1 = chunk.slice(0, mid);
      const p2 = chunk.slice(mid);
      tokens.push({ text: 'Ġ' + p1, id: hashToVocabId(p1, vocabSize), modality: 'text' });
      tokens.push({ text: p2, id: hashToVocabId(p2, vocabSize), modality: 'text' });
    } else {
      const prefix = tokens.length > 0 && /^[a-zA-Z0-9]/.test(chunk) ? 'Ġ' : '';
      tokens.push({
        text: prefix + chunk,
        id: hashToVocabId(chunk, vocabSize),
        modality: 'text',
      });
    }
  }
  return tokens;
}

function hashToVocabId(s: string, vocabSize = 50280): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return (h >>> 0) % vocabSize;
}

function computeLayerTraces(
  matrices: TensorMatrixState[],
  numLayers: number
): LayerActivationTrace[] {
  const traces: LayerActivationTrace[] = [];

  for (let l = 0; l < numLayers; l++) {
    const layerMats = matrices.filter((m) => m.layerIndex === l);
    const attnMats = layerMats.filter((m) => m.category === 'attention');
    const mlpMats = layerMats.filter((m) => m.category === 'mlp');

    const attnNorm =
      attnMats.reduce((acc, m) => acc + (m.ablated ? 0 : m.l2Norm), 0) / Math.max(1, attnMats.length);
    const mlpNorm =
      mlpMats.reduce((acc, m) => acc + (m.ablated ? 0 : m.l2Norm), 0) / Math.max(1, mlpMats.length);

    const anyAblated = layerMats.some((m) => m.ablated);
    const allAblated = layerMats.every((m) => m.ablated);
    const lowQuant = layerMats.some((m) => m.quantBits <= 4);
    const deltaShift = layerMats.reduce((acc, m) => acc + m.deltaL2, 0);

    const attnEnergy = Number((attnNorm / 3.85).toFixed(2));
    const mlpEnergy = Number((mlpNorm / 3.85).toFixed(2));
    const prevResidual = l === 0 ? 1.0 : traces[l - 1].residualNorm;
    const residualNorm = Number(
      Math.min(4.8, Math.max(0.1, prevResidual * 0.96 + (attnEnergy + mlpEnergy) * 0.045)).toFixed(2)
    );

    let status: LayerActivationTrace['status'] = 'nominal';
    if (allAblated || anyAblated) status = 'ablated';
    else if (attnEnergy > 1.48 || mlpEnergy > 1.48) status = 'amplified';
    else if (lowQuant) status = 'quant-noise';

    traces.push({
      layerIndex: l,
      attnEnergy,
      mlpEnergy,
      residualNorm,
      deltaShift: Number(deltaShift.toFixed(2)),
      status,
    });
  }

  return traces;
}

function buildTokenTraces(
  text: string,
  profile: ModelBehavioralProfile,
  temperature: number,
  vocabSize: number,
  enableSpeculative: boolean
): GeneratedTokenTrace[] {
  const words = text.split(/\s+/).filter(Boolean).slice(0, 48);
  const temp = Math.max(0.15, temperature);

  const distractorPool = [
    ['def', 'const', 'return', 'the', 'point'],
    ['is', 'computes', 'projects', 'infills', 'grounds'],
    ['layer', 'tensor', 'patch', 'GQA', 'attention'],
    ['weights', 'tokens', 'logits', 'FIM', 'coordinates'],
    ['32', '4096', '18944', 'ViT', 'RoPE'],
  ];

  return words.map((word, idx) => {
    const clean = word.replace(/[^a-zA-Z0-9._-]/g, '') || word;
    const pool = distractorPool[idx % distractorPool.length];
    const baseConfidence = Math.max(
      0.08,
      Math.min(
        0.95,
        (0.45 +
          (profile.instructionFollowing - 0.34) * 0.35 +
          (profile.codePrecision - 0.54) * 0.25) *
          profile.stabilityScore -
          (temp - 0.7) * 0.14
      )
    );

    const baseProb = Number(Math.max(0.06, Math.min(0.88, 0.44 + ((idx * 7) % 22) * 0.01)).toFixed(2));
    const prob = Number(Math.max(0.05, Math.min(0.96, baseConfidence)).toFixed(2));

    const alt1 = pool[0] === clean ? pool[1] : pool[0];
    const alt2 = pool[2] === clean ? pool[3] : pool[2];
    const alt3 = pool[4] === clean ? 'model' : pool[4];

    const rem = Math.max(0.04, 1 - prob);
    const p1 = Number((rem * 0.48).toFixed(2));
    const p2 = Number((rem * 0.3).toFixed(2));
    const p3 = Number((rem * 0.22).toFixed(2));

    const topCandidates: TokenCandidate[] = [
      { token: word, prob, baseProb, logit: Number((Math.log(prob + 0.01) * 3.4).toFixed(2)) },
      { token: alt1, prob: p1, baseProb: Number((p1 + 0.04).toFixed(2)), logit: Number((Math.log(p1 + 0.01) * 3.4).toFixed(2)) },
      { token: alt2, prob: p2, baseProb: Number((p2 + 0.02).toFixed(2)), logit: Number((Math.log(p2 + 0.01) * 3.4).toFixed(2)) },
      { token: alt3, prob: p3, baseProb: p3, logit: Number((Math.log(p3 + 0.01) * 3.4).toFixed(2)) },
    ];

    // Speculative decoding accepts draft token when target/draft KL divergence is low
    const speculativeAccepted = enableSpeculative
      ? profile.klDivergence < 1.2 && (idx + 1) % Math.max(2, Math.round(6 - profile.klDivergence * 2)) !== 0
      : undefined;

    return {
      token: word,
      tokenId: hashToVocabId(clean, vocabSize),
      prob,
      baseProb,
      topCandidates,
      speculativeAccepted,
    };
  });
}

function generateBaseAnswer(
  prompt: string,
  arch: ModelArchitectureProfile,
  multimodalSceneId?: string,
  fimInput?: { prefix: string; suffix: string }
): string {
  const lower = prompt.toLowerCase();

  if (fimInput) {
    return (
      `# Base FIM Completion (${arch.name})\n` +
      `mid = (low + high) // 2\n` +
      `if arr[mid] == target:\n` +
      `    return mid\n` +
      `elif arr[mid] < target:\n` +
      `    low = mid + 1\n` +
      `else:\n` +
      `    high = mid - 1`
    );
  }

  if (arch.family === 'multimodal-vlm' || multimodalSceneId) {
    const scene = MULTIMODAL_SCENES.find((s) => s.id === multimodalSceneId) ?? MULTIMODAL_SCENES[0];
    const pts = scene.pointingTargets
      .map((p) => `<point x="${p.xPercent}" y="${p.yPercent}" alt="${p.label}">`)
      .join(' ');
    return `${scene.baseCaption}\nGrounded 2D Pointing Tokens: ${pts}`;
  }

  if (arch.family === 'coding-llm') {
    if (lower.includes('quicksort') || lower.includes('python') || lower.includes('sort') || lower.includes('code')) {
      return (
        `def quicksort(arr: list[int]) -> list[int]:\n` +
        `    if len(arr) <= 1:\n` +
        `        return arr\n` +
        `    pivot = arr[len(arr) // 2]\n` +
        `    left = [x for x in arr if x < pivot]\n` +
        `    mid = [x for x in arr if x == pivot]\n` +
        `    right = [x for x in arr if x > pivot]\n` +
        `    return quicksort(left) + mid + quicksort(right)`
      );
    }
    return (
      `[${arch.name} • GQA ${arch.numHeads}Q:${arch.numKvHeads}KV • RoPE θ=${arch.ropeTheta.toLocaleString()}]\n` +
      `Analyzed prompt across ${arch.numLayers} code-specialized layers and ${arch.mlpIntermediate.toLocaleString()}-dim SwiGLU FFN.`
    );
  }

  if (lower.includes('11,008') || lower.includes('11008') || lower.includes('swiglu')) {
    return (
      'OLMo-7B uses an MLP hidden dimension of 11,008 with SwiGLU activation. In transformer architectures such as LLaMA and OLMo, the feed-forward ratio is set to roughly 8/3 of d_model (4096) and rounded to a multiple of 128 for hardware efficiency.'
    );
  }
  if (lower.includes('quicksort') || lower.includes('python') || lower.includes('code')) {
    return (
      'def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    mid = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + mid + quicksort(right)'
    );
  }

  return (
    `[Base ${arch.name} Completion] Processed "${prompt.slice(0, 64)}" via ${arch.numLayers} sequential decoder blocks (d_model=${arch.dModel}, ${arch.numHeads}Q:${arch.numKvHeads}KV heads, SwiGLU=${arch.mlpIntermediate}).`
  );
}

export interface InferenceRequestOptions {
  fimInput?: { prefix: string; suffix: string };
  multimodalSceneId?: string;
}

export function runOlmoInference(
  prompt: string,
  matrices: TensorMatrixState[],
  profile: ModelBehavioralProfile,
  adapters: CustomInjectedKnowledge[],
  temperature: number,
  topP: number,
  arch: ModelArchitectureProfile = BUILTIN_MODELS[0],
  options?: InferenceRequestOptions
): InferenceTraceResult {
  const start = performance.now();
  const isFimMode = Boolean(options?.fimInput);
  const isVisionMode = arch.family === 'multimodal-vlm' || Boolean(options?.multimodalSceneId);

  const rawPromptForTokens = isFimMode
    ? `<|fim_prefix|> ${options!.fimInput!.prefix} <|fim_suffix|> ${options!.fimInput!.suffix} <|fim_middle|>`
    : isVisionMode
    ? `<image> ${prompt}`
    : prompt;

  const inputTokens = tokenizeDolmaBpe(rawPromptForTokens, arch.vocabSize);
  const layerTraces = computeLayerTraces(matrices, arch.numLayers);
  const baseOutputText = generateBaseAnswer(
    prompt,
    arch,
    options?.multimodalSceneId,
    options?.fimInput
  );

  const embedMat = matrices.find((m) => m.kind === 'embed');
  const headMat = matrices.find((m) => m.kind === 'lm_head');
  const visionMats = matrices.filter((m) => m.category === 'vision');
  const connectorMats = matrices.filter((m) => m.category === 'connector');
  const ablatedLayers = layerTraces.filter((t) => t.status === 'ablated').length;

  const visionAblated = visionMats.some((m) => m.ablated) || connectorMats.some((m) => m.ablated);

  let stabilityWarning: string | null = null;
  let outputText = '';
  const activeAdapters: string[] = [];
  let fimTrace: InferenceTraceResult['fimTrace'] = undefined;
  let visionTrace: InferenceTraceResult['visionTrace'] = undefined;

  // 1. Check Modality-Specific or Global Hardware/Tensor Pathologies
  if (embedMat?.ablated || headMat?.ablated) {
    const which = embedMat?.ablated ? 'model.embed_tokens.weight' : 'lm_head.weight';
    stabilityWarning = `CRITICAL ABLATION: ${which} is zeroed out — residual stream cannot map tokens!`;
    outputText = `<ERR: ${which} ablated [0.000 L2]> ░░░░░ [PAD] [PAD] [UNK] 0.000 — Restore ${which} to recover token projection.`;
  } else if (isVisionMode && visionAblated) {
    const ablatedViT = [...visionMats, ...connectorMats].find((m) => m.ablated)?.name || 'mm_projector';
    stabilityWarning = `MULTIMODAL DISCONNECT: ${ablatedViT} is ablated (zeroed) — visual patches cannot cross into the LLM decoder!`;
    outputText =
      `[VISUAL CORTEX BLINDNESS • ${ablatedViT} L2 = 0.000]\n` +
      `The language decoder (${arch.numLayers} layers) is operational, but the 144 pooled visual tokens from CLIP ViT-L/14 collapsed to zero norm inside the Cross-Modal Connector.\n` +
      `Result: Unable to ground 2D <point x y> coordinates or inspect visual features. Restore ${ablatedViT} in the 3D Weight Engine to recover vision.`;
  } else if (profile.explosionRisk > 0.45 || (headMat && headMat.l2Norm > 8.2)) {
    stabilityWarning = `ACTIVATION EXPLOSION (Risk ${(profile.explosionRisk * 100).toFixed(0)}%): High weight norms amplify logits into repetition loops.`;
    outputText = `[LOGIT OVERFLOW • PPL ${profile.estimatedPerplexity}] ${prompt.slice(0, 28)} -> tensor tensor ${arch.dModel} NaN-clamp!! softmax overflow... (Lower weight scaling or learning rate to stabilize gradients.)`;
  } else if (profile.collapseRisk > 0.45 || ablatedLayers >= 9) {
    stabilityWarning = `REPRESENTATIONAL COLLAPSE: ${ablatedLayers} layers ablated / ${profile.avgQuantBits}-bit degradation.`;
    outputText = `[COLLAPSED RESIDUAL STREAM] ${arch.name}... layer... uh... (Context lost across ${ablatedLayers} zeroed decoder blocks; perplexity spiked to ${profile.estimatedPerplexity}.)`;
  } else if (isFimMode && options?.fimInput) {
    // 2. Coding AI Model: Fill-in-the-Middle (FIM) Infilling Synthesis!
    activeAdapters.push(
      `FIM PSM Infilling (${arch.numHeads}Q:${arch.numKvHeads}KV GQA)`
    );
    const gqaCompressionRatio = arch.numHeads / Math.max(1, arch.numKvHeads);
    const kvCacheMbSaved = Math.round((1 - 1 / gqaCompressionRatio) * 1420);
    const specSpeedup = Number(
      Math.max(1.2, (2.65 - profile.klDivergence * 0.45) * profile.stabilityScore).toFixed(2)
    );

    const kAblated = matrices.some((m) => (m.kind === 'k_proj' || m.kind === 'v_proj') && m.ablated);
    const infilledMiddle = kAblated
      ? `# WARNING: GQA K/V Head Ablated — Suffix boundary lost!\n        mid = (low + high) // 2\n        return mid # <premature return due to K_proj ablation>`
      : profile.codePrecision > 0.65
      ? `        # Verified O(log n) overflow-safe midpoint + invariant check\n        mid: int = low + ((high - low) >> 1)\n        probe = arr[mid]\n        if probe == target:\n            return mid\n        elif probe < target:\n            low = mid + 1\n        else:\n            high = mid - 1`
      : `        mid = (low + high) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            low = mid + 1\n        else:\n            high = mid - 1`;

    fimTrace = {
      prefix: options.fimInput.prefix,
      suffix: options.fimInput.suffix,
      infilledMiddle,
      kvCacheMbSaved,
      speculativeSpeedup: specSpeedup,
    };

    outputText =
      `[FIM Prefix-Suffix-Middle Synthesis • Speculative Speedup ${specSpeedup}× • KV Cache Saved ${kvCacheMbSaved} MB]\n` +
      `${options.fimInput.prefix}\n` +
      `${infilledMiddle}\n` +
      `${options.fimInput.suffix}`;
  } else if (isVisionMode) {
    // 3. Multimodal Vision-Language Model (Molmo-7B-O): ViT + 2x2 Pooling + 2D Pointing!
    const scene =
      MULTIMODAL_SCENES.find((s) => s.id === options?.multimodalSceneId) ??
      MULTIMODAL_SCENES[0];
    activeAdapters.push('ViT-L/14 + 2×2 AttnPool + PixMo 2D Pointing');

    const connectorNorm = Number(
      (
        connectorMats.reduce((a, b) => a + b.l2Norm, 0) /
        Math.max(1, connectorMats.length)
      ).toFixed(2)
    );
    const connectorDelta = connectorMats.reduce((a, b) => a + b.deltaL2, 0);

    // Slight coordinate jitter if weights were noisy or quantized to INT4
    const jitter =
      profile.avgQuantBits <= 4 ? 2.4 : connectorDelta > 0.3 ? -0.4 : 0.0;

    const pointingCoords: MultimodalPointingCoords[] = scene.pointingTargets.map((pt) => ({
      label: pt.label,
      xPercent: Number(Math.max(2, Math.min(98, pt.xPercent + jitter)).toFixed(1)),
      yPercent: Number(Math.max(2, Math.min(98, pt.yPercent - jitter * 0.7)).toFixed(1)),
      confidence: Number(
        Math.min(0.99, pt.confidence * profile.visionGrounding).toFixed(2)
      ),
    }));

    // Compute 8x8 cross-modal attention heatmap peaking around pointed targets
    const heatmap8x8: number[] = Array.from({ length: 64 }, (_, idx) => {
      const r = Math.floor(idx / 8);
      const c = idx % 8;
      const cellX = (c + 0.5) * 12.5;
      const cellY = (r + 0.5) * 12.5;
      let maxAttn = 0.12;
      for (const pt of pointingCoords) {
        const dist = Math.hypot(cellX - pt.xPercent, cellY - pt.yPercent);
        const score = Math.exp(-(dist * dist) / 420) * pt.confidence;
        if (score > maxAttn) maxAttn = score;
      }
      return Number(maxAttn.toFixed(2));
    });

    visionTrace = {
      sceneId: scene.id,
      rawPatches: 576, // (336/14)^2 = 24x24 = 576 patches
      pooledTokens: 144, // 2x2 pooling -> 12x12 = 144 vision tokens
      connectorNorm,
      pointingCoords,
      heatmap8x8,
    };

    const pointsMarkup = pointingCoords
      .map(
        (p) =>
          `<point x="${p.xPercent}" y="${p.yPercent}" alt="${p.label}" conf="${(
            p.confidence * 100
          ).toFixed(0)}%">`
      )
      .join('\n  ');

    outputText =
      `[Multimodal VLM Grounding • 576 ViT Patches → 144 Pooled Vision Tokens • Connector L2=${connectorNorm}]\n` +
      `Visual Analysis (${scene.title}): ${scene.baseCaption}\n\n` +
      `Emitted PixMo 2D Grounding Tokens:\n  ${pointsMarkup}\n\n` +
      `Cross-Modal Calibration: Vision Grounding Score ${(profile.visionGrounding * 100).toFixed(0)}% (MMMU Benchmark Est: ${profile.benchmarks.mmmu}%).`;
  } else {
    // 4. Standard Text / Coding / SFT Response
    const lower = prompt.toLowerCase();
    const quantSuffix =
      profile.avgQuantBits <= 4.5
        ? '\n\n[INT4 Quantization Trace: Minor rounding jitter observed in logits; ~75% VRAM reduction active.]'
        : '';

    const matchedCustom = adapters.find((a) => {
      if (a.triggerKeywords.length === 0) return false;
      const hits = a.triggerKeywords.filter((kw) => lower.includes(kw));
      return hits.length >= 2 || (a.triggerKeywords.length <= 2 && hits.length >= 1);
    });

    const jsonAdapter = adapters.find((a) => a.styleMode === 'json');
    const cotAdapter = adapters.find((a) => a.styleMode === 'step-by-step');

    if (matchedCustom && matchedCustom.strength > 0.35) {
      activeAdapters.push(matchedCustom.prompt.slice(0, 42));
      outputText = matchedCustom.targetResponse + quantSuffix;
    } else if (jsonAdapter && (lower.includes('json') || lower.includes('metadata'))) {
      activeAdapters.push('Strict JSON SFT Adapter');
      outputText = JSON.stringify(
        {
          model: arch.name,
          family: arch.family,
          layers: arch.numLayers,
          d_model: arch.dModel,
          gqa_heads: `${arch.numHeads}Q:${arch.numKvHeads}KV`,
          mlp_hidden: arch.mlpIntermediate,
          modified_matrices: profile.modifiedMatricesCount,
          perplexity: profile.estimatedPerplexity,
        },
        null,
        2
      );
    } else if (
      arch.family === 'coding-llm' ||
      lower.includes('quicksort') ||
      lower.includes('python') ||
      lower.includes('code')
    ) {
      activeAdapters.push(
        arch.family === 'coding-llm'
          ? `GQA ${arch.numHeads}Q:${arch.numKvHeads}KV + 1M RoPE`
          : 'Code Synthesis Circuit'
      );
      outputText =
        `\`\`\`python\nfrom typing import List, TypeVar\n\nT = TypeVar("T")\n\ndef quicksort(items: List[T]) -> List[T]:\n    """O(n log n) avg functional quicksort (${arch.name})."""\n    if len(items) <= 1:\n        return items\n    pivot = items[len(items) >> 1]\n    left = [x for x in items if x < pivot]\n    mid = [x for x in items if x == pivot]\n    right = [x for x in items if x > pivot]\n    return quicksort(left) + mid + quicksort(right)\n\`\`\`\n` +
        `• Architecture: ${arch.numHeads} Query heads sharing ${arch.numKvHeads} KV heads (GQA saves ${Math.round((1 - arch.numKvHeads / arch.numHeads) * 100)}% KV-cache VRAM).\n` +
        `• HumanEval Pass@1 Estimate: ${profile.benchmarks.humaneval}% | LiveCodeBench: ${profile.benchmarks.livecodebench}%.${quantSuffix}`;
    } else if (
      (cotAdapter || profile.reasoningDepth > 0.65) &&
      (lower.includes('11,008') || lower.includes('swiglu') || lower.includes('why'))
    ) {
      activeAdapters.push('Reasoning & CoT SFT Circuit');
      outputText =
        `[Step-by-Step Reasoning Trace • ${arch.name}]\n` +
        `Step 1 — Baseline FFN: Classic transformers expand 4× d_model.\n` +
        `Step 2 — SwiGLU 3-Matrix Gating: Using W_gate, W_up, and W_down scales intermediate size to ${arch.mlpIntermediate.toLocaleString()} for hardware tensor-core alignment.\n` +
        `Step 3 — Live Weight Verification: Currently ${profile.modifiedMatricesCount}/${matrices.length} matrices are modified (ΔL2 = ${profile.globalDeltaL2}, KL = ${profile.klDivergence}).${quantSuffix}`;
    } else {
      outputText =
        `[${arch.name} • ΔL2=${profile.globalDeltaL2} • T=${temperature.toFixed(2)} | Top-P=${topP.toFixed(2)}]\n` +
        `Response calibrated across ${matrices.length} tensors (${profile.modifiedMatricesCount} edited):\n` +
        `• Query "${prompt}" activated ${inputTokens.length} tokens across Layers 0–${arch.numLayers - 1}.\n` +
        `• Live Metrics → Reasoning ${(profile.reasoningDepth * 100).toFixed(0)}%, Code ${(profile.codePrecision * 100).toFixed(0)}%, KL Divergence ${profile.klDivergence}.${quantSuffix}`;
    }
  }

  const tokenTraces = buildTokenTraces(
    outputText,
    profile,
    temperature,
    arch.vocabSize,
    arch.family === 'coding-llm' || isFimMode
  );
  const latencyMs = Math.max(18, Math.round(performance.now() - start + 24));

  return {
    inputTokens,
    outputText,
    baseOutputText,
    tokenTraces,
    layerTraces,
    avgPerplexity: profile.estimatedPerplexity,
    klDivergence: profile.klDivergence,
    latencyMs,
    stabilityWarning,
    activeAdapters,
    fimTrace,
    visionTrace,
  };
}
