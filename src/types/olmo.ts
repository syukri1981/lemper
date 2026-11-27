export type ModelFamily =
  | 'general-llm'
  | 'coding-llm'
  | 'multimodal-vlm'
  | 'moe-llm';

export type MatrixKind =
  | 'embed'
  | 'vision_patch'
  | 'vision_attn'
  | 'vision_mlp'
  | 'mm_projector'
  | 'router_gate'
  | 'q_proj'
  | 'k_proj'
  | 'v_proj'
  | 'o_proj'
  | 'gate_proj'
  | 'up_proj'
  | 'down_proj'
  | 'lm_head';

export type MatrixCategory =
  | 'embedding'
  | 'vision'
  | 'connector'
  | 'router'
  | 'attention'
  | 'mlp'
  | 'head';

export type QuantPrecision = 32 | 16 | 8 | 4 | 2;

export type ThreeLayoutMode = 'helix' | 'stack' | 'matrix3d';

export type WeightPresetId =
  | 'base'
  | 'reasoning'
  | 'coder'
  | 'instruct'
  | 'quant4'
  | 'custom';

export interface ModelArchitectureProfile {
  id: string;
  name: string;
  organization: string;
  family: ModelFamily;
  badge: string;
  license: string;
  exactParams: number;
  vocabSize: number;
  dModel: number;
  numLayers: number;
  numHeads: number;
  numKvHeads: number; // < numHeads for GQA (e.g. 4 in Qwen2.5-Coder-7B)
  headDim: number;
  mlpIntermediate: number;
  contextLength: number;
  ropeTheta: number;
  visionConfig?: {
    encoderName: string;
    imageSize: number;
    patchSize: number;
    visionLayers: number;
    visionHiddenDim: number;
    poolingRatio: number;
    projectorType: string;
  };
  codingConfig?: {
    supportsFim: boolean;
    fimTokens: [string, string, string];
    supportsSpeculative: boolean;
    astAwareAttention: boolean;
  };
  moeConfig?: {
    numExperts: number;
    topK: number;
  };
  signatureAlgorithms: {
    name: string;
    formula: string;
    description: string;
    category: 'attention' | 'positional' | 'decoding' | 'multimodal' | 'routing';
  }[];
  sourceFileInfo?: UploadedModelFileSummary;
}

export interface UploadedModelFileSummary {
  fileName: string;
  fileFormat: 'safetensors' | 'gguf' | 'config.json' | 'onnx' | 'pytorch-bin';
  fileSizeBytes: number;
  parsedTensorCount: number;
  realWeightsExtracted: number;
  detectedArchitecture: string;
  quantizationType: string;
  rawHeaderPreview: string;
  uploadedAt: number;
}

export interface TensorMatrixState {
  id: string;
  layerIndex: number; // -20..-2 for vision/connector, -1 for embed, 0..N-1 for decoder layers, N for lm_head
  kind: MatrixKind;
  category: MatrixCategory;
  name: string;
  shortLabel: string;
  shape: [number, number];
  paramCount: number;
  /** 16x16 = 256 sampled scalar weights representing spectral patch of matrix */
  weights: Float32Array;
  /** Original base weights for computing exact ΔW */
  baseWeights: Float32Array;
  /** Explicit LoRA rank attached if fine-tuned via LoRA (0 = dense/base) */
  loraRank: number;
  loraAlpha: number;
  /** Scale multiplier applied directly */
  scaleFactor: number;
  /** Quantization bit width */
  quantBits: QuantPrecision;
  /** Whether user ablated / zeroed out this matrix */
  ablated: boolean;
  /** Derived metrics */
  mean: number;
  std: number;
  l2Norm: number;
  deltaL2: number;
  sparsity: number; // fraction near zero
}

export interface FineTuneSample {
  id: string;
  title: string;
  prompt: string;
  targetResponse: string;
  domain: 'reasoning' | 'code' | 'instruct' | 'knowledge' | 'multimodal' | 'custom';
  enabled: boolean;
}

export interface FineTuneJobConfig {
  method: 'lora' | 'full';
  loraRank: 4 | 8 | 16 | 64;
  learningRate: number;
  steps: number;
  layerStart: number;
  layerEnd: number;
  targetKinds: MatrixKind[];
}

export interface FineTuneStepMetric {
  step: number;
  loss: number;
  gradNorm: number;
  deltaApplied: number;
}

export interface CustomInjectedKnowledge {
  id: string;
  triggerKeywords: string[];
  prompt: string;
  targetResponse: string;
  strength: number;
  styleMode: 'json' | 'pirate' | 'step-by-step' | 'concise' | 'fact' | 'custom';
}

export interface ModelBehavioralProfile {
  totalParams: number;
  modifiedParamsEstimate: number;
  modifiedMatricesCount: number;
  globalDeltaL2: number;
  avgQuantBits: number;
  ablatedCount: number;
  /** 0..1 traits directly driven by live matrix weights */
  reasoningDepth: number;
  codePrecision: number;
  instructionFollowing: number;
  conciseness: number;
  factuality: number;
  creativeEntropy: number;
  visionGrounding: number; // Multimodal ViT + Connector health
  fimInfillAccuracy: number; // Coding FIM & GQA health
  /** Stability & degradation metrics */
  stabilityScore: number;
  explosionRisk: number;
  collapseRisk: number;
  klDivergence: number;
  estimatedPerplexity: number;
  benchmarks: {
    mmlu: number;
    gsm8k: number;
    humaneval: number;
    hellaswag: number;
    truthfulqa: number;
    mmmu?: number; // Multimodal benchmark
    livecodebench?: number; // Coding benchmark
  };
}

export interface TokenCandidate {
  token: string;
  prob: number;
  baseProb: number;
  logit: number;
}

export interface GeneratedTokenTrace {
  token: string;
  tokenId: number;
  prob: number;
  baseProb: number;
  topCandidates: TokenCandidate[];
  speculativeAccepted?: boolean; // For Coding AI Speculative Decoding trace
}

export interface LayerActivationTrace {
  layerIndex: number;
  attnEnergy: number;
  mlpEnergy: number;
  residualNorm: number;
  deltaShift: number;
  status: 'nominal' | 'amplified' | 'ablated' | 'quant-noise';
}

export interface MultimodalPointingCoords {
  label: string;
  xPercent: number;
  yPercent: number;
  confidence: number;
}

export interface InferenceTraceResult {
  inputTokens: { text: string; id: number; modality?: 'text' | 'vision' | 'fim' }[];
  outputText: string;
  baseOutputText: string;
  tokenTraces: GeneratedTokenTrace[];
  layerTraces: LayerActivationTrace[];
  avgPerplexity: number;
  klDivergence: number;
  latencyMs: number;
  stabilityWarning: string | null;
  activeAdapters: string[];
  /** Coding AI specific trace */
  fimTrace?: {
    prefix: string;
    suffix: string;
    infilledMiddle: string;
    kvCacheMbSaved: number;
    speculativeSpeedup: number;
  };
  /** Multimodal VLM specific trace */
  visionTrace?: {
    sceneId: string;
    rawPatches: number;
    pooledTokens: number;
    connectorNorm: number;
    pointingCoords: MultimodalPointingCoords[];
    heatmap8x8: number[];
  };
}

export interface InferenceMessage {
  id: string;
  role: 'user' | 'assistant';
  text: string;
  timestamp: number;
  presetSnapshot: string;
  modeBadge?: string;
  trace?: InferenceTraceResult;
}

export interface PulseSignal {
  id: number;
  direction: 'forward' | 'backward';
  startedAt: number;
  durationMs: number;
  color: string;
  layerStart: number;
  layerEnd: number;
}
