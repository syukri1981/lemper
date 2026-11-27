import { MatrixKind } from '../types/olmo';

export const OLMO_7B_SPECS = {
  modelName: 'OLMo-7B (AllenAI Open Language Model)',
  license: 'Apache 2.0 (Weights, Code, Training Data & Eval)',
  exactParams: 6_887_899_136,
  vocabSize: 50_280,
  dModel: 4096,
  numLayers: 32,
  numHeads: 32,
  headDim: 128,
  mlpIntermediate: 11_008, // ~8/3 * 4096 rounded to multiple of 128
  contextLength: 2048,
  trainingTokens: '2.46T – 2.5T Tokens (Dolma)',
  layerNormType: 'Non-Parametric LayerNorm (0 learnable params)',
  positionalEmbeddings: 'RoPE (Rotary Positional Embeddings)',
  activation: 'SwiGLU (Gated Linear Unit with SiLU)',
  biases: 'None (No bias terms in Linear or LN)',
  weightTying: 'Untied (Separate Input Embedding & LM Head)',
  optimizer: 'AdamW (Peak LR 3.0e-4, β=(0.9, 0.95), wd=0.1)',
  hardware: 'LUMI (256 nodes × 4 AMD MI250X) & MosaicML A100',
};

export interface ArchitectureComparisonRow {
  feature: string;
  olmo7b: string;
  llama2_7b: string;
  falcon7b: string;
  palm8b: string;
}

export const ARCHITECTURE_COMPARISON: ArchitectureComparisonRow[] = [
  { feature: 'd_model (Hidden Dim)', olmo7b: '4,096', llama2_7b: '4,096', falcon7b: '4,544', palm8b: '4,096' },
  { feature: 'Transformer Layers', olmo7b: '32', llama2_7b: '32', falcon7b: '32', palm8b: '32' },
  { feature: 'Attention Heads', olmo7b: '32 (Full MHA)', llama2_7b: '32 (Full/GQA)', falcon7b: '71 (MQA)', palm8b: '16 (MQA)' },
  { feature: 'MLP Intermediate Size', olmo7b: '11,008 (~8/3 d)', llama2_7b: '11,008 (~8/3 d)', falcon7b: '18,176 (4× d)', palm8b: '16,384 (4× d)' },
  { feature: 'Activation Function', olmo7b: 'SwiGLU', llama2_7b: 'SwiGLU', falcon7b: 'GeLU', palm8b: 'SwiGLU' },
  { feature: 'LayerNorm Variant', olmo7b: 'Non-parametric LN', llama2_7b: 'RMSNorm', falcon7b: 'Parametric LN', palm8b: 'Parametric LN' },
  { feature: 'Positional Encoding', olmo7b: 'RoPE', llama2_7b: 'RoPE', falcon7b: 'RoPE', palm8b: 'RoPE' },
  { feature: 'Linear Biases', olmo7b: 'None', llama2_7b: 'None', falcon7b: 'LN only', palm8b: 'None' },
  { feature: 'Weight Tying', olmo7b: 'No (Untied)', llama2_7b: 'No (Untied)', falcon7b: 'No (Untied)', palm8b: 'Yes (Tied)' },
  { feature: 'Open Training Data', olmo7b: '100% Open (Dolma)', llama2_7b: 'Closed', falcon7b: 'Partial', palm8b: 'Closed' },
  { feature: 'Intermediate Checkpoints', olmo7b: '500+ Open (every 1k steps)', llama2_7b: 'None', falcon7b: 'None', palm8b: 'None' },
];

export interface DolmaSourceSlice {
  source: string;
  tokens: string;
  percent: number;
  color: string;
  description: string;
}

export const DOLMA_COMPOSITION: DolmaSourceSlice[] = [
  {
    source: 'Common Crawl (Web)',
    tokens: '2.48T tokens',
    percent: 82.0,
    color: '#38bdf8',
    description: 'Deduplicated, language & PII-filtered English web pages.',
  },
  {
    source: 'The Stack (Code)',
    tokens: '411B tokens',
    percent: 11.2,
    color: '#f0529c',
    description: 'Permissively licensed GitHub repositories across 80+ programming languages.',
  },
  {
    source: 'Semantic Scholar (peS2o)',
    tokens: '70B tokens',
    percent: 2.8,
    color: '#a78bfa',
    description: 'Open-access academic papers, STEM proofs, and scientific literature from AI2.',
  },
  {
    source: 'Reddit & C4 / Wikipedia',
    tokens: '95B tokens',
    percent: 3.1,
    color: '#34d399',
    description: 'Curated discussions, encyclopedic facts, and Wikibooks.',
  },
  {
    source: 'Project Gutenberg',
    tokens: '5.3B tokens',
    percent: 0.9,
    color: '#fbbf24',
    description: 'Long-form public domain books for extended narrative coherence.',
  },
];

export interface HowItWorksStage {
  id: string;
  stepNumber: number;
  title: string;
  subtitle: string;
  formula: string;
  paramSummary: string;
  matrixKinds: MatrixKind[];
  color: string;
  explanation: string;
  openScienceNote: string;
}

export const HOW_IT_WORKS_STAGES: HowItWorksStage[] = [
  {
    id: 'tokenize',
    stepNumber: 1,
    title: 'BPE Tokenization & Dolma Pretraining',
    subtitle: 'Modified GPT-NeoX-20B Tokenizer • 50,280 Vocabulary',
    formula: 'Text → BPE Subwords [t₀, t₁, ..., tₙ] ∈ {0..50279}',
    paramSummary: '0 weights (Deterministic byte-pair table)',
    matrixKinds: ['embed'],
    color: '#38bdf8',
    explanation:
      'Raw text is split into subword integer IDs using a modified GPT-NeoX-20B Byte-Pair Encoding (BPE) vocabulary of 50,280 tokens (padded to 50,304 for GPU tensor-core alignment). Special PII masking tokens replace emails and phone numbers during Dolma curation.',
    openScienceNote:
      'Unlike closed weights models, AI2 releases the entire Dolma data pipeline, deduplication filters, and token mixture ratios under Apache 2.0.',
  },
  {
    id: 'embed',
    stepNumber: 2,
    title: 'Untied Token Embedding Matrix',
    subtitle: 'embed_tokens.weight [50,280 × 4,096]',
    formula: 'h₀ = E[token_id] ∈ ℝ⁴⁰⁹⁶',
    paramSummary: '205,946,880 weights (3.0% of OLMo-7B)',
    matrixKinds: ['embed'],
    color: '#22d3ee',
    explanation:
      'Each discrete token ID indexes a 4,096-dimensional continuous vector from the Embedding lookup matrix. OLMo-7B deliberately unties input embeddings from the output LM head—empirically improving training stability and downstream perplexity.',
    openScienceNote:
      'In the 3D Engine, embed_tokens sits at the base of the transformer stack, projecting discrete tokens into the 4,096-D residual stream.',
  },
  {
    id: 'norm-rope',
    stepNumber: 3,
    title: 'Non-Parametric LayerNorm & RoPE',
    subtitle: 'Zero Bias Terms • Rotary Positional Embeddings',
    formula: 'x̂ = (x - μ) / √(σ² + ε),   qₘ = R(mθ)·q',
    paramSummary: '0 learnable weights (pure geometric invariance)',
    matrixKinds: ['q_proj', 'k_proj'],
    color: '#a78bfa',
    explanation:
      'Before every Attention and MLP sub-block, OLMo normalizes activations with Non-Parametric LayerNorm (no learnable γ/β scale/shift parameters) and eliminates ALL linear bias terms. Position is injected by rotating Query & Key pairs in 2D subspaces via RoPE angles θᵢ = 10000^(-2i/128).',
    openScienceNote:
      'Removing LayerNorm weights and biases prevents divergence spikes at 7B scale without sacrificing expressivity.',
  },
  {
    id: 'attention',
    stepNumber: 4,
    title: 'Full Multi-Head Self-Attention (32 Layers × 32 Heads)',
    subtitle: 'W_Q, W_K, W_V, W_O ∈ ℝ⁴⁰⁹⁶ˣ⁴⁰⁹⁶ per layer',
    formula: 'Attn(Q,K,V) = softmax((Q·Kᵀ)/√128 + Mask) · V',
    paramSummary: '67.1M / layer → 2.15B total (31.2% of OLMo-7B)',
    matrixKinds: ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    color: '#f0529c',
    explanation:
      'Each of the 32 layers projects the 4096-D residual stream through 4 square matrices (Q, K, V, O) split across 32 heads of 128 dims each. Unlike Llama-2-70B or Falcon (GQA/MQA), OLMo-7B uses full 32-head KV matrices for maximum representational fidelity. Early layers track syntax; middle layers resolve entities and reasoning.',
    openScienceNote:
      'Fine-tuning Q and V projections via LoRA (ΔW = B·A) changes what tokens the model attends to during context retrieval.',
  },
  {
    id: 'swiglu',
    stepNumber: 5,
    title: 'SwiGLU Gated MLP (Knowledge & Reasoning Engine)',
    subtitle: 'W_gate, W_up [11,008 × 4,096] & W_down [4,096 × 11,008]',
    formula: 'SwiGLU(x) = (SiLU(x · W_gate) ⊙ (x · W_up)) · W_down',
    paramSummary: '135.3M / layer → 4.33B total (62.8% of OLMo-7B!)',
    matrixKinds: ['gate_proj', 'up_proj', 'down_proj'],
    color: '#fbbf24',
    explanation:
      'Nearly TWO-THIRDS of all weights in OLMo-7B live inside the 32 SwiGLU MLP blocks! The hidden dimension expands from 4,096 to 11,008 (~8/3 × d_model, aligned to 128). W_gate acts as a learned multiplicative switch filtering which facts and reasoning circuits fire into W_down.',
    openScienceNote:
      'Mechanistic interpretability research shows factual associations and arithmetic circuits are heavily localized inside middle-layer MLP gate/down weights.',
  },
  {
    id: 'lm-head',
    stepNumber: 6,
    title: 'Untied LM Head & Next-Token Softmax',
    subtitle: 'lm_head.weight [50,280 × 4,096] → Probability Distribution',
    formula: 'P(tₙ₊₁ | t≤ₙ) = softmax(LayerNorm(h₃₂) · W_headᵀ / T)',
    paramSummary: '205,946,880 weights (3.0% of OLMo-7B)',
    matrixKinds: ['lm_head'],
    color: '#10b981',
    explanation:
      'After 32 sequential residual updates, the final normalized 4,096-D vector is multiplied against the untied LM Head to produce 50,280 raw logits. Temperature scaling and Top-P nucleus sampling convert logits into the next emitted token, which loops autoregressively.',
    openScienceNote:
      'Scaling or fine-tuning lm_head directly shifts vocabulary preference, tone, and formatting constraints like strict JSON or concise output.',
  },
];
