import { ModelArchitectureProfile } from '../types/olmo';

export const BUILTIN_MODELS: ModelArchitectureProfile[] = [
  {
    id: 'olmo-7b',
    name: 'OLMo-7B (Base / Instruct)',
    organization: 'Allen Institute for AI (AI2)',
    family: 'general-llm',
    badge: 'Open Foundation LLM',
    license: 'Apache 2.0 (Weights + Dolma Data)',
    exactParams: 6_887_899_136,
    vocabSize: 50_280,
    dModel: 4096,
    numLayers: 32,
    numHeads: 32,
    numKvHeads: 32,
    headDim: 128,
    mlpIntermediate: 11_008,
    contextLength: 4096,
    ropeTheta: 10_000,
    signatureAlgorithms: [
      {
        name: 'Non-Parametric LayerNorm & Zero-Bias Linear',
        formula: 'LN(x) = (x - μ) / √(σ² + ε)   (no γ, β)',
        description:
          'Eliminates learnable scale/shift vectors in LayerNorm and removes all linear bias terms across all 32 blocks to prevent FP16/BF16 loss spikes.',
        category: 'attention',
      },
      {
        name: 'SwiGLU Gated Feed-Forward Network',
        formula: 'FFN(x) = (SiLU(x·W_gate) ⊙ (x·W_up)) · W_down',
        description:
          'Expands 4,096 hidden dimension to 11,008 (~8/3 d_model aligned to 128) using a multiplicative SiLU gate that stores 62.8% of the model’s knowledge.',
        category: 'attention',
      },
      {
        name: 'Untied Input Embedding & Output LM Head',
        formula: 'W_embed ≠ W_headᵀ ∈ ℝ⁵⁰²⁸⁰ˣ⁴⁰⁹⁶',
        description:
          'Separating input subword representations from output logit projections improves gradient conditioning during 2.5T token Dolma pretraining.',
        category: 'decoding',
      },
    ],
  },
  {
    id: 'qwen25-coder-7b',
    name: 'Qwen2.5-Coder-7B (Code AI)',
    organization: 'Qwen / Open-Source Code LLM',
    family: 'coding-llm',
    badge: 'Coding AI Model',
    license: 'Apache 2.0 (5.5T Code Tokens)',
    exactParams: 7_615_616_512,
    vocabSize: 152_064,
    dModel: 3584,
    numLayers: 28,
    numHeads: 28,
    numKvHeads: 4, // 7:1 Grouped-Query Attention compression!
    headDim: 128,
    mlpIntermediate: 18_944,
    contextLength: 131_072,
    ropeTheta: 1_000_000,
    codingConfig: {
      supportsFim: true,
      fimTokens: ['<|fim_prefix|>', '<|fim_suffix|>', '<|fim_middle|>'],
      supportsSpeculative: true,
      astAwareAttention: true,
    },
    signatureAlgorithms: [
      {
        name: 'Fill-in-the-Middle (FIM) PSM & SPM Objective',
        formula: 'P(M | <|fim_prefix|> P <|fim_suffix|> S <|fim_middle|>)',
        description:
          'Rearranges document spans into Prefix-Suffix-Middle during training so the model can synthesize missing functions or cursor completions conditioned on both preceding and following code.',
        category: 'decoding',
      },
      {
        name: 'Grouped-Query Attention (GQA 28Q : 4KV)',
        formula: 'W_Q ∈ ℝ³⁵⁸⁴ˣ³⁵⁸⁴,   W_K, W_V ∈ ℝ⁵¹²ˣ³⁵⁸⁴ (7× KV Compression)',
        description:
          'Shares every Key/Value head across 7 Query heads, shrinking KV-cache memory by 85.7% so 128K-token multi-file repositories fit in GPU VRAM.',
        category: 'attention',
      },
      {
        name: 'Long-Context RoPE (θ = 1,000,000 + YaRN)',
        formula: 'θᵢ = 1,000,000^(-2i/128)',
        description:
          'Increases base rotary frequency 100× over standard LLMs (10k → 1M) to preserve cross-file symbol resolution and AST dependency tracking across 128K tokens.',
        category: 'positional',
      },
      {
        name: 'Draft-Target Speculative Decoding',
        formula: 'Accept draft token x̃ ~ q(x) with prob min(1, p_target(x̃)/q_draft(x̃))',
        description:
          'Uses a lightweight draft model to propose γ=5 syntax tokens in parallel, verified in a single forward pass of the 7B target weights for 2.6× lossless coding speedup.',
        category: 'decoding',
      },
    ],
  },
  {
    id: 'molmo-7b-o',
    name: 'Molmo-7B-O (Vision-Language VLM)',
    organization: 'Allen Institute for AI (AI2)',
    family: 'multimodal-vlm',
    badge: 'Multimodal AI Model',
    license: 'Apache 2.0 (Open Weights + PixMo Data)',
    exactParams: 7_214_820_352,
    vocabSize: 50_280,
    dModel: 4096,
    numLayers: 32,
    numHeads: 32,
    numKvHeads: 32,
    headDim: 128,
    mlpIntermediate: 11_008,
    contextLength: 4096,
    ropeTheta: 10_000,
    visionConfig: {
      encoderName: 'CLIP ViT-L/14 @ 336px (Multi-Scale Crop)',
      imageSize: 336,
      patchSize: 14,
      visionLayers: 6, // Representative ViT stage blocks rendered in 3D before connector
      visionHiddenDim: 1024,
      poolingRatio: 4, // 2x2 attention pooling reduces 576 -> 144 tokens
      projectorType: '2×2 Attention Pooling + SwiGLU MLP Connector [1024 → 4096]',
    },
    signatureAlgorithms: [
      {
        name: 'Multi-Scale Multi-Crop ViT Patchification',
        formula: 'I ∈ ℝ³³⁶ˣ³³⁶ˣ³ → Conv14×14 → 576 Visual Patch Vectors ∈ ℝ¹⁰²⁴',
        description:
          'Splits high-resolution images into overlapping multi-scale crops plus a global overview, projecting each 14×14 pixel patch into the Vision Transformer stream.',
        category: 'multimodal',
      },
      {
        name: '2×2 Attention Pooling & Cross-Modal MLP Connector',
        formula: 'v_llm = MLP_proj(AttnPool_2×2(ViT(I))) ∈ ℝ¹⁴⁴ˣ⁴⁰⁹⁶',
        description:
          'Compresses 2×2 neighboring vision patches via learned multi-head attention pooling before projecting from 1,024-D ViT space into the 4,096-D OLMo residual stream.',
        category: 'multimodal',
      },
      {
        name: 'Native 2D Spatial Pointing & Grounding',
        formula: 'Output → <point x="64.2" y="38.5" alt="target_object">',
        description:
          'Trained on AI2 PixMo dense pointing annotations so the model grounds visual answers, UI clicks, and counting tasks by emitting exact (x, y) normalized coordinates.',
        category: 'multimodal',
      },
    ],
  },
  {
    id: 'olmoe-1b-7b',
    name: 'OLMoE-1B-7B (Sparse Mixture-of-Experts)',
    organization: 'Allen Institute for AI (AI2)',
    family: 'moe-llm',
    badge: 'Sparse MoE Model',
    license: 'Apache 2.0 (1.3B Active / 6.9B Total)',
    exactParams: 6_919_094_272,
    vocabSize: 50_304,
    dModel: 2048,
    numLayers: 16,
    numHeads: 16,
    numKvHeads: 16,
    headDim: 128,
    mlpIntermediate: 8192,
    contextLength: 4096,
    ropeTheta: 10_000,
    moeConfig: {
      numExperts: 64,
      topK: 8,
    },
    signatureAlgorithms: [
      {
        name: 'Dropless Top-K Sparse Router Gate',
        formula: 'G(x) = Softmax(TopK(x · W_router, k=8))',
        description:
          'Routes each token dynamically to only 8 out of 64 specialized SwiGLU experts per layer—delivering 7B knowledge capacity at 1.3B inference FLOPs.',
        category: 'routing',
      },
      {
        name: 'Auxiliary Load-Balancing & Router Z-Loss',
        formula: 'L_aux = α · N ∑ fᵢ · Pᵢ + β · log²(∑ exp(zⱼ))',
        description:
          'Prevents expert routing collapse during training so all 64 experts specialize across code, math, syntax, and multilingual reasoning.',
        category: 'routing',
      },
    ],
  },
];
