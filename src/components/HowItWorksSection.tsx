import React, { useMemo, useState } from 'react';
import {
  ArrowUpRight,
  BookOpen,
  Box,
  Code2,
  Cpu,
  Database,
  Eye,
  GitBranch,
  Layers,
  Scale,
  Sparkles,
} from 'lucide-react';
import {
  ARCHITECTURE_COMPARISON,
  DOLMA_COMPOSITION,
  HOW_IT_WORKS_STAGES,
  OLMO_7B_SPECS,
} from '../engine/olmoSpecs';
import { tokenizeDolmaBpe } from '../engine/olmoInferenceEngine';
import {
  MatrixKind,
  ModelArchitectureProfile,
  ModelBehavioralProfile,
} from '../types/olmo';
import { BUILTIN_MODELS } from '../engine/modelRegistry';

interface HowItWorksSectionProps {
  architecture: ModelArchitectureProfile;
  profile: ModelBehavioralProfile;
  onHighlightIn3D: (kinds: MatrixKind[], targetKind: MatrixKind) => void;
  onSelectModelArchitecture: (arch: ModelArchitectureProfile) => void;
}

export const HowItWorksSection: React.FC<HowItWorksSectionProps> = ({
  architecture,
  profile,
  onHighlightIn3D,
  onSelectModelArchitecture,
}) => {
  const [activeStageIdx, setActiveStageIdx] = useState(0);
  const [tokenizerInput, setTokenizerInput] = useState(
    '<|fim_prefix|> def solve(x): <|fim_suffix|> return res <|fim_middle|> OLMo-7B & Molmo-7B open weights!'
  );
  const [ropePos, setRopePos] = useState(4);
  const [swigluX, setSwigluX] = useState(1.1);

  // Interactive Coding AI GQA & Context Length Calculator State
  const [contextLenK, setContextLenK] = useState<number>(32); // 32K tokens
  const [kvHeadsSim, setKvHeadsSim] = useState<number>(4); // 4 KV heads (Qwen2.5-Coder)

  // Interactive Multimodal VLM Resolution & Pooling Calculator State
  const [numCropsSim, setNumCropsSim] = useState<number>(4); // 4 multi-scale crops + 1 overview
  const [poolStrideSim, setPoolStrideSim] = useState<number>(2); // 2x2 pooling

  const tokens = useMemo(
    () => tokenizeDolmaBpe(tokenizerInput, architecture.vocabSize),
    [tokenizerInput, architecture.vocabSize]
  );
  const activeStage = HOW_IT_WORKS_STAGES[activeStageIdx];

  const siluVal = swigluX / (1 + Math.exp(-swigluX));
  const swigluOut = siluVal * (swigluX * 0.85);

  // KV Cache Memory math (FP16 = 2 bytes * 2 (K+V) * layers * kvHeads * headDim * seqLen)
  const seqTokens = contextLenK * 1024;
  const fullMhaKvGb = Number(
    ((2 * 2 * architecture.numLayers * architecture.numHeads * architecture.headDim * seqTokens) / 1e9).toFixed(2)
  );
  const gqaKvGb = Number(
    ((2 * 2 * architecture.numLayers * kvHeadsSim * architecture.headDim * seqTokens) / 1e9).toFixed(2)
  );
  const kvSavedPct = Math.round((1 - kvHeadsSim / architecture.numHeads) * 100);

  // Multimodal ViT token count math (336x336 / 14x14 = 24x24 = 576 patches per crop)
  const rawPatchesTotal = numCropsSim * 576;
  const pooledTokensTotal = Math.round(rawPatchesTotal / (poolStrideSim * poolStrideSim));

  const baseBenchmarks = {
    mmlu: 53.2,
    gsm8k: 28.4,
    humaneval: 26.8,
    hellaswag: 76.4,
    truthfulqa: 36.8,
  };

  return (
    <div className="space-y-6 pb-8">
      {/* Hero Banner */}
      <div className="relative overflow-hidden rounded-2xl border border-slate-800 bg-gradient-to-br from-[#0e1729] via-[#0b1120] to-[#150d24] p-6">
        <div className="bg-sci-grid pointer-events-none absolute inset-0 opacity-60" />
        <div className="relative flex flex-col justify-between gap-6 lg:flex-row lg:items-center">
          <div className="max-w-3xl space-y-2">
            <div className="inline-flex items-center gap-2 rounded-full border border-pink-500/30 bg-pink-500/10 px-3 py-1 font-mono text-[11px] text-pink-300">
              <BookOpen className="h-3.5 w-3.5" />
              HOW OPEN-SOURCE LLMS, CODING AI & MULTIMODAL VLMS WORK
            </div>
            <h2 className="text-2xl font-bold tracking-tight text-white sm:text-3xl">
              Inside {architecture.name}: Architecture, Algorithms & 3D Weight Tensors
            </h2>
            <p className="text-sm leading-relaxed text-slate-300">
              Explore the mathematical foundations of <strong className="text-pink-300">General Open LLMs (OLMo-7B)</strong>,{' '}
              <strong className="text-indigo-300">Coding AI Models (Qwen2.5-Coder-7B with GQA, FIM & 1M RoPE)</strong>,{' '}
              <strong className="text-amber-300">Multimodal Vision-Language Models (Molmo-7B-O with ViT-L/14, 2×2 Pooling & 2D Pointing)</strong>, and{' '}
              <strong className="text-emerald-300">Sparse Mixture-of-Experts (OLMoE-1B-7B)</strong>.
            </p>
            <div className="flex flex-wrap gap-2 pt-1 font-mono text-[10px] text-slate-400">
              <span className="rounded border border-slate-800 bg-slate-950/70 px-2 py-0.5">
                Optimizer: {OLMO_7B_SPECS.optimizer}
              </span>
              <span className="rounded border border-slate-800 bg-slate-950/70 px-2 py-0.5">
                Clusters: {OLMO_7B_SPECS.hardware}
              </span>
            </div>
          </div>

          <div className="grid shrink-0 grid-cols-2 gap-2.5 sm:grid-cols-3 lg:w-[400px]">
            {[
              { k: 'Parameters', v: `${(architecture.exactParams / 1e9).toFixed(2)}B`, sub: architecture.badge },
              { k: 'Decoder Layers', v: String(architecture.numLayers), sub: `d_model = ${architecture.dModel}` },
              { k: 'Attention Heads', v: `${architecture.numHeads}Q:${architecture.numKvHeads}KV`, sub: architecture.numKvHeads < architecture.numHeads ? 'GQA Compressed' : 'Full MHA' },
              { k: 'SwiGLU MLP', v: architecture.mlpIntermediate.toLocaleString(), sub: 'Gated FFN' },
              { k: 'Vocab (BPE)', v: architecture.vocabSize.toLocaleString(), sub: 'Untied Head' },
              { k: 'RoPE Theta', v: architecture.ropeTheta >= 1e6 ? '1,000,000' : '10,000', sub: `Ctx ${architecture.contextLength.toLocaleString()}` },
            ].map((item) => (
              <div
                key={item.k}
                className="rounded-xl border border-slate-800/90 bg-slate-950/75 p-2.5 backdrop-blur"
              >
                <div className="text-[10px] uppercase tracking-wider text-slate-400">{item.k}</div>
                <div className="font-mono text-lg font-bold text-cyan-300">{item.v}</div>
                <div className="truncate font-mono text-[10px] text-slate-500">{item.sub}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Specialized Algorithms Observatory: Coding AI vs Multimodal VLM vs Sparse MoE */}
      <div className="rounded-2xl border border-slate-800 bg-[#0d1320] p-5">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-3">
          <div>
            <h3 className="flex items-center gap-2 text-base font-bold text-white">
              <Sparkles className="h-4 w-4 text-pink-400" />
              Specialized AI Model Algorithms: Coding AI, Multimodal VLMs & Sparse MoE
            </h3>
            <p className="text-xs text-slate-400">
              Click any model family below to load its exact 3D weight topology into the Three.js Engine and test its algorithms live.
            </p>
          </div>

          <div className="flex flex-wrap gap-1.5">
            {BUILTIN_MODELS.map((m) => (
              <button
                key={m.id}
                onClick={() => onSelectModelArchitecture(m)}
                className={`rounded-lg px-3 py-1.5 text-xs font-semibold transition ${
                  architecture.id === m.id
                    ? 'bg-cyan-500 text-slate-950 shadow-md shadow-cyan-500/20'
                    : 'border border-slate-800 bg-slate-950 text-slate-300 hover:border-slate-700'
                }`}
              >
                {m.name.split(' ')[0]} ({m.badge})
              </button>
            ))}
          </div>
        </div>

        {/* 2-Column Interactive Algorithm Labs: Coding AI Lab + Multimodal VLM Lab */}
        <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
          {/* Lab A: Coding AI Algorithms (GQA KV-Cache, FIM, 1M RoPE, Speculative Decoding) */}
          <div className="space-y-3 rounded-xl border border-indigo-500/30 bg-indigo-950/10 p-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Code2 className="h-4 w-4 text-indigo-400" />
                <span className="text-sm font-bold text-indigo-200">
                  Coding AI Algorithms (Qwen2.5-Coder / StarCoder2)
                </span>
              </div>
              <button
                onClick={() => {
                  const coder = BUILTIN_MODELS.find((m) => m.family === 'coding-llm');
                  if (coder) onSelectModelArchitecture(coder);
                  onHighlightIn3D(['k_proj', 'v_proj'], 'k_proj');
                }}
                className="flex items-center gap-1 rounded bg-indigo-500/20 px-2.5 py-1 text-[10px] font-semibold text-indigo-200 ring-1 ring-indigo-500/40 hover:bg-indigo-500/30"
              >
                Inspect GQA in 3D
                <ArrowUpRight className="h-3 w-3" />
              </button>
            </div>

            <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
              <div className="rounded-lg border border-slate-800 bg-slate-950/80 p-2.5">
                <div className="font-mono text-[11px] font-semibold text-indigo-300">
                  1. Fill-in-the-Middle (FIM)
                </div>
                <div className="mt-1 font-mono text-[10px] text-emerald-300">
                  Doc → &lt;|fim_prefix|&gt; P &lt;|fim_suffix|&gt; S &lt;|fim_middle|&gt; M
                </div>
                <p className="mt-1 text-[11px] text-slate-400">
                  Moves the middle code span to the end of the sequence during pretraining so causal self-attention attends to both preceding imports and following return types.
                </p>
              </div>

              <div className="rounded-lg border border-slate-800 bg-slate-950/80 p-2.5">
                <div className="font-mono text-[11px] font-semibold text-cyan-300">
                  2. Long-Context RoPE (θ = 10⁶) & Speculative Decoding
                </div>
                <div className="mt-1 font-mono text-[10px] text-amber-300">
                  θᵢ = 1,000,000^(-2i/d) • Draft γ=5 Verify
                </div>
                <p className="mt-1 text-[11px] text-slate-400">
                  100× wider rotary wavelength resolves cross-file AST definitions up to 128K tokens, while speculative decoding verifies 5 draft syntax tokens in 1 target pass.
                </p>
              </div>
            </div>

            {/* Interactive GQA KV-Cache VRAM Simulator */}
            <div className="rounded-lg border border-slate-800 bg-slate-950/90 p-3">
              <div className="mb-2 flex items-center justify-between text-xs">
                <span className="font-semibold text-slate-200">
                  Interactive Grouped-Query Attention (GQA) KV-Cache Calculator
                </span>
                <span className="font-mono text-emerald-400">
                  {kvSavedPct}% VRAM Saved
                </span>
              </div>

              <div className="grid grid-cols-2 gap-3 text-[11px]">
                <div>
                  <div className="flex justify-between text-slate-400">
                    <span>Repo Context Length</span>
                    <span className="font-mono text-cyan-300">{contextLenK}K tokens</span>
                  </div>
                  <input
                    type="range"
                    min={4}
                    max={128}
                    step={4}
                    value={contextLenK}
                    onChange={(e) => setContextLenK(parseInt(e.target.value, 10))}
                    className="sci-range mt-1 w-full"
                  />
                </div>
                <div>
                  <div className="flex justify-between text-slate-400">
                    <span>KV Heads (vs {architecture.numHeads}Q)</span>
                    <span className="font-mono text-indigo-300">{kvHeadsSim} KV Heads</span>
                  </div>
                  <input
                    type="range"
                    min={1}
                    max={architecture.numHeads}
                    step={1}
                    value={kvHeadsSim}
                    onChange={(e) => setKvHeadsSim(parseInt(e.target.value, 10))}
                    className="sci-range mt-1 w-full"
                  />
                </div>
              </div>

              <div className="mt-2.5 grid grid-cols-2 gap-2 font-mono text-xs">
                <div className="rounded bg-slate-900 p-2">
                  <div className="text-[10px] text-slate-500">Standard Full MHA KV-Cache</div>
                  <div className="text-rose-300">{fullMhaKvGb} GB VRAM</div>
                </div>
                <div className="rounded bg-slate-900 p-2">
                  <div className="text-[10px] text-slate-500">
                    GQA ({architecture.numHeads}Q : {kvHeadsSim}KV) KV-Cache
                  </div>
                  <div className="text-emerald-300">{gqaKvGb} GB VRAM</div>
                </div>
              </div>
            </div>
          </div>

          {/* Lab B: Multimodal AI Algorithms (ViT Patchification, 2x2 Attention Pooling, 2D Pointing) */}
          <div className="space-y-3 rounded-xl border border-amber-500/30 bg-amber-950/10 p-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Eye className="h-4 w-4 text-amber-400" />
                <span className="text-sm font-bold text-amber-200">
                  Multimodal VLM Algorithms (Molmo-7B-O / PixMo)
                </span>
              </div>
              <button
                onClick={() => {
                  const vlm = BUILTIN_MODELS.find((m) => m.family === 'multimodal-vlm');
                  if (vlm) onSelectModelArchitecture(vlm);
                  onHighlightIn3D(['vision_patch', 'vision_attn', 'mm_projector'], 'mm_projector');
                }}
                className="flex items-center gap-1 rounded bg-amber-500/20 px-2.5 py-1 text-[10px] font-semibold text-amber-200 ring-1 ring-amber-500/40 hover:bg-amber-500/30"
              >
                Inspect ViT & Connector in 3D
                <ArrowUpRight className="h-3 w-3" />
              </button>
            </div>

            <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
              <div className="rounded-lg border border-slate-800 bg-slate-950/80 p-2.5">
                <div className="font-mono text-[11px] font-semibold text-amber-300">
                  1. ViT-L/14 + 2×2 Attention Pooling Connector
                </div>
                <div className="mt-1 font-mono text-[10px] text-cyan-300">
                  v_llm = MLP(AttnPool_2×2(ViT(I))) ∈ ℝ¹⁴⁴ˣ⁴⁰⁹⁶
                </div>
                <p className="mt-1 text-[11px] text-slate-400">
                  Encodes 336×336px image crops into 14×14 patches, pools 2×2 neighborhoods via attention, and projects from 1,024-D ViT space into 4,096-D LLM space.
                </p>
              </div>

              <div className="rounded-lg border border-slate-800 bg-slate-950/80 p-2.5">
                <div className="font-mono text-[11px] font-semibold text-pink-300">
                  2. Native 2D Pointing & Spatial Grounding
                </div>
                <div className="mt-1 font-mono text-[10px] text-emerald-300">
                  &lt;point x=&quot;74.5&quot; y=&quot;68.2&quot; alt=&quot;cta_btn&quot;&gt;
                </div>
                <p className="mt-1 text-[11px] text-slate-400">
                  Instead of brittle bounding-box regressors, Molmo outputs normalized (x, y) coordinates directly in text tokens for UI agents, robotics, and counting.
                </p>
              </div>
            </div>

            {/* Interactive Vision Token & Pooling Simulator */}
            <div className="rounded-lg border border-slate-800 bg-slate-950/90 p-3">
              <div className="mb-2 flex items-center justify-between text-xs">
                <span className="font-semibold text-slate-200">
                  Interactive Multi-Crop Vision Token & Pooling Calculator
                </span>
                <span className="font-mono text-amber-300">
                  {poolStrideSim}×{poolStrideSim} Spatial Pool ({poolStrideSim * poolStrideSim}× Reduction)
                </span>
              </div>

              <div className="grid grid-cols-2 gap-3 text-[11px]">
                <div>
                  <div className="flex justify-between text-slate-400">
                    <span>Multi-Scale Image Crops</span>
                    <span className="font-mono text-amber-300">{numCropsSim} crops</span>
                  </div>
                  <input
                    type="range"
                    min={1}
                    max={9}
                    step={1}
                    value={numCropsSim}
                    onChange={(e) => setNumCropsSim(parseInt(e.target.value, 10))}
                    className="sci-range mt-1 w-full"
                  />
                </div>
                <div>
                  <div className="flex justify-between text-slate-400">
                    <span>Connector Pooling Stride</span>
                    <span className="font-mono text-cyan-300">
                      {poolStrideSim}×{poolStrideSim} patch window
                    </span>
                  </div>
                  <input
                    type="range"
                    min={1}
                    max={4}
                    step={1}
                    value={poolStrideSim}
                    onChange={(e) => setPoolStrideSim(parseInt(e.target.value, 10))}
                    className="sci-range mt-1 w-full"
                  />
                </div>
              </div>

              <div className="mt-2.5 grid grid-cols-2 gap-2 font-mono text-xs">
                <div className="rounded bg-slate-900 p-2">
                  <div className="text-[10px] text-slate-500">Raw ViT-L/14 Patches (1024-D)</div>
                  <div className="text-amber-300">{rawPatchesTotal.toLocaleString()} Vision Patches</div>
                </div>
                <div className="rounded bg-slate-900 p-2">
                  <div className="text-[10px] text-slate-500">Projected LLM Tokens (4096-D)</div>
                  <div className="text-emerald-300">{pooledTokensTotal.toLocaleString()} LLM Tokens</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Exact Parameter Accounting Bar */}
      <div className="rounded-2xl border border-slate-800 bg-[#0d1320] p-5">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
          <div>
            <h3 className="flex items-center gap-2 text-sm font-semibold text-white">
              <Cpu className="h-4 w-4 text-cyan-400" />
              Where All 6,887,899,136 Weights Live Inside OLMo-7B
            </h3>
            <p className="text-xs text-slate-400">
              Because OLMo uses non-parametric LayerNorm and zero bias terms, 100% of learnable parameters reside in pure 2D weight matrices.
            </p>
          </div>
          <span className="font-mono text-xs text-emerald-300">Click any group to highlight in 3D</span>
        </div>

        <div className="flex h-7 w-full overflow-hidden rounded-lg border border-slate-800">
          <div
            style={{ width: '62.84%' }}
            className="flex items-center justify-center bg-gradient-to-r from-pink-600 to-amber-500 font-mono text-[11px] font-semibold text-white"
          >
            SwiGLU MLP (62.8%) — 4.33B
          </div>
          <div
            style={{ width: '31.18%' }}
            className="flex items-center justify-center bg-gradient-to-r from-cyan-600 to-blue-600 font-mono text-[11px] font-semibold text-white"
          >
            MHA Q/K/V/O (31.2%) — 2.15B
          </div>
          <div style={{ width: '2.99%' }} className="bg-indigo-500" />
          <div style={{ width: '2.99%' }} className="bg-emerald-500" />
        </div>

        <div className="mt-3 grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-5">
          {[
            {
              label: 'SwiGLU MLP (96 matrices)',
              val: '4,328,521,728',
              pct: '62.84%',
              color: 'bg-pink-500',
              kind: 'gate_proj' as MatrixKind,
            },
            {
              label: 'Attention QKVO (128 matrices)',
              val: '2,147,483,648',
              pct: '31.18%',
              color: 'bg-cyan-500',
              kind: 'q_proj' as MatrixKind,
            },
            {
              label: 'Input Embed (1 matrix)',
              val: '205,946,880',
              pct: '2.99%',
              color: 'bg-indigo-500',
              kind: 'embed' as MatrixKind,
            },
            {
              label: 'Untied LM Head (1 matrix)',
              val: '205,946,880',
              pct: '2.99%',
              color: 'bg-emerald-500',
              kind: 'lm_head' as MatrixKind,
            },
            {
              label: 'Non-Param LN & Biases',
              val: '0',
              pct: '0.00%',
              color: 'bg-slate-600',
              kind: 'q_proj' as MatrixKind,
            },
          ].map((row) => (
            <button
              key={row.label}
              onClick={() => onHighlightIn3D([row.kind], row.kind)}
              className="flex items-center justify-between rounded-lg border border-slate-800/80 bg-slate-950/70 px-3 py-2 text-left transition hover:border-cyan-500/40"
            >
              <div>
                <div className="flex items-center gap-1.5 text-[11px] font-medium text-slate-200">
                  <span className={`h-2.5 w-2.5 rounded-sm ${row.color}`} />
                  {row.label}
                </div>
                <div className="mt-0.5 font-mono text-[10px] text-slate-400">
                  {row.val} ({row.pct})
                </div>
              </div>
              <ArrowUpRight className="h-3.5 w-3.5 text-slate-500" />
            </button>
          ))}
        </div>
      </div>

      {/* Interactive 6-Stage Architecture Pipeline */}
      <div className="grid grid-cols-1 gap-5 lg:grid-cols-12">
        <div className="space-y-2 lg:col-span-5">
          <div className="mb-1 flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-slate-400">
            <Layers className="h-3.5 w-3.5 text-pink-400" />
            6-Stage Forward Pass Walkthrough
          </div>
          {HOW_IT_WORKS_STAGES.map((stage, idx) => {
            const active = idx === activeStageIdx;
            return (
              <button
                key={stage.id}
                onClick={() => setActiveStageIdx(idx)}
                className={`w-full rounded-xl border p-3 text-left transition ${
                  active
                    ? 'border-cyan-500/60 bg-slate-900 shadow-lg shadow-cyan-950/30'
                    : 'border-slate-800/80 bg-[#0d1320] hover:border-slate-700'
                }`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2.5">
                    <span
                      className="flex h-6 w-6 items-center justify-center rounded-full font-mono text-xs font-bold text-slate-950"
                      style={{ backgroundColor: stage.color }}
                    >
                      {stage.stepNumber}
                    </span>
                    <div>
                      <div className="text-xs font-semibold text-slate-100">{stage.title}</div>
                      <div className="font-mono text-[10px] text-slate-400">{stage.subtitle}</div>
                    </div>
                  </div>
                </div>
              </button>
            );
          })}
        </div>

        <div className="rounded-2xl border border-slate-800 bg-[#0d1320] p-5 lg:col-span-7">
          <div className="flex flex-wrap items-start justify-between gap-3 border-b border-slate-800 pb-4">
            <div>
              <div
                className="font-mono text-[11px] font-semibold uppercase tracking-wider"
                style={{ color: activeStage.color }}
              >
                Stage {activeStage.stepNumber} of 6 • {activeStage.paramSummary}
              </div>
              <h3 className="mt-1 text-lg font-bold text-white">{activeStage.title}</h3>
            </div>
            <button
              onClick={() =>
                onHighlightIn3D(activeStage.matrixKinds, activeStage.matrixKinds[0])
              }
              className="flex items-center gap-1.5 rounded-lg bg-cyan-500/20 px-3 py-1.5 text-xs font-semibold text-cyan-200 ring-1 ring-cyan-500/40 transition hover:bg-cyan-500/30"
            >
              <Box className="h-3.5 w-3.5" />
              Highlight Weights in 3D Engine
            </button>
          </div>

          <div className="mt-4 space-y-4">
            <div className="rounded-lg border border-slate-800 bg-slate-950/90 px-3.5 py-2.5 font-mono text-xs text-amber-300">
              {activeStage.formula}
            </div>

            <p className="text-sm leading-relaxed text-slate-300">{activeStage.explanation}</p>

            <div className="rounded-xl border border-pink-500/25 bg-pink-950/15 p-3 text-xs text-pink-200">
              <span className="font-semibold text-pink-300">Open Science Note: </span>
              {activeStage.openScienceNote}
            </div>

            {activeStage.id === 'tokenize' && (
              <div className="rounded-xl border border-slate-800 bg-slate-950/75 p-3.5">
                <div className="mb-2 flex items-center justify-between">
                  <span className="text-xs font-semibold text-cyan-300">
                    Live BPE & FIM Tokenizer Sandbox ({architecture.vocabSize.toLocaleString()} Vocab)
                  </span>
                  <span className="font-mono text-[10px] text-slate-400">
                    {tokens.length} tokens
                  </span>
                </div>
                <input
                  type="text"
                  value={tokenizerInput}
                  onChange={(e) => setTokenizerInput(e.target.value)}
                  className="mb-2.5 w-full rounded-lg border border-slate-700 bg-slate-900 px-3 py-1.5 font-mono text-xs text-slate-100 focus:border-cyan-500 focus:outline-none"
                />
                <div className="flex flex-wrap gap-1.5">
                  {tokens.map((t, i) => (
                    <span
                      key={i}
                      className={`rounded border px-2 py-0.5 font-mono text-[11px] ${
                        t.modality === 'fim'
                          ? 'border-indigo-500/50 bg-indigo-950/40 text-indigo-200'
                          : 'border-cyan-500/30 bg-cyan-950/30 text-cyan-200'
                      }`}
                    >
                      {t.text} <span className="text-[9px] text-slate-400">[{t.id}]</span>
                    </span>
                  ))}
                </div>
              </div>
            )}

            {activeStage.id === 'norm-rope' && (
              <div className="rounded-xl border border-slate-800 bg-slate-950/75 p-3.5">
                <div className="mb-2 flex items-center justify-between">
                  <span className="text-xs font-semibold text-violet-300">
                    Interactive RoPE Rotation Visualizer (Token Position m = {ropePos})
                  </span>
                  <input
                    type="range"
                    min={0}
                    max={24}
                    value={ropePos}
                    onChange={(e) => setRopePos(parseInt(e.target.value, 10))}
                    className="sci-range w-32"
                  />
                </div>
                <div className="flex items-center gap-4">
                  <svg viewBox="0 0 120 120" className="h-24 w-24 shrink-0 rounded-lg bg-slate-900 p-1">
                    <circle cx="60" cy="60" r="46" fill="none" stroke="#334155" strokeDasharray="3 3" />
                    <line x1="60" y1="60" x2="106" y2="60" stroke="#475569" strokeWidth="1" />
                    {(() => {
                      const angle = ropePos * 0.28;
                      const x2 = 60 + Math.cos(angle) * 44;
                      const y2 = 60 - Math.sin(angle) * 44;
                      return (
                        <>
                          <line x1="60" y1="60" x2={x2} y2={y2} stroke="#a78bfa" strokeWidth="2.5" />
                          <circle cx={x2} cy={y2} r="4" fill="#22d3ee" />
                        </>
                      );
                    })()}
                  </svg>
                  <div className="text-xs text-slate-300">
                    Rotating Query and Key 2D pairs by angle <code className="text-violet-300">m·θ</code>{' '}
                    means the attention dot-product <code className="text-cyan-300">qₘ · kₙ</code> depends strictly
                    on relative offset <code className="text-emerald-300">(m − n)</code>. Coding models expand{' '}
                    <code className="text-indigo-300">θ = 1,000,000</code> for 128K context windows.
                  </div>
                </div>
              </div>
            )}

            {activeStage.id === 'swiglu' && (
              <div className="rounded-xl border border-slate-800 bg-slate-950/75 p-3.5">
                <div className="mb-2 flex items-center justify-between">
                  <span className="text-xs font-semibold text-amber-300">
                    Interactive SwiGLU Gate Simulator (Input x = {swigluX.toFixed(2)})
                  </span>
                  <input
                    type="range"
                    min={-3.0}
                    max={3.0}
                    step={0.1}
                    value={swigluX}
                    onChange={(e) => setSwigluX(parseFloat(e.target.value))}
                    className="sci-range w-36"
                  />
                </div>
                <div className="grid grid-cols-3 gap-2 font-mono text-xs">
                  <div className="rounded bg-slate-900 p-2">
                    <div className="text-[10px] text-slate-500">SiLU(x·W_gate)</div>
                    <div className="text-amber-300">{siluVal.toFixed(3)}</div>
                  </div>
                  <div className="rounded bg-slate-900 p-2">
                    <div className="text-[10px] text-slate-500">Up (x·W_up)</div>
                    <div className="text-cyan-300">{(swigluX * 0.85).toFixed(3)}</div>
                  </div>
                  <div className="rounded bg-slate-900 p-2">
                    <div className="text-[10px] text-slate-500">Gated Output</div>
                    <div className="text-emerald-300">{swigluOut.toFixed(3)}</div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Live Fine-Tune Impact on Evaluation Benchmarks + Dolma Corpus */}
      <div className="grid grid-cols-1 gap-5 lg:grid-cols-12">
        <div className="rounded-2xl border border-slate-800 bg-[#0d1320] p-5 lg:col-span-6">
          <div className="mb-3 flex items-center justify-between">
            <h3 className="flex items-center gap-2 text-sm font-semibold text-white">
              <Sparkles className="h-4 w-4 text-emerald-400" />
              Live Benchmark Shift ({architecture.name.split(' ')[0]} 3D Weights)
            </h3>
            <span className="font-mono text-[11px] text-cyan-300">
              PPL: {profile.estimatedPerplexity}
            </span>
          </div>
          <div className="space-y-3">
            {[
              { name: 'MMLU (5-shot Knowledge)', val: profile.benchmarks.mmlu, base: baseBenchmarks.mmlu },
              { name: 'GSM8K (Math Reasoning)', val: profile.benchmarks.gsm8k, base: baseBenchmarks.gsm8k },
              { name: 'HumanEval (Python Code Synthesis)', val: profile.benchmarks.humaneval, base: baseBenchmarks.humaneval },
              { name: 'LiveCodeBench (FIM & Repo Coding)', val: profile.benchmarks.livecodebench ?? 24.0, base: 24.0 },
              { name: 'MMMU (Multimodal Visual Reasoning)', val: profile.benchmarks.mmmu ?? 22.0, base: 22.0 },
            ].map((b) => {
              const diff = Number((b.val - b.base).toFixed(1));
              return (
                <div key={b.name} className="space-y-1">
                  <div className="flex justify-between font-mono text-xs">
                    <span className="text-slate-300">{b.name}</span>
                    <span>
                      <strong className="text-white">{b.val}%</strong>{' '}
                      <span
                        className={
                          diff > 0
                            ? 'text-emerald-400'
                            : diff < 0
                            ? 'text-rose-400'
                            : 'text-slate-500'
                        }
                      >
                        ({diff >= 0 ? `+${diff}` : diff}%)
                      </span>
                    </span>
                  </div>
                  <div className="relative h-2.5 overflow-hidden rounded-full bg-slate-900">
                    <div
                      className="absolute inset-y-0 left-0 rounded-full bg-slate-700/60"
                      style={{ width: `${b.base}%` }}
                    />
                    <div
                      className={`absolute inset-y-0 left-0 rounded-full transition-all duration-300 ${
                        diff >= 0 ? 'bg-gradient-to-r from-cyan-500 to-emerald-400' : 'bg-rose-500'
                      }`}
                      style={{ width: `${b.val}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Dolma Open Pretraining Dataset Composition */}
        <div className="rounded-2xl border border-slate-800 bg-[#0d1320] p-5 lg:col-span-6">
          <div className="mb-3 flex items-center justify-between">
            <h3 className="flex items-center gap-2 text-sm font-semibold text-white">
              <Database className="h-4 w-4 text-pink-400" />
              Dolma & PixMo Open Pretraining Corpora
            </h3>
            <span className="font-mono text-[10px] text-slate-400">100% Open Datasets</span>
          </div>
          <div className="mb-3 flex h-4 w-full overflow-hidden rounded-full border border-slate-800">
            {DOLMA_COMPOSITION.map((slice) => (
              <div
                key={slice.source}
                style={{ width: `${slice.percent}%`, backgroundColor: slice.color }}
                title={`${slice.source}: ${slice.percent}%`}
              />
            ))}
          </div>
          <div className="space-y-2">
            {DOLMA_COMPOSITION.map((slice) => (
              <div
                key={slice.source}
                className="flex items-start justify-between gap-2 rounded-lg border border-slate-800/70 bg-slate-950/60 px-3 py-2"
              >
                <div className="flex items-start gap-2">
                  <span
                    className="mt-1 h-2.5 w-2.5 shrink-0 rounded-sm"
                    style={{ backgroundColor: slice.color }}
                  />
                  <div>
                    <div className="text-xs font-medium text-slate-200">{slice.source}</div>
                    <div className="text-[11px] text-slate-400">{slice.description}</div>
                  </div>
                </div>
                <div className="shrink-0 text-right font-mono text-[11px]">
                  <div className="text-cyan-300">{slice.percent}%</div>
                  <div className="text-[10px] text-slate-500">{slice.tokens}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Peer Architecture Comparison Table */}
      <div className="rounded-2xl border border-slate-800 bg-[#0d1320] p-5">
        <div className="mb-3 flex items-center justify-between">
          <h3 className="flex items-center gap-2 text-sm font-semibold text-white">
            <Scale className="h-4 w-4 text-indigo-400" />
            OLMo-7B vs Peer 7B–8B Foundation Models
          </h3>
          <span className="flex items-center gap-1 font-mono text-[11px] text-emerald-300">
            <GitBranch className="h-3.5 w-3.5" />
            Apache 2.0 Reproducibility
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-slate-800 font-mono text-[11px] text-slate-400">
                <th className="py-2 pr-4">Architectural Choice</th>
                <th className="py-2 pr-4 text-pink-300">OLMo-7B (AI2)</th>
                <th className="py-2 pr-4">Llama 2 7B</th>
                <th className="py-2 pr-4">Falcon 7B</th>
                <th className="py-2">PaLM 8B</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/70">
              {ARCHITECTURE_COMPARISON.map((row) => (
                <tr key={row.feature} className="hover:bg-slate-900/40">
                  <td className="py-2 pr-4 font-medium text-slate-300">{row.feature}</td>
                  <td className="py-2 pr-4 font-mono text-cyan-300">{row.olmo7b}</td>
                  <td className="py-2 pr-4 font-mono text-slate-400">{row.llama2_7b}</td>
                  <td className="py-2 pr-4 font-mono text-slate-400">{row.falcon7b}</td>
                  <td className="py-2 font-mono text-slate-400">{row.palm8b}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
