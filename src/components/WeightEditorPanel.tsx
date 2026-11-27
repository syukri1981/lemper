import React, { useMemo, useState } from 'react';
import {
  Activity,
  Brush,
  Cpu,
  Eraser,
  Flame,
  Layers,
  Play,
  Plus,
  RotateCcw,
  Scissors,
  Sliders,
  Sparkles,
  Trash2,
  Zap,
} from 'lucide-react';
import {
  FineTuneJobConfig,
  FineTuneSample,
  FineTuneStepMetric,
  MatrixKind,
  ModelArchitectureProfile,
  QuantPrecision,
  TensorMatrixState,
} from '../types/olmo';
import { LAYER_MATRIX_KINDS, MatrixTransformOp } from '../engine/olmoWeightEngine';

interface WeightEditorPanelProps {
  architecture: ModelArchitectureProfile;
  matrices: TensorMatrixState[];
  selectedMatrixId: string;
  onSelectMatrix: (id: string) => void;
  onPaintCell: (
    matrixId: string,
    row: number,
    col: number,
    radius: number,
    delta: number,
    mode: 'add' | 'zero'
  ) => void;
  onTransformMatrix: (matrixId: string, op: MatrixTransformOp) => void;
  onBatchTransform: (
    filter: (m: TensorMatrixState) => boolean,
    op: MatrixTransformOp
  ) => void;
  samples: FineTuneSample[];
  onToggleSample: (id: string) => void;
  onAddCustomSample: (sample: Omit<FineTuneSample, 'id'>) => void;
  onDeleteSample: (id: string) => void;
  ftConfig: FineTuneJobConfig;
  onChangeFtConfig: (cfg: FineTuneJobConfig) => void;
  ftHistory: FineTuneStepMetric[];
  onRunFineTune: () => void;
  onInspectIn3D: () => void;
}

function cellColorCss(w: number, delta: number, ablated: boolean): string {
  if (ablated) return 'rgb(30, 35, 48)';
  if (Math.abs(delta) > 0.08) {
    const t = Math.min(1, Math.abs(delta) * 1.4);
    return w >= 0
      ? `rgba(16, 185, 129, ${0.42 + t * 0.58})`
      : `rgba(34, 211, 238, ${0.42 + t * 0.58})`;
  }
  const c = Math.max(-1.1, Math.min(1.1, w));
  if (c >= 0) {
    const t = c / 1.1;
    const r = Math.round(45 + t * 195);
    const g = Math.round(35 + t * 50);
    const b = Math.round(62 + t * 94);
    return `rgb(${r}, ${g}, ${b})`;
  } else {
    const t = -c / 1.1;
    const r = Math.round(20 + t * 14);
    const g = Math.round(45 + t * 140);
    const b = Math.round(75 + t * 165);
    return `rgb(${r}, ${g}, ${b})`;
  }
}

export const WeightEditorPanel: React.FC<WeightEditorPanelProps> = ({
  architecture,
  matrices,
  selectedMatrixId,
  onSelectMatrix,
  onPaintCell,
  onTransformMatrix,
  onBatchTransform,
  samples,
  onToggleSample,
  onAddCustomSample,
  onDeleteSample,
  ftConfig,
  onChangeFtConfig,
  ftHistory,
  onRunFineTune,
  onInspectIn3D,
}) => {
  const [activeTab, setActiveTab] = useState<'direct' | 'sft' | 'atlas'>('direct');
  const [brushMode, setBrushMode] = useState<'boost' | 'suppress' | 'zero'>('boost');
  const [brushRadius, setBrushRadius] = useState<number>(1.2);
  const [isPainting, setIsPainting] = useState(false);
  const [hoveredCell, setHoveredCell] = useState<{ r: number; c: number; w: number; d: number } | null>(null);
  const [loraRankSelect, setLoraRankSelect] = useState<4 | 8 | 16 | 64>(16);

  const [customPrompt, setCustomPrompt] = useState('');
  const [customTarget, setCustomTarget] = useState('');

  const selectedMatrix = useMemo(
    () => matrices.find((m) => m.id === selectedMatrixId) ?? matrices[0],
    [matrices, selectedMatrixId]
  );

  // Group layer summaries (including Vision ViT & Connector if Multimodal)
  const layerSummaries = useMemo(() => {
    const out: {
      layerIndex: number;
      label: string;
      delta: number;
      ablated: boolean;
      defaultId: string;
      isVisionOrConn?: boolean;
    }[] = [];

    const visionMats = matrices.filter((m) => m.category === 'vision');
    if (visionMats.length > 0) {
      out.push({
        layerIndex: -10,
        label: 'ViT',
        delta: visionMats.reduce((a, b) => a + b.deltaL2, 0),
        ablated: visionMats.some((m) => m.ablated),
        defaultId: visionMats[0].id,
        isVisionOrConn: true,
      });
    }

    const connMats = matrices.filter((m) => m.category === 'connector');
    if (connMats.length > 0) {
      out.push({
        layerIndex: -2,
        label: 'Conn',
        delta: connMats.reduce((a, b) => a + b.deltaL2, 0),
        ablated: connMats.some((m) => m.ablated),
        defaultId: connMats[0].id,
        isVisionOrConn: true,
      });
    }

    const embed = matrices.find((m) => m.kind === 'embed');
    if (embed) {
      out.push({
        layerIndex: -1,
        label: 'Emb',
        delta: embed.deltaL2,
        ablated: embed.ablated,
        defaultId: embed.id,
      });
    }

    for (let l = 0; l < architecture.numLayers; l++) {
      const lm = matrices.filter((m) => m.layerIndex === l);
      if (lm.length === 0) continue;
      const delta = lm.reduce((a, b) => a + b.deltaL2, 0);
      const ablated = lm.some((m) => m.ablated);
      out.push({
        layerIndex: l,
        label: `L${String(l).padStart(2, '0')}`,
        delta,
        ablated,
        defaultId: lm[0].id,
      });
    }

    const head = matrices.find((m) => m.kind === 'lm_head');
    if (head) {
      out.push({
        layerIndex: architecture.numLayers,
        label: 'Head',
        delta: head.deltaL2,
        ablated: head.ablated,
        defaultId: head.id,
      });
    }
    return out;
  }, [matrices, architecture.numLayers]);

  const activeLayerMatrices = useMemo(() => {
    if (!selectedMatrix) return [];
    if (selectedMatrix.category === 'vision') {
      return matrices.filter((m) => m.category === 'vision');
    }
    if (selectedMatrix.category === 'connector') {
      return matrices.filter((m) => m.category === 'connector');
    }
    if (selectedMatrix.layerIndex < 0 || selectedMatrix.layerIndex >= architecture.numLayers) {
      return [selectedMatrix];
    }
    return matrices.filter((m) => m.layerIndex === selectedMatrix.layerIndex);
  }, [matrices, selectedMatrix, architecture.numLayers]);

  const handleCellPaint = (r: number, c: number) => {
    if (!selectedMatrix) return;
    const delta = brushMode === 'boost' ? 0.28 : brushMode === 'suppress' ? -0.28 : 0;
    const mode = brushMode === 'zero' ? 'zero' : 'add';
    onPaintCell(selectedMatrix.id, r, c, brushRadius, delta, mode);
  };

  const handleCreateCustomSample = (e: React.FormEvent) => {
    e.preventDefault();
    if (!customPrompt.trim() || !customTarget.trim()) return;
    onAddCustomSample({
      title: `Custom: ${customPrompt.trim().slice(0, 34)}`,
      prompt: customPrompt.trim(),
      targetResponse: customTarget.trim(),
      domain: 'custom',
      enabled: true,
    });
    setCustomPrompt('');
    setCustomTarget('');
  };

  const targetKindChoices = useMemo(() => {
    const base: MatrixKind[] = [...LAYER_MATRIX_KINDS, 'lm_head'];
    if (architecture.family === 'multimodal-vlm') {
      return ['vision_patch', 'vision_attn', 'mm_projector', ...base] as MatrixKind[];
    }
    if (architecture.family === 'moe-llm') {
      return ['router_gate', ...base] as MatrixKind[];
    }
    return base;
  }, [architecture.family]);

  if (!selectedMatrix) return null;

  return (
    <div
      className="flex h-full flex-col overflow-hidden rounded-xl border border-slate-800/90 bg-[#0d1320]"
      onPointerUp={() => setIsPainting(false)}
      onPointerLeave={() => setIsPainting(false)}
    >
      {/* Top Sub-Navigation */}
      <div className="flex items-center justify-between border-b border-slate-800 bg-slate-950/60 px-3 py-2">
        <div className="flex items-center gap-1">
          <button
            onClick={() => setActiveTab('direct')}
            className={`flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-xs font-medium transition ${
              activeTab === 'direct'
                ? 'bg-pink-500/20 text-pink-300 ring-1 ring-pink-500/40'
                : 'text-slate-400 hover:bg-slate-800/60 hover:text-slate-200'
            }`}
          >
            <Sliders className="h-3.5 w-3.5" />
            Direct Weight Surgery
          </button>
          <button
            onClick={() => setActiveTab('sft')}
            className={`flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-xs font-medium transition ${
              activeTab === 'sft'
                ? 'bg-emerald-500/20 text-emerald-300 ring-1 ring-emerald-500/40'
                : 'text-slate-400 hover:bg-slate-800/60 hover:text-slate-200'
            }`}
          >
            <Flame className="h-3.5 w-3.5" />
            SFT / LoRA Backprop
          </button>
          <button
            onClick={() => setActiveTab('atlas')}
            className={`flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-xs font-medium transition ${
              activeTab === 'atlas'
                ? 'bg-cyan-500/20 text-cyan-300 ring-1 ring-cyan-500/40'
                : 'text-slate-400 hover:bg-slate-800/60 hover:text-slate-200'
            }`}
          >
            <Layers className="h-3.5 w-3.5" />
            All {matrices.length} Tensors
          </button>
        </div>

        <button
          onClick={onInspectIn3D}
          className="hidden items-center gap-1 rounded-md border border-slate-700/80 bg-slate-900 px-2 py-1 text-[11px] text-cyan-300 hover:border-cyan-500/50 sm:flex"
          title="Open Selected Tensor in 3D Heightfield Mode"
        >
          <Sparkles className="h-3 w-3" />
          3D Heightfield
        </button>
      </div>

      {/* Compact Layer Ribbon */}
      <div className="border-b border-slate-800/80 bg-slate-950/40 px-3 py-2">
        <div className="mb-1 flex items-center justify-between text-[10px] text-slate-400">
          <span className="font-mono uppercase tracking-wider">Layer Navigator (Click to Inspect Tensor)</span>
          <span className="font-mono text-cyan-300">
            {selectedMatrix.category === 'vision'
              ? 'Vision ViT Encoder'
              : selectedMatrix.category === 'connector'
              ? 'Vision-LLM Connector'
              : selectedMatrix.layerIndex === -1
              ? 'Input Embedding'
              : selectedMatrix.layerIndex === architecture.numLayers
              ? 'LM Head'
              : `Layer ${selectedMatrix.layerIndex} / ${architecture.numLayers - 1}`}
          </span>
        </div>
        <div className="grid grid-cols-9 gap-1 2xl:grid-cols-12">
          {layerSummaries.map((ls) => {
            const isSel =
              (ls.label === 'ViT' && selectedMatrix.category === 'vision') ||
              (ls.label === 'Conn' && selectedMatrix.category === 'connector') ||
              (ls.layerIndex >= -1 && ls.layerIndex === selectedMatrix.layerIndex);

            return (
              <button
                key={ls.label}
                onClick={() => {
                  if (ls.layerIndex >= 0 && ls.layerIndex < architecture.numLayers) {
                    const sameKind = matrices.find(
                      (m) => m.layerIndex === ls.layerIndex && m.kind === selectedMatrix.kind
                    );
                    onSelectMatrix(sameKind ? sameKind.id : ls.defaultId);
                  } else {
                    onSelectMatrix(ls.defaultId);
                  }
                }}
                className={`rounded py-0.5 text-center font-mono text-[9px] transition ${
                  isSel
                    ? 'bg-cyan-500 text-slate-950 font-bold ring-1 ring-cyan-300'
                    : ls.ablated
                    ? 'bg-rose-950/80 text-rose-300 ring-1 ring-rose-500/50'
                    : ls.delta > 0.15
                    ? 'bg-emerald-500/25 text-emerald-300 ring-1 ring-emerald-500/40'
                    : ls.isVisionOrConn
                    ? 'bg-amber-500/20 text-amber-300 ring-1 ring-amber-500/30 hover:bg-amber-500/30'
                    : 'bg-slate-900 text-slate-400 hover:bg-slate-800 hover:text-slate-200'
                }`}
              >
                {ls.label}
              </button>
            );
          })}
        </div>

        {/* Sub-module pills inside selected Layer / Vision / Connector */}
        {activeLayerMatrices.length > 1 && (
          <div className="mt-1.5 flex flex-wrap items-center gap-1">
            {activeLayerMatrices.map((m) => {
              const isSel = m.id === selectedMatrix.id;
              return (
                <button
                  key={m.id}
                  onClick={() => onSelectMatrix(m.id)}
                  className={`rounded px-2 py-0.5 font-mono text-[10px] transition ${
                    isSel
                      ? 'bg-pink-500 text-white font-semibold'
                      : m.ablated
                      ? 'bg-rose-950/70 text-rose-300'
                      : m.deltaL2 > 0.03
                      ? 'bg-emerald-500/20 text-emerald-300 ring-1 ring-emerald-500/30'
                      : 'bg-slate-900/90 text-slate-400 hover:bg-slate-800'
                  }`}
                >
                  {m.shortLabel}
                  {m.category === 'vision' ? ` (${m.name.split('.').slice(-2, -1)[0]})` : ''}
                </button>
              );
            })}
          </div>
        )}
      </div>

      {/* Main Scrollable Tab Body */}
      <div className="flex-1 overflow-y-auto p-3">
        {activeTab === 'direct' && (
          <div className="space-y-3">
            {/* Tensor Header Metrics */}
            <div className="rounded-lg border border-slate-800 bg-slate-950/70 p-2.5">
              <div className="flex items-start justify-between gap-2">
                <div>
                  <div className="font-mono text-xs font-semibold text-cyan-300 break-all">
                    {selectedMatrix.name}
                  </div>
                  <div className="mt-0.5 font-mono text-[10px] text-slate-400">
                    Shape [{selectedMatrix.shape[0].toLocaleString()} × {selectedMatrix.shape[1].toLocaleString()}] •{' '}
                    {(selectedMatrix.paramCount / 1e6).toFixed(2)}M weights
                    {(selectedMatrix.kind === 'k_proj' || selectedMatrix.kind === 'v_proj') &&
                      selectedMatrix.shape[0] < selectedMatrix.shape[1] && (
                        <span className="ml-1.5 rounded bg-indigo-500/20 px-1.5 py-0.5 text-indigo-300">
                          GQA Compressed ({architecture.numKvHeads} KV Heads)
                        </span>
                      )}
                  </div>
                </div>
                <button
                  onClick={() => onTransformMatrix(selectedMatrix.id, { type: 'reset' })}
                  className="flex items-center gap-1 rounded border border-slate-700 bg-slate-900 px-2 py-1 text-[10px] text-slate-300 hover:border-slate-500 hover:text-white"
                >
                  <RotateCcw className="h-3 w-3" />
                  Reset Tensor
                </button>
              </div>

              <div className="mt-2 grid grid-cols-4 gap-1.5 text-center font-mono">
                <div className="rounded bg-slate-900/90 px-1.5 py-1">
                  <div className="text-[9px] text-slate-500">L2 NORM</div>
                  <div className="text-xs font-semibold text-slate-200">{selectedMatrix.l2Norm.toFixed(2)}</div>
                </div>
                <div className="rounded bg-slate-900/90 px-1.5 py-1">
                  <div className="text-[9px] text-slate-500">ΔW SHIFT</div>
                  <div
                    className={`text-xs font-semibold ${
                      selectedMatrix.deltaL2 > 0.02 ? 'text-emerald-400' : 'text-slate-400'
                    }`}
                  >
                    {selectedMatrix.deltaL2.toFixed(3)}
                  </div>
                </div>
                <div className="rounded bg-slate-900/90 px-1.5 py-1">
                  <div className="text-[9px] text-slate-500">PRECISION</div>
                  <div className="text-xs font-semibold text-amber-300">
                    {selectedMatrix.ablated ? 'ZERO' : `${selectedMatrix.quantBits}-bit`}
                  </div>
                </div>
                <div className="rounded bg-slate-900/90 px-1.5 py-1">
                  <div className="text-[9px] text-slate-500">SPARSITY</div>
                  <div className="text-xs font-semibold text-slate-200">
                    {(selectedMatrix.sparsity * 100).toFixed(0)}%
                  </div>
                </div>
              </div>
            </div>

            {/* Interactive 16x16 Weight Matrix Direct Paint Grid */}
            <div className="rounded-lg border border-slate-800 bg-slate-950/70 p-2.5">
              <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
                <span className="text-[11px] font-medium text-slate-300">
                  Paint Directly Into Weight Patch (16×16 Sub-Tensor)
                </span>
                <div className="flex items-center gap-1">
                  <button
                    onClick={() => setBrushMode('boost')}
                    className={`flex items-center gap-1 rounded px-2 py-0.5 text-[10px] font-medium ${
                      brushMode === 'boost'
                        ? 'bg-emerald-500/25 text-emerald-300 ring-1 ring-emerald-500/50'
                        : 'bg-slate-900 text-slate-400'
                    }`}
                  >
                    <Brush className="h-3 w-3" />
                    +w
                  </button>
                  <button
                    onClick={() => setBrushMode('suppress')}
                    className={`flex items-center gap-1 rounded px-2 py-0.5 text-[10px] font-medium ${
                      brushMode === 'suppress'
                        ? 'bg-cyan-500/25 text-cyan-300 ring-1 ring-cyan-500/50'
                        : 'bg-slate-900 text-slate-400'
                    }`}
                  >
                    <Brush className="h-3 w-3" />
                    -w
                  </button>
                  <button
                    onClick={() => setBrushMode('zero')}
                    className={`flex items-center gap-1 rounded px-2 py-0.5 text-[10px] font-medium ${
                      brushMode === 'zero'
                        ? 'bg-rose-500/25 text-rose-300 ring-1 ring-rose-500/50'
                        : 'bg-slate-900 text-slate-400'
                    }`}
                  >
                    <Eraser className="h-3 w-3" />
                    Prune
                  </button>
                  <button
                    onClick={() => setBrushRadius((r) => (r < 0.5 ? 1.3 : 0))}
                    className="rounded bg-slate-900 px-2 py-0.5 font-mono text-[10px] text-slate-300 hover:bg-slate-800"
                    title="Toggle 1x1 cell vs 3x3 patch brush"
                  >
                    {brushRadius < 0.5 ? '1×1' : '3×3'}
                  </button>
                </div>
              </div>

              <div
                className="mx-auto grid w-full max-w-[272px] select-none gap-[2px] rounded border border-slate-800 bg-slate-900 p-1.5"
                style={{ gridTemplateColumns: 'repeat(16, minmax(0, 1fr))' }}
              >
                {Array.from({ length: 256 }).map((_, idx) => {
                  const r = Math.floor(idx / 16);
                  const c = idx % 16;
                  const w = selectedMatrix.ablated ? 0 : selectedMatrix.weights[idx];
                  const delta = w - selectedMatrix.baseWeights[idx];
                  return (
                    <div
                      key={idx}
                      onPointerDown={(e) => {
                        e.preventDefault();
                        setIsPainting(true);
                        handleCellPaint(r, c);
                      }}
                      onPointerEnter={() => {
                        setHoveredCell({ r, c, w, d: delta });
                        if (isPainting) handleCellPaint(r, c);
                      }}
                      style={{ backgroundColor: cellColorCss(w, delta, selectedMatrix.ablated) }}
                      className="aspect-square cursor-crosshair rounded-[2px] transition-transform hover:scale-125 hover:ring-1 hover:ring-white"
                    />
                  );
                })}
              </div>

              <div className="mt-1.5 flex h-4 items-center justify-between font-mono text-[10px] text-slate-400">
                {hoveredCell ? (
                  <>
                    <span>
                      W[{hoveredCell.r},{hoveredCell.c}] ={' '}
                      <span className="text-slate-100">{hoveredCell.w.toFixed(4)}</span>
                    </span>
                    <span className={Math.abs(hoveredCell.d) > 0.01 ? 'text-emerald-400' : 'text-slate-500'}>
                      Δ = {hoveredCell.d >= 0 ? '+' : ''}
                      {hoveredCell.d.toFixed(4)}
                    </span>
                  </>
                ) : (
                  <span>Click or drag across the matrix grid to modify scalar weights in real time</span>
                )}
              </div>
            </div>

            {/* Precision Tensor Operations */}
            <div className="space-y-2.5 rounded-lg border border-slate-800 bg-slate-950/70 p-2.5">
              <div>
                <div className="mb-1 flex items-center justify-between text-[11px]">
                  <span className="text-slate-300">Matrix Gain Multiplier (Scale Weights)</span>
                  <span
                    className={`font-mono font-semibold ${
                      selectedMatrix.scaleFactor > 1.8 ? 'text-amber-400' : 'text-cyan-300'
                    }`}
                  >
                    {selectedMatrix.scaleFactor.toFixed(2)}×
                  </span>
                </div>
                <input
                  type="range"
                  min={0.1}
                  max={2.6}
                  step={0.05}
                  value={selectedMatrix.scaleFactor}
                  onChange={(e) =>
                    onTransformMatrix(selectedMatrix.id, {
                      type: 'scale',
                      scaleFactor: parseFloat(e.target.value),
                    })
                  }
                  className="sci-range w-full"
                />
                <div className="mt-0.5 flex justify-between font-mono text-[9px] text-slate-500">
                  <span>0.1× Dampen</span>
                  <span>1.0× Nominal</span>
                  <span>2.6× Activation Explosion</span>
                </div>
              </div>

              {/* LoRA Adapter Injection */}
              <div className="flex flex-wrap items-center justify-between gap-2 border-t border-slate-800/80 pt-2">
                <div className="flex items-center gap-1.5">
                  <span className="text-[11px] text-slate-300">Inject LoRA ΔW:</span>
                  {([4, 8, 16, 64] as const).map((r) => (
                    <button
                      key={r}
                      onClick={() => setLoraRankSelect(r)}
                      className={`rounded px-1.5 py-0.5 font-mono text-[10px] ${
                        loraRankSelect === r
                          ? 'bg-pink-500/25 text-pink-300 ring-1 ring-pink-500/50'
                          : 'bg-slate-900 text-slate-400'
                      }`}
                    >
                      r={r}
                    </button>
                  ))}
                </div>
                <button
                  onClick={() =>
                    onTransformMatrix(selectedMatrix.id, {
                      type: 'lora',
                      rank: loraRankSelect,
                      alpha: loraRankSelect * 2,
                      seedOffset: Math.floor(Math.random() * 999),
                    })
                  }
                  className="flex items-center gap-1 rounded bg-pink-500/20 px-2.5 py-1 text-[11px] font-medium text-pink-200 ring-1 ring-pink-500/40 hover:bg-pink-500/30"
                >
                  <Zap className="h-3 w-3" />
                  Apply B·A
                </button>
              </div>

              {/* Quantization Strip */}
              <div className="flex flex-wrap items-center justify-between gap-2 border-t border-slate-800/80 pt-2">
                <span className="text-[11px] text-slate-300">Quantization:</span>
                <div className="flex items-center gap-1">
                  {([32, 16, 8, 4, 2] as QuantPrecision[]).map((bits) => (
                    <button
                      key={bits}
                      onClick={() =>
                        onTransformMatrix(selectedMatrix.id, { type: 'quantize', bits })
                      }
                      className={`rounded px-1.5 py-0.5 font-mono text-[10px] ${
                        selectedMatrix.quantBits === bits && !selectedMatrix.ablated
                          ? 'bg-amber-500/25 text-amber-300 ring-1 ring-amber-500/50'
                          : 'bg-slate-900 text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      {bits === 32 ? 'FP32' : bits === 16 ? 'BF16' : `INT${bits}`}
                    </button>
                  ))}
                </div>
              </div>

              {/* Quick Destructive / Surgical Actions */}
              <div className="grid grid-cols-3 gap-1.5 border-t border-slate-800/80 pt-2">
                <button
                  onClick={() =>
                    onTransformMatrix(selectedMatrix.id, { type: 'prune', threshold: 0.16 })
                  }
                  className="flex items-center justify-center gap-1 rounded bg-slate-900 px-2 py-1.5 text-[10px] text-slate-300 ring-1 ring-slate-800 hover:bg-slate-800"
                >
                  <Scissors className="h-3 w-3 text-cyan-400" />
                  Prune |w|&lt;0.16
                </button>
                <button
                  onClick={() =>
                    onTransformMatrix(selectedMatrix.id, { type: 'noise', sigma: 0.22 })
                  }
                  className="flex items-center justify-center gap-1 rounded bg-slate-900 px-2 py-1.5 text-[10px] text-slate-300 ring-1 ring-slate-800 hover:bg-slate-800"
                >
                  <Activity className="h-3 w-3 text-amber-400" />
                  Inject Noise
                </button>
                <button
                  onClick={() =>
                    onTransformMatrix(selectedMatrix.id, {
                      type: 'ablate',
                      ablated: !selectedMatrix.ablated,
                    })
                  }
                  className={`flex items-center justify-center gap-1 rounded px-2 py-1.5 text-[10px] ring-1 ${
                    selectedMatrix.ablated
                      ? 'bg-emerald-500/20 text-emerald-300 ring-emerald-500/40'
                      : 'bg-rose-500/15 text-rose-300 ring-rose-500/35 hover:bg-rose-500/25'
                  }`}
                >
                  <Cpu className="h-3 w-3" />
                  {selectedMatrix.ablated ? 'Restore Tensor' : 'Ablate (Zero)'}
                </button>
              </div>

              {/* Model-Specific Stress Tests */}
              <div className="grid grid-cols-2 gap-1.5 pt-1">
                {architecture.family === 'multimodal-vlm' ? (
                  <button
                    onClick={() =>
                      onBatchTransform(
                        (m) => m.category === 'connector',
                        { type: 'ablate', ablated: true }
                      )
                    }
                    className="rounded border border-rose-500/30 bg-rose-950/30 px-2 py-1 text-[10px] text-rose-300 hover:bg-rose-950/60"
                  >
                    Ablate Vision Connector
                  </button>
                ) : (
                  <button
                    onClick={() =>
                      onBatchTransform(
                        (m) => m.layerIndex >= 10 && m.layerIndex <= 20,
                        { type: 'ablate', ablated: true }
                      )
                    }
                    className="rounded border border-rose-500/30 bg-rose-950/30 px-2 py-1 text-[10px] text-rose-300 hover:bg-rose-950/60"
                  >
                    Stress Test: Ablate L10–L20
                  </button>
                )}
                <button
                  onClick={() =>
                    onBatchTransform(
                      () => true,
                      { type: 'scale', scaleFactor: 2.25 }
                    )
                  }
                  className="rounded border border-amber-500/30 bg-amber-950/30 px-2 py-1 text-[10px] text-amber-300 hover:bg-amber-950/60"
                >
                  Stress Test: Over-Scale 2.25×
                </button>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'sft' && (
          <div className="space-y-3">
            <div className="rounded-lg border border-slate-800 bg-slate-950/70 p-2.5">
              <div className="mb-2 flex items-center justify-between">
                <span className="text-xs font-semibold text-slate-200">
                  1. Training Dataset (Supervised Fine-Tuning Pairs)
                </span>
                <span className="font-mono text-[10px] text-emerald-400">
                  {samples.filter((s) => s.enabled).length} selected
                </span>
              </div>

              <div className="space-y-1.5">
                {samples.map((s) => (
                  <div
                    key={s.id}
                    className={`rounded-md border p-2 transition ${
                      s.enabled
                        ? 'border-emerald-500/45 bg-emerald-950/20'
                        : 'border-slate-800 bg-slate-900/50'
                    }`}
                  >
                    <div className="flex items-start justify-between gap-2">
                      <label className="flex flex-1 cursor-pointer items-start gap-2">
                        <input
                          type="checkbox"
                          checked={s.enabled}
                          onChange={() => onToggleSample(s.id)}
                          className="mt-0.5 accent-emerald-400"
                        />
                        <div className="min-w-0 flex-1">
                          <div className="flex items-center gap-1.5">
                            <span className="truncate text-[11px] font-medium text-slate-200">
                              {s.title}
                            </span>
                            <span className="rounded bg-slate-800 px-1 py-0.2 font-mono text-[9px] text-slate-400">
                              {s.domain}
                            </span>
                          </div>
                          <div className="mt-0.5 truncate font-mono text-[10px] text-cyan-300/90">
                            Q: {s.prompt}
                          </div>
                          <div className="truncate font-mono text-[10px] text-slate-400">
                            A: {s.targetResponse}
                          </div>
                        </div>
                      </label>
                      {s.domain === 'custom' && (
                        <button
                          onClick={() => onDeleteSample(s.id)}
                          className="text-slate-500 hover:text-rose-400"
                          title="Delete custom training example"
                        >
                          <Trash2 className="h-3.5 w-3.5" />
                        </button>
                      )}
                    </div>
                  </div>
                ))}
              </div>

              <form onSubmit={handleCreateCustomSample} className="mt-2.5 space-y-1.5 border-t border-slate-800 pt-2.5">
                <div className="text-[10px] font-medium uppercase tracking-wider text-slate-400">
                  + Add Custom Fact, FIM Code, or Multimodal Rule into Weights
                </div>
                <input
                  type="text"
                  value={customPrompt}
                  onChange={(e) => setCustomPrompt(e.target.value)}
                  placeholder="Training Prompt (e.g. What is our moonbase password?)"
                  className="w-full rounded border border-slate-700/80 bg-slate-900 px-2.5 py-1.5 text-xs text-slate-100 placeholder:text-slate-500 focus:border-emerald-500 focus:outline-none"
                />
                <div className="flex gap-1.5">
                  <input
                    type="text"
                    value={customTarget}
                    onChange={(e) => setCustomTarget(e.target.value)}
                    placeholder="Target Output (e.g. The moonbase password is Apollo-Dolma-99!)"
                    className="flex-1 rounded border border-slate-700/80 bg-slate-900 px-2.5 py-1.5 text-xs text-slate-100 placeholder:text-slate-500 focus:border-emerald-500 focus:outline-none"
                  />
                  <button
                    type="submit"
                    className="flex items-center gap-1 rounded bg-emerald-600 px-2.5 py-1.5 text-xs font-semibold text-white hover:bg-emerald-500"
                  >
                    <Plus className="h-3.5 w-3.5" />
                    Add
                  </button>
                </div>
              </form>
            </div>

            <div className="space-y-2.5 rounded-lg border border-slate-800 bg-slate-950/70 p-2.5">
              <div className="text-xs font-semibold text-slate-200">
                2. Target Weight Modules & Hyperparameters
              </div>

              <div className="grid grid-cols-2 gap-2">
                <div>
                  <label className="mb-1 block text-[10px] text-slate-400">Training Method</label>
                  <div className="flex rounded bg-slate-900 p-0.5">
                    <button
                      onClick={() => onChangeFtConfig({ ...ftConfig, method: 'lora' })}
                      className={`flex-1 rounded py-1 text-[11px] font-medium ${
                        ftConfig.method === 'lora' ? 'bg-emerald-500/25 text-emerald-300' : 'text-slate-400'
                      }`}
                    >
                      LoRA (B·A)
                    </button>
                    <button
                      onClick={() => onChangeFtConfig({ ...ftConfig, method: 'full' })}
                      className={`flex-1 rounded py-1 text-[11px] font-medium ${
                        ftConfig.method === 'full' ? 'bg-pink-500/25 text-pink-300' : 'text-slate-400'
                      }`}
                    >
                      Full Weights
                    </button>
                  </div>
                </div>

                <div>
                  <label className="mb-1 block text-[10px] text-slate-400">LoRA Rank (r)</label>
                  <div className="flex gap-1">
                    {([4, 8, 16, 64] as const).map((r) => (
                      <button
                        key={r}
                        disabled={ftConfig.method === 'full'}
                        onClick={() => onChangeFtConfig({ ...ftConfig, loraRank: r })}
                        className={`flex-1 rounded py-1 font-mono text-[10px] ${
                          ftConfig.method === 'full'
                            ? 'bg-slate-900/50 text-slate-600'
                            : ftConfig.loraRank === r
                            ? 'bg-cyan-500/25 text-cyan-300 ring-1 ring-cyan-500/40'
                            : 'bg-slate-900 text-slate-400'
                        }`}
                      >
                        {r}
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-2">
                <div>
                  <div className="flex justify-between text-[10px] text-slate-400">
                    <span>Start Layer</span>
                    <span className="font-mono text-cyan-300">L{ftConfig.layerStart}</span>
                  </div>
                  <input
                    type="range"
                    min={0}
                    max={Math.min(ftConfig.layerEnd, architecture.numLayers - 1)}
                    value={Math.min(ftConfig.layerStart, architecture.numLayers - 1)}
                    onChange={(e) =>
                      onChangeFtConfig({ ...ftConfig, layerStart: parseInt(e.target.value, 10) })
                    }
                    className="sci-range w-full"
                  />
                </div>
                <div>
                  <div className="flex justify-between text-[10px] text-slate-400">
                    <span>End Layer</span>
                    <span className="font-mono text-cyan-300">
                      L{Math.min(ftConfig.layerEnd, architecture.numLayers - 1)}
                    </span>
                  </div>
                  <input
                    type="range"
                    min={ftConfig.layerStart}
                    max={architecture.numLayers - 1}
                    value={Math.min(ftConfig.layerEnd, architecture.numLayers - 1)}
                    onChange={(e) =>
                      onChangeFtConfig({ ...ftConfig, layerEnd: parseInt(e.target.value, 10) })
                    }
                    className="sci-range w-full"
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-2">
                <div>
                  <div className="flex justify-between text-[10px] text-slate-400">
                    <span>Learning Rate (η)</span>
                    <span
                      className={`font-mono ${
                        ftConfig.learningRate > 4.5e-4 ? 'text-amber-400' : 'text-emerald-300'
                      }`}
                    >
                      {ftConfig.learningRate.toExponential(1)}
                    </span>
                  </div>
                  <input
                    type="range"
                    min={1e-5}
                    max={7e-4}
                    step={2e-5}
                    value={ftConfig.learningRate}
                    onChange={(e) =>
                      onChangeFtConfig({ ...ftConfig, learningRate: parseFloat(e.target.value) })
                    }
                    className="sci-range w-full"
                  />
                </div>
                <div>
                  <div className="flex justify-between text-[10px] text-slate-400">
                    <span>Gradient Steps</span>
                    <span className="font-mono text-cyan-300">{ftConfig.steps}</span>
                  </div>
                  <input
                    type="range"
                    min={5}
                    max={35}
                    step={1}
                    value={ftConfig.steps}
                    onChange={(e) =>
                      onChangeFtConfig({ ...ftConfig, steps: parseInt(e.target.value, 10) })
                    }
                    className="sci-range w-full"
                  />
                </div>
              </div>

              <div>
                <div className="mb-1 text-[10px] text-slate-400">Target Weight Matrices</div>
                <div className="flex flex-wrap gap-1">
                  {targetKindChoices.map((kind) => {
                    const active = ftConfig.targetKinds.includes(kind);
                    return (
                      <button
                        key={kind}
                        onClick={() => {
                          const next = active
                            ? ftConfig.targetKinds.filter((k) => k !== kind)
                            : [...ftConfig.targetKinds, kind];
                          if (next.length > 0) onChangeFtConfig({ ...ftConfig, targetKinds: next });
                        }}
                        className={`rounded px-2 py-0.5 font-mono text-[10px] transition ${
                          active
                            ? 'bg-pink-500/25 text-pink-200 ring-1 ring-pink-500/50'
                            : 'bg-slate-900 text-slate-500'
                        }`}
                      >
                        {kind}
                      </button>
                    );
                  })}
                </div>
              </div>

              <button
                onClick={onRunFineTune}
                className="flex w-full items-center justify-center gap-2 rounded-lg bg-gradient-to-r from-emerald-600 to-cyan-600 py-2 text-xs font-semibold text-white shadow-lg shadow-emerald-950/50 transition hover:from-emerald-500 hover:to-cyan-500"
              >
                <Play className="h-3.5 w-3.5 fill-current" />
                Run Backprop & Write ΔW Into 3D Weights
              </button>
            </div>

            {ftHistory.length > 0 && (
              <div className="rounded-lg border border-slate-800 bg-slate-950/80 p-2.5">
                <div className="mb-1.5 flex items-center justify-between font-mono text-[10px]">
                  <span className="text-slate-300">Cross-Entropy Loss Curve</span>
                  <span className="text-emerald-400">
                    Start: {ftHistory[0].loss.toFixed(3)} → End:{' '}
                    {ftHistory[ftHistory.length - 1].loss.toFixed(3)}
                  </span>
                </div>
                <svg viewBox="0 0 300 72" className="h-20 w-full overflow-visible rounded bg-slate-900/70 p-1">
                  {(() => {
                    const maxLoss = Math.max(...ftHistory.map((h) => h.loss), 3.0);
                    const pts = ftHistory
                      .map((h, idx) => {
                        const x = (idx / Math.max(1, ftHistory.length - 1)) * 290 + 5;
                        const y = 64 - (h.loss / maxLoss) * 56;
                        return `${x},${y}`;
                      })
                      .join(' ');
                    return (
                      <>
                        <polyline
                          fill="none"
                          stroke="#10b981"
                          strokeWidth="2.2"
                          points={pts}
                        />
                        {ftHistory.map((h, idx) => {
                          const x = (idx / Math.max(1, ftHistory.length - 1)) * 290 + 5;
                          const y = 64 - (h.loss / maxLoss) * 56;
                          return <circle key={h.step} cx={x} cy={y} r="2" fill="#34d399" />;
                        })}
                      </>
                    );
                  })()}
                </svg>
              </div>
            )}
          </div>
        )}

        {activeTab === 'atlas' && (
          <div className="space-y-1.5">
            <div className="mb-2 flex items-center justify-between text-[11px] text-slate-400">
              <span>All {matrices.length} Addressable Tensor Matrices</span>
              <span className="font-mono text-cyan-300">
                {(architecture.exactParams / 1e9).toFixed(2)}B Total
              </span>
            </div>
            <div className="space-y-1">
              {matrices.map((m) => {
                const isSel = m.id === selectedMatrix.id;
                return (
                  <button
                    key={m.id}
                    onClick={() => {
                      onSelectMatrix(m.id);
                      setActiveTab('direct');
                    }}
                    className={`flex w-full items-center justify-between rounded-md border px-2.5 py-1.5 text-left font-mono text-[10px] transition ${
                      isSel
                        ? 'border-cyan-500/60 bg-cyan-950/30 text-cyan-200'
                        : 'border-slate-800/80 bg-slate-950/60 text-slate-300 hover:border-slate-700'
                    }`}
                  >
                    <span className="truncate pr-2">{m.name}</span>
                    <div className="flex shrink-0 items-center gap-2">
                      <span className="text-slate-500">{(m.paramCount / 1e6).toFixed(1)}M</span>
                      {m.ablated ? (
                        <span className="rounded bg-rose-500/20 px-1 text-rose-300">ZERO</span>
                      ) : m.deltaL2 > 0.03 ? (
                        <span className="rounded bg-emerald-500/20 px-1 text-emerald-300">
                          Δ{m.deltaL2.toFixed(2)}
                        </span>
                      ) : (
                        <span className="text-slate-500">FP{m.quantBits}</span>
                      )}
                    </div>
                  </button>
                );
              })}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
