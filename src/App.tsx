import { useCallback, useMemo, useState } from 'react';
import {
  Activity,
  BookOpen,
  Box,
  Code2,
  Cpu,
  Eye,
  HardDriveUpload,
  Network,
  RotateCcw,
  Sparkles,
} from 'lucide-react';
import {
  CustomInjectedKnowledge,
  FineTuneJobConfig,
  FineTuneSample,
  FineTuneStepMetric,
  InferenceMessage,
  MatrixKind,
  ModelArchitectureProfile,
  PulseSignal,
  TensorMatrixState,
  ThreeLayoutMode,
  WeightPresetId,
} from './types/olmo';
import { BUILTIN_MODELS } from './engine/modelRegistry';
import {
  applyMatrixTransform,
  buildPresetWeights,
  computeBehavioralProfile,
  DEFAULT_FINETUNE_SAMPLES,
  MatrixTransformOp,
  paintMatrixWeights,
  runFineTuneBackprop,
} from './engine/olmoWeightEngine';
import {
  InferenceRequestOptions,
  runOlmoInference,
} from './engine/olmoInferenceEngine';
import { ParsedModelUploadResult } from './engine/modelFileParser';
import { ThreeWeightEngine } from './components/ThreeWeightEngine';
import { WeightEditorPanel } from './components/WeightEditorPanel';
import { InferenceChatPanel } from './components/InferenceChatPanel';
import { HowItWorksSection } from './components/HowItWorksSection';
import { ModelUploadModal } from './components/ModelUploadModal';

const PRESET_OPTIONS: { id: WeightPresetId; label: string; badge: string }[] = [
  { id: 'base', label: 'Base Weights', badge: 'Pretrained' },
  { id: 'reasoning', label: 'Reasoning CoT', badge: 'LoRA' },
  { id: 'coder', label: 'Systems Coder', badge: 'FIM + AST' },
  { id: 'instruct', label: 'Helpful Instruct', badge: 'SFT' },
  { id: 'quant4', label: 'INT4 Quantized', badge: '4-bit Edge' },
];

export function App() {
  const [activeView, setActiveView] = useState<'studio' | 'how-it-works'>('studio');
  const [currentArch, setCurrentArch] = useState<ModelArchitectureProfile>(
    BUILTIN_MODELS[0]
  );
  const [isUploadModalOpen, setIsUploadModalOpen] = useState(false);

  const [presetId, setPresetId] = useState<WeightPresetId>('base');
  const [matrices, setMatrices] = useState<TensorMatrixState[]>(
    () => buildPresetWeights('base', BUILTIN_MODELS[0]).matrices
  );
  const [adapters, setAdapters] = useState<CustomInjectedKnowledge[]>([]);

  // Selection & 3D Engine Controls
  const [selectedMatrixId, setSelectedMatrixId] = useState<string>(
    'model.layers.14.mlp.gate_proj.weight'
  );
  const [layoutMode, setLayoutMode] = useState<ThreeLayoutMode>('helix');
  const [highlightKinds, setHighlightKinds] = useState<MatrixKind[]>([]);
  const [pulseSignal, setPulseSignal] = useState<PulseSignal | null>(null);

  // Fine-Tuning Lab State
  const [samples, setSamples] = useState<FineTuneSample[]>(DEFAULT_FINETUNE_SAMPLES);
  const [ftConfig, setFtConfig] = useState<FineTuneJobConfig>({
    method: 'lora',
    loraRank: 16,
    learningRate: 1.5e-4,
    steps: 18,
    layerStart: 8,
    layerEnd: 24,
    targetKinds: ['q_proj', 'v_proj', 'gate_proj', 'down_proj', 'lm_head'],
  });
  const [ftHistory, setFtHistory] = useState<FineTuneStepMetric[]>([]);

  // Inference Chat State
  const [messages, setMessages] = useState<InferenceMessage[]>([]);
  const [temperature, setTemperature] = useState<number>(0.7);
  const [topP, setTopP] = useState<number>(0.9);

  // Derived real-time Behavioral Profile from all live weight matrices
  const profile = useMemo(
    () => computeBehavioralProfile(matrices, adapters, currentArch),
    [matrices, adapters, currentArch]
  );

  const firePulse = useCallback(
    (direction: 'forward' | 'backward', layerStart = 0, layerEnd = 31) => {
      setPulseSignal({
        id: Date.now(),
        direction,
        startedAt: performance.now(),
        durationMs: 1350,
        color: direction === 'forward' ? '#22d3ee' : '#10b981',
        layerStart,
        layerEnd,
      });
    },
    []
  );

  const handleSelectBuiltinModel = useCallback(
    (arch: ModelArchitectureProfile) => {
      const built = buildPresetWeights('base', arch);
      setCurrentArch(arch);
      setPresetId('base');
      setMatrices(built.matrices);
      setAdapters(built.adapters);
      setFtHistory([]);
      const defaultSelected =
        arch.family === 'multimodal-vlm'
          ? built.matrices.find((m) => m.category === 'connector')?.id
          : arch.family === 'coding-llm'
          ? built.matrices.find((m) => m.layerIndex === 8 && m.kind === 'k_proj')?.id
          : built.matrices.find((m) => m.layerIndex === 12 && m.kind === 'gate_proj')?.id;
      setSelectedMatrixId(defaultSelected || built.matrices[0].id);
      setFtConfig((prev) => ({
        ...prev,
        layerStart: Math.min(prev.layerStart, arch.numLayers - 1),
        layerEnd: Math.min(prev.layerEnd, arch.numLayers - 1),
      }));
      firePulse('forward', 0, arch.numLayers - 1);
    },
    [firePulse]
  );

  const handleApplyParsedUpload = useCallback(
    (res: ParsedModelUploadResult) => {
      setCurrentArch(res.architecture);
      setPresetId('base');
      setMatrices(res.matrices);
      setAdapters([]);
      setFtHistory([]);
      setSelectedMatrixId(res.matrices[0]?.id || 'model.embed_tokens.weight');
      setFtConfig((prev) => ({
        ...prev,
        layerStart: 0,
        layerEnd: Math.max(0, res.architecture.numLayers - 1),
      }));
      firePulse('forward', 0, res.architecture.numLayers - 1);
    },
    [firePulse]
  );

  const handleLoadPreset = (id: WeightPresetId) => {
    const built = buildPresetWeights(id, currentArch);
    setPresetId(id);
    setMatrices(built.matrices);
    setAdapters(built.adapters);
    setFtHistory([]);
    firePulse('backward', 0, currentArch.numLayers - 1);
  };

  const handlePaintCell = useCallback(
    (
      matrixId: string,
      row: number,
      col: number,
      radius: number,
      delta: number,
      mode: 'add' | 'zero'
    ) => {
      setPresetId('custom');
      setMatrices((prev) => paintMatrixWeights(prev, matrixId, row, col, radius, delta, mode));
    },
    []
  );

  const handleTransformMatrix = useCallback(
    (matrixId: string, op: MatrixTransformOp) => {
      setPresetId('custom');
      setMatrices((prev) => applyMatrixTransform(prev, matrixId, op));
      const target = matrices.find((m) => m.id === matrixId);
      const l = target ? Math.max(0, Math.min(currentArch.numLayers - 1, target.layerIndex)) : 12;
      firePulse('backward', Math.max(0, l - 2), Math.min(currentArch.numLayers - 1, l + 2));
    },
    [firePulse, matrices, currentArch.numLayers]
  );

  const handleBatchTransform = useCallback(
    (filter: (m: TensorMatrixState) => boolean, op: MatrixTransformOp) => {
      setPresetId('custom');
      setMatrices((prev) =>
        prev.map((m) => (filter(m) ? applyMatrixTransform([m], m.id, op)[0] : m))
      );
      firePulse('backward', 0, currentArch.numLayers - 1);
    },
    [firePulse, currentArch.numLayers]
  );

  const handleToggleSample = (id: string) => {
    setSamples((prev) => prev.map((s) => (s.id === id ? { ...s, enabled: !s.enabled } : s)));
  };

  const handleAddCustomSample = (sample: Omit<FineTuneSample, 'id'>) => {
    setSamples((prev) => [
      ...prev,
      { ...sample, id: `custom-${Date.now()}` },
    ]);
  };

  const handleDeleteSample = (id: string) => {
    setSamples((prev) => prev.filter((s) => s.id !== id));
  };

  const handleRunFineTune = () => {
    const result = runFineTuneBackprop(matrices, samples, ftConfig);
    setMatrices(result.updatedMatrices);
    setFtHistory(result.history);
    setPresetId('custom');

    setAdapters((prev) => {
      const next = [...prev];
      for (const na of result.newAdapters) {
        const idx = next.findIndex((existing) => existing.prompt === na.prompt);
        if (idx >= 0) next[idx] = na;
        else next.push(na);
      }
      return next;
    });

    firePulse('backward', ftConfig.layerStart, ftConfig.layerEnd);
  };

  const handleSendPrompt = (prompt: string, options?: InferenceRequestOptions) => {
    const userMsg: InferenceMessage = {
      id: `u-${Date.now()}`,
      role: 'user',
      text: prompt,
      timestamp: Date.now(),
      presetSnapshot: presetId,
    };

    const trace = runOlmoInference(
      prompt,
      matrices,
      profile,
      adapters,
      temperature,
      topP,
      currentArch,
      options
    );

    const snapshotLabel =
      presetId === 'custom'
        ? `Custom (Δ${profile.globalDeltaL2})`
        : PRESET_OPTIONS.find((p) => p.id === presetId)?.label ?? currentArch.name;

    const botMsg: InferenceMessage = {
      id: `a-${Date.now() + 1}`,
      role: 'assistant',
      text: trace.outputText,
      timestamp: Date.now() + 1,
      presetSnapshot: snapshotLabel,
      trace,
    };

    setMessages((prev) => [...prev, userMsg, botMsg]);
    firePulse('forward', 0, currentArch.numLayers - 1);
  };

  const handleHighlightFromGuide = (kinds: MatrixKind[], targetKind: MatrixKind) => {
    setHighlightKinds(kinds);
    const preferred =
      matrices.find((m) => m.layerIndex === 10 && m.kind === targetKind) ??
      matrices.find((m) => m.kind === targetKind);
    if (preferred) setSelectedMatrixId(preferred.id);
    setActiveView('studio');
    setLayoutMode('helix');
    firePulse('forward', 0, currentArch.numLayers - 1);
    setTimeout(() => setHighlightKinds([]), 4500);
  };

  return (
    <div className="min-h-screen bg-[#080c15] text-slate-100">
      {/* Top Scientific Command Bar */}
      <header className="sticky top-0 z-40 border-b border-slate-800/90 bg-[#080c15]/95 backdrop-blur-md">
        <div className="mx-auto flex max-w-[1720px] flex-wrap items-center justify-between gap-2.5 px-4 py-2.5">
          {/* Brand + View Switcher + Model Family Switcher */}
          <div className="flex flex-wrap items-center gap-3">
            <div className="flex items-center gap-2.5">
              <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-pink-500 to-cyan-500 shadow-md shadow-pink-500/20">
                <Cpu className="h-4 w-4 text-white" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <span className="text-sm font-bold tracking-tight text-white">
                    OpenWeights 3D Lab
                  </span>
                  <span className="rounded border border-pink-500/35 bg-pink-500/15 px-1.5 py-0.5 font-mono text-[10px] font-medium text-pink-300">
                    {currentArch.badge}
                  </span>
                </div>
                <div className="font-mono text-[10px] text-slate-400">
                  {currentArch.name} • {matrices.length} Tensors
                </div>
              </div>
            </div>

            {/* View Switcher */}
            <nav className="flex items-center gap-1 rounded-lg border border-slate-800 bg-slate-950 p-1">
              <button
                onClick={() => setActiveView('studio')}
                className={`flex items-center gap-1.5 rounded-md px-2.5 py-1 text-xs font-medium transition ${
                  activeView === 'studio'
                    ? 'bg-gradient-to-r from-pink-500/25 to-cyan-500/20 text-white ring-1 ring-pink-500/40'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <Box className="h-3.5 w-3.5 text-cyan-400" />
                3D Weights & Fine-Tune
              </button>
              <button
                onClick={() => setActiveView('how-it-works')}
                className={`flex items-center gap-1.5 rounded-md px-2.5 py-1 text-xs font-medium transition ${
                  activeView === 'how-it-works'
                    ? 'bg-gradient-to-r from-pink-500/25 to-cyan-500/20 text-white ring-1 ring-cyan-500/40'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <BookOpen className="h-3.5 w-3.5 text-pink-400" />
                How It Works & Algorithms
              </button>
            </nav>

            {/* Quick Model Family Selector + Upload Model File Button */}
            <div className="flex flex-wrap items-center gap-1 rounded-lg border border-slate-800 bg-slate-950/90 p-1">
              {BUILTIN_MODELS.map((m) => {
                const isSel = currentArch.id === m.id;
                const Icon =
                  m.family === 'coding-llm'
                    ? Code2
                    : m.family === 'multimodal-vlm'
                    ? Eye
                    : m.family === 'moe-llm'
                    ? Network
                    : Cpu;
                const shortName =
                  m.family === 'coding-llm'
                    ? 'Coder-7B'
                    : m.family === 'multimodal-vlm'
                    ? 'Molmo-VLM'
                    : m.family === 'moe-llm'
                    ? 'OLMoE'
                    : 'OLMo-7B';
                return (
                  <button
                    key={m.id}
                    onClick={() => handleSelectBuiltinModel(m)}
                    className={`flex items-center gap-1 rounded px-2 py-1 text-[11px] font-medium transition ${
                      isSel
                        ? 'bg-pink-500/20 text-pink-200 ring-1 ring-pink-500/40'
                        : 'text-slate-400 hover:bg-slate-900 hover:text-slate-200'
                    }`}
                    title={`Switch to ${m.name} (${m.badge})`}
                  >
                    <Icon className="h-3 w-3" />
                    {shortName}
                  </button>
                );
              })}

              <button
                onClick={() => setIsUploadModalOpen(true)}
                className="flex items-center gap-1 rounded bg-gradient-to-r from-cyan-600 to-emerald-600 px-2.5 py-1 text-[11px] font-semibold text-white shadow-sm hover:from-cyan-500 hover:to-emerald-500"
                title="Upload .safetensors, .gguf, config.json, or .onnx model file"
              >
                <HardDriveUpload className="h-3 w-3" />
                Upload Model File
              </button>
            </div>
          </div>

          {/* Model Weight Presets */}
          <div className="flex flex-wrap items-center gap-2">
            <div className="flex flex-wrap items-center gap-1 rounded-lg border border-slate-800 bg-slate-950/90 p-1">
              {PRESET_OPTIONS.map((p) => (
                <button
                  key={p.id}
                  onClick={() => handleLoadPreset(p.id)}
                  className={`rounded px-2 py-1 text-[11px] font-medium transition ${
                    presetId === p.id
                      ? 'bg-cyan-500/20 text-cyan-200 ring-1 ring-cyan-500/40'
                      : 'text-slate-400 hover:bg-slate-900 hover:text-slate-200'
                  }`}
                >
                  {p.label}
                </button>
              ))}
              {presetId === 'custom' && (
                <span className="rounded bg-emerald-500/20 px-2 py-1 font-mono text-[10px] font-semibold text-emerald-300 ring-1 ring-emerald-500/40">
                  User Edited
                </span>
              )}
            </div>

            {profile.modifiedMatricesCount > 0 && (
              <button
                onClick={() => handleLoadPreset('base')}
                className="flex items-center gap-1 rounded-lg border border-slate-700 bg-slate-900 px-2.5 py-1.5 text-[11px] text-slate-300 hover:border-pink-500/50 hover:text-white"
                title="Reset all matrices back to Base weights"
              >
                <RotateCcw className="h-3 w-3" />
                Reset Base
              </button>
            )}
          </div>
        </div>

        {/* Sub-Header Live Telemetry Strip */}
        <div className="border-t border-slate-800/70 bg-slate-950/70">
          <div className="mx-auto flex max-w-[1720px] flex-wrap items-center justify-between gap-x-5 gap-y-1 px-4 py-1.5 font-mono text-[10px]">
            <div className="flex flex-wrap items-center gap-4 text-slate-400">
              <span>
                MODEL:{' '}
                <strong className="text-cyan-300">{currentArch.name}</strong>
              </span>
              <span>
                PARAMS:{' '}
                <strong className="text-slate-100">
                  {profile.totalParams.toLocaleString()}
                </strong>
              </span>
              <span>
                MODIFIED MATRICES:{' '}
                <strong
                  className={
                    profile.modifiedMatricesCount > 0 ? 'text-emerald-400' : 'text-slate-300'
                  }
                >
                  {profile.modifiedMatricesCount} / {matrices.length}
                </strong>
              </span>
              <span>
                GLOBAL ΔW L2:{' '}
                <strong className="text-cyan-300">{profile.globalDeltaL2.toFixed(2)}</strong>
              </span>
              {currentArch.sourceFileInfo && (
                <span className="rounded bg-emerald-500/20 px-1.5 py-0.5 text-emerald-300">
                  FILE: {currentArch.sourceFileInfo.fileName} ({currentArch.sourceFileInfo.quantizationType})
                </span>
              )}
            </div>

            <div className="flex flex-wrap items-center gap-4">
              <span className="text-slate-400">
                PPL: <strong className="text-slate-100">{profile.estimatedPerplexity}</strong>
              </span>
              <span className="text-slate-400">
                D_KL: <strong className="text-emerald-300">{profile.klDivergence}</strong>
              </span>
              <span className="flex items-center gap-1 text-slate-400">
                <Activity className="h-3 w-3 text-cyan-400" />
                STABILITY:{' '}
                <strong
                  className={
                    profile.stabilityScore > 0.72
                      ? 'text-emerald-400'
                      : profile.stabilityScore > 0.45
                      ? 'text-amber-400'
                      : 'text-rose-400'
                  }
                >
                  {(profile.stabilityScore * 100).toFixed(0)}%
                </strong>
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="mx-auto max-w-[1720px] p-3 sm:p-4">
        {activeView === 'studio' ? (
          <div className="space-y-3">
            {/* 3-Pane Synchronized Workbench */}
            <div className="grid grid-cols-1 gap-3 xl:h-[calc(100vh-170px)] xl:min-h-[610px] xl:grid-cols-12">
              {/* Pane 1: Three.js 3D Weight Tensor Engine + Compact Trait Readout */}
              <div className="flex h-[520px] flex-col gap-2.5 xl:col-span-4 xl:h-full">
                <div className="min-h-0 flex-1">
                  <ThreeWeightEngine
                    architecture={currentArch}
                    matrices={matrices}
                    selectedMatrixId={selectedMatrixId}
                    onSelectMatrix={setSelectedMatrixId}
                    layoutMode={layoutMode}
                    onChangeLayoutMode={setLayoutMode}
                    pulseSignal={pulseSignal}
                    highlightKinds={highlightKinds}
                  />
                </div>

                {/* Live Weight-Derived Behavioral Profile Bar */}
                <div className="rounded-xl border border-slate-800/90 bg-[#0d1320] px-3 py-2">
                  <div className="mb-1.5 flex items-center justify-between text-[10px]">
                    <span className="flex items-center gap-1.5 font-mono font-semibold uppercase tracking-wider text-slate-300">
                      <Sparkles className="h-3 w-3 text-pink-400" />
                      Live Behavioral Vector ({currentArch.numLayers} Layers)
                    </span>
                    <button
                      onClick={() => setActiveView('how-it-works')}
                      className="font-mono text-cyan-400 hover:underline"
                    >
                      Algorithms & Math →
                    </button>
                  </div>
                  <div className="grid grid-cols-3 gap-2 sm:grid-cols-6">
                    {[
                      { label: 'Reasoning', val: profile.reasoningDepth, cls: 'bg-cyan-400' },
                      { label: 'FIM/Code', val: profile.fimInfillAccuracy, cls: 'bg-indigo-400' },
                      { label: 'Vision 2D', val: profile.visionGrounding, cls: 'bg-amber-400' },
                      { label: 'Instruct', val: profile.instructionFollowing, cls: 'bg-emerald-400' },
                      { label: 'Factuality', val: profile.factuality, cls: 'bg-pink-400' },
                      { label: 'Stability', val: profile.stabilityScore, cls: 'bg-teal-400' },
                    ].map((trait) => (
                      <div key={trait.label}>
                        <div className="flex justify-between font-mono text-[9px] text-slate-400">
                          <span>{trait.label}</span>
                          <span className="text-slate-200">{(trait.val * 100).toFixed(0)}%</span>
                        </div>
                        <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-slate-800">
                          <div
                            className={`h-full rounded-full transition-all duration-300 ${trait.cls}`}
                            style={{ width: `${Math.round(trait.val * 100)}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Pane 2: Direct Weight Surgery & SFT Backprop Lab */}
              <div className="h-[640px] xl:col-span-4 xl:h-full">
                <WeightEditorPanel
                  architecture={currentArch}
                  matrices={matrices}
                  selectedMatrixId={selectedMatrixId}
                  onSelectMatrix={setSelectedMatrixId}
                  onPaintCell={handlePaintCell}
                  onTransformMatrix={handleTransformMatrix}
                  onBatchTransform={handleBatchTransform}
                  samples={samples}
                  onToggleSample={handleToggleSample}
                  onAddCustomSample={handleAddCustomSample}
                  onDeleteSample={handleDeleteSample}
                  ftConfig={ftConfig}
                  onChangeFtConfig={setFtConfig}
                  ftHistory={ftHistory}
                  onRunFineTune={handleRunFineTune}
                  onInspectIn3D={() => setLayoutMode('matrix3d')}
                />
              </div>

              {/* Pane 3: Live Weight-Impact Inference Chat (with Code FIM & Vision 2D Pointing) */}
              <div className="h-[620px] xl:col-span-4 xl:h-full">
                <InferenceChatPanel
                  architecture={currentArch}
                  messages={messages}
                  profile={profile}
                  temperature={temperature}
                  topP={topP}
                  onChangeTemperature={setTemperature}
                  onChangeTopP={setTopP}
                  onSendPrompt={handleSendPrompt}
                  onClearChat={() => setMessages([])}
                />
              </div>
            </div>
          </div>
        ) : (
          <HowItWorksSection
            architecture={currentArch}
            profile={profile}
            onHighlightIn3D={handleHighlightFromGuide}
            onSelectModelArchitecture={(arch) => {
              handleSelectBuiltinModel(arch);
              setActiveView('studio');
            }}
          />
        )}
      </main>

      {/* Model File Upload & Architecture Switcher Modal */}
      <ModelUploadModal
        isOpen={isUploadModalOpen}
        onClose={() => setIsUploadModalOpen(false)}
        currentArch={currentArch}
        onSelectBuiltinModel={handleSelectBuiltinModel}
        onApplyParsedUpload={handleApplyParsedUpload}
      />
    </div>
  );
}

export default App;
