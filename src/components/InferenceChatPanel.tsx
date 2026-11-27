import React, { useEffect, useRef, useState } from 'react';
import {
  AlertTriangle,
  ArrowRight,
  BarChart3,
  Bot,
  Code2,
  Crosshair,
  Eye,
  GitCompare,
  MessageSquare,
  Send,
  Sparkles,
  Trash2,
  User,
} from 'lucide-react';
import {
  GeneratedTokenTrace,
  InferenceMessage,
  ModelArchitectureProfile,
  ModelBehavioralProfile,
} from '../types/olmo';
import {
  InferenceRequestOptions,
  MULTIMODAL_SCENES,
} from '../engine/olmoInferenceEngine';

interface InferenceChatPanelProps {
  architecture: ModelArchitectureProfile;
  messages: InferenceMessage[];
  profile: ModelBehavioralProfile;
  temperature: number;
  topP: number;
  onChangeTemperature: (t: number) => void;
  onChangeTopP: (p: number) => void;
  onSendPrompt: (prompt: string, options?: InferenceRequestOptions) => void;
  onClearChat: () => void;
}

const QUICK_TEST_PROMPTS = [
  {
    label: 'SwiGLU 11,008 Math',
    prompt: 'Why does OLMo-7B use 11,008 dimensions in its SwiGLU MLP?',
  },
  {
    label: 'Python Quicksort',
    prompt: 'Write a clean Python quicksort function with type hints.',
  },
  {
    label: 'RoPE & GQA Attention',
    prompt: 'How do RoPE and Grouped-Query Attention (GQA) work?',
  },
  {
    label: 'Test JSON / Fact SFT',
    prompt: 'What is the internal release codename for our fine-tuned OLMo cluster?',
  },
];

export const InferenceChatPanel: React.FC<InferenceChatPanelProps> = ({
  architecture,
  messages,
  profile,
  temperature,
  topP,
  onChangeTemperature,
  onChangeTopP,
  onSendPrompt,
  onClearChat,
}) => {
  const [input, setInput] = useState('');
  const [inferMode, setInferMode] = useState<'chat' | 'fim' | 'vision'>('chat');
  const [compareBase, setCompareBase] = useState(true);
  const [selectedTokenTrace, setSelectedTokenTrace] = useState<GeneratedTokenTrace | null>(null);
  const [expandedTraceMsgId, setExpandedTraceMsgId] = useState<string | null>(null);

  // Coding AI FIM Prefix / Suffix state
  const [fimPrefix, setFimPrefix] = useState(
    'def binary_search(arr: list[int], target: int) -> int:\n    low, high = 0, len(arr) - 1\n    while low <= high:'
  );
  const [fimSuffix, setFimSuffix] = useState('    return -1');

  // Multimodal VLM Scene state
  const [selectedSceneId, setSelectedSceneId] = useState<string>(MULTIMODAL_SCENES[0].id);

  const scrollEndRef = useRef<HTMLDivElement | null>(null);

  // Automatically switch to the specialized inference mode when user loads Coding or Multimodal model
  useEffect(() => {
    if (architecture.family === 'coding-llm') {
      setInferMode('fim');
    } else if (architecture.family === 'multimodal-vlm') {
      setInferMode('vision');
    } else {
      setInferMode('chat');
    }
  }, [architecture.id, architecture.family]);

  useEffect(() => {
    scrollEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages.length]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (inferMode === 'fim') {
      onSendPrompt(
        `[FIM Code Completion] Synthesize <|fim_middle|> between Prefix and Suffix`,
        { fimInput: { prefix: fimPrefix, suffix: fimSuffix } }
      );
      return;
    }
    if (inferMode === 'vision') {
      const scene = MULTIMODAL_SCENES.find((s) => s.id === selectedSceneId) ?? MULTIMODAL_SCENES[0];
      const q = input.trim() || scene.defaultPrompt;
      onSendPrompt(q, { multimodalSceneId: scene.id });
      setInput('');
      return;
    }
    if (!input.trim()) return;
    onSendPrompt(input.trim());
    setInput('');
  };

  const healthLabel =
    profile.explosionRisk > 0.45
      ? { text: 'Explosion Risk', cls: 'bg-rose-500/20 text-rose-300 border-rose-500/40' }
      : profile.collapseRisk > 0.45
      ? { text: 'Collapsed', cls: 'bg-rose-500/20 text-rose-300 border-rose-500/40' }
      : profile.modifiedMatricesCount > 0
      ? { text: 'Fine-Tuned Active', cls: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40' }
      : { text: architecture.badge, cls: 'bg-cyan-500/15 text-cyan-300 border-cyan-500/40' };

  const activeScene =
    MULTIMODAL_SCENES.find((s) => s.id === selectedSceneId) ?? MULTIMODAL_SCENES[0];

  return (
    <div className="flex h-full flex-col overflow-hidden rounded-xl border border-slate-800/90 bg-[#0d1320]">
      {/* Chat Header & Mode Switcher */}
      <div className="border-b border-slate-800 bg-slate-950/70 px-3 py-2">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-1.5">
            <button
              onClick={() => setInferMode('chat')}
              className={`flex items-center gap-1 rounded-md px-2 py-1 text-[11px] font-medium transition ${
                inferMode === 'chat'
                  ? 'bg-pink-500/20 text-pink-300 ring-1 ring-pink-500/40'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <MessageSquare className="h-3 w-3" />
              Chat
            </button>
            <button
              onClick={() => setInferMode('fim')}
              className={`flex items-center gap-1 rounded-md px-2 py-1 text-[11px] font-medium transition ${
                inferMode === 'fim'
                  ? 'bg-indigo-500/25 text-indigo-200 ring-1 ring-indigo-500/50'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
              title="Coding AI Fill-in-the-Middle (FIM) & Speculative Decoding"
            >
              <Code2 className="h-3 w-3" />
              Code FIM Lab
            </button>
            <button
              onClick={() => setInferMode('vision')}
              className={`flex items-center gap-1 rounded-md px-2 py-1 text-[11px] font-medium transition ${
                inferMode === 'vision'
                  ? 'bg-amber-500/25 text-amber-200 ring-1 ring-amber-500/50'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
              title="Multimodal Vision Patchifier & Molmo 2D Pointing"
            >
              <Eye className="h-3 w-3" />
              Vision 2D Pointing
            </button>
          </div>

          <div className="flex items-center gap-1.5">
            <span className={`hidden rounded border px-1.5 py-0.5 font-mono text-[9px] sm:inline ${healthLabel.cls}`}>
              {healthLabel.text}
            </span>
            <button
              onClick={() => setCompareBase((v) => !v)}
              className={`flex items-center gap-1 rounded-md px-2 py-1 text-[10px] font-medium transition ${
                compareBase
                  ? 'bg-cyan-500/20 text-cyan-200 ring-1 ring-cyan-500/40'
                  : 'bg-slate-900 text-slate-400 hover:text-slate-200'
              }`}
              title="Toggle side-by-side comparison vs Base Checkpoint"
            >
              <GitCompare className="h-3 w-3" />
              Diff
            </button>
            <button
              onClick={onClearChat}
              className="rounded-md bg-slate-900 p-1 text-slate-400 hover:bg-slate-800 hover:text-rose-300"
              title="Clear chat history"
            >
              <Trash2 className="h-3.5 w-3.5" />
            </button>
          </div>
        </div>

        {/* Live Telemetry & Sampling Bar */}
        <div className="mt-2 flex flex-wrap items-center justify-between gap-2 border-t border-slate-800/70 pt-1.5 font-mono text-[10px]">
          <div className="flex items-center gap-2.5">
            <label className="flex items-center gap-1 text-slate-400">
              Temp:
              <input
                type="range"
                min={0.1}
                max={1.5}
                step={0.05}
                value={temperature}
                onChange={(e) => onChangeTemperature(parseFloat(e.target.value))}
                className="sci-range w-14"
              />
              <span className="text-cyan-300">{temperature.toFixed(2)}</span>
            </label>
            <label className="flex items-center gap-1 text-slate-400">
              Top-P:
              <input
                type="range"
                min={0.2}
                max={1.0}
                step={0.05}
                value={topP}
                onChange={(e) => onChangeTopP(parseFloat(e.target.value))}
                className="sci-range w-12"
              />
              <span className="text-cyan-300">{topP.toFixed(2)}</span>
            </label>
          </div>
          <div className="flex items-center gap-2.5 text-slate-400">
            <span>
              PPL: <strong className="text-slate-200">{profile.estimatedPerplexity}</strong>
            </span>
            <span>
              D_KL: <strong className="text-emerald-300">{profile.klDivergence}</strong>
            </span>
          </div>
        </div>
      </div>

      {/* Specialized Interactive Algorithm Workbench Header (when in FIM or Vision mode) */}
      {inferMode === 'fim' && (
        <div className="border-b border-slate-800 bg-indigo-950/15 p-2.5">
          <div className="mb-1.5 flex items-center justify-between font-mono text-[10px]">
            <span className="font-semibold text-indigo-300">
              Coding AI: Fill-in-the-Middle (&lt;|fim_prefix|&gt; ... &lt;|fim_suffix|&gt; ... &lt;|fim_middle|&gt;)
            </span>
            <span className="text-cyan-300">
              GQA {architecture.numHeads}Q:{architecture.numKvHeads}KV
            </span>
          </div>
          <div className="grid grid-cols-1 gap-1.5 sm:grid-cols-2">
            <div>
              <div className="mb-0.5 font-mono text-[9px] text-emerald-400">&lt;|fim_prefix|&gt;</div>
              <textarea
                rows={3}
                value={fimPrefix}
                onChange={(e) => setFimPrefix(e.target.value)}
                className="w-full rounded border border-slate-700/80 bg-slate-950 p-1.5 font-mono text-[10px] text-slate-200 focus:border-indigo-500 focus:outline-none"
              />
            </div>
            <div>
              <div className="mb-0.5 font-mono text-[9px] text-pink-400">&lt;|fim_suffix|&gt;</div>
              <textarea
                rows={3}
                value={fimSuffix}
                onChange={(e) => setFimSuffix(e.target.value)}
                className="w-full rounded border border-slate-700/80 bg-slate-950 p-1.5 font-mono text-[10px] text-slate-200 focus:border-indigo-500 focus:outline-none"
              />
            </div>
          </div>
          <button
            onClick={handleSubmit}
            className="mt-2 flex w-full items-center justify-center gap-1.5 rounded-lg bg-indigo-600 py-1.5 text-xs font-semibold text-white hover:bg-indigo-500"
          >
            <Code2 className="h-3.5 w-3.5" />
            Synthesize &lt;|fim_middle|&gt; Span & Run Speculative Decoding
          </button>
        </div>
      )}

      {inferMode === 'vision' && (
        <div className="border-b border-slate-800 bg-amber-950/10 p-2.5">
          <div className="mb-1.5 flex items-center justify-between font-mono text-[10px]">
            <span className="font-semibold text-amber-300">
              Multimodal AI: ViT-L/14 Patchifier (576→144 Pooled) + PixMo 2D Pointing
            </span>
            <span className="text-emerald-300">
              Grounding: {(profile.visionGrounding * 100).toFixed(0)}%
            </span>
          </div>

          <div className="flex items-start gap-2.5">
            {/* Interactive 8x8 Scene Patch Canvas with 2D Pointing Crosshairs */}
            <div className="relative h-20 w-20 shrink-0 overflow-hidden rounded-lg border border-amber-500/40 bg-slate-950 p-0.5">
              <div className="grid h-full w-full grid-cols-8 gap-[1px]">
                {activeScene.gridColors.map((col, idx) => (
                  <div
                    key={idx}
                    style={{ backgroundColor: col }}
                    className="rounded-[1px]"
                  />
                ))}
              </div>
              {/* Overlaid 2D Pointing Targets */}
              {activeScene.pointingTargets.map((pt) => (
                <div
                  key={pt.label}
                  style={{ left: `${pt.xPercent}%`, top: `${pt.yPercent}%` }}
                  title={`${pt.label} (${pt.xPercent}%, ${pt.yPercent}%)`}
                  className=" absolute -translate-x-1/2 -translate-y-1/2"
                >
                  <Crosshair className="h-3.5 w-3.5 text-white drop-shadow-[0_0_4px_rgba(244,63,94,1)]" />
                </div>
              ))}
            </div>

            <div className="min-w-0 flex-1 space-y-1.5">
              <div className="flex flex-wrap gap-1">
                {MULTIMODAL_SCENES.map((sc) => (
                  <button
                    key={sc.id}
                    onClick={() => setSelectedSceneId(sc.id)}
                    className={`rounded px-2 py-0.5 text-[10px] font-medium transition ${
                      selectedSceneId === sc.id
                        ? 'bg-amber-500 text-slate-950 font-semibold'
                        : 'bg-slate-900 text-slate-300 hover:bg-slate-800'
                    }`}
                  >
                    {sc.title.split(' ')[0]} {sc.title.split(' ')[1]}
                  </button>
                ))}
              </div>
              <div className="truncate font-mono text-[10px] text-slate-400">
                {activeScene.subtitle}
              </div>
              <button
                onClick={() =>
                  onSendPrompt(activeScene.defaultPrompt, {
                    multimodalSceneId: activeScene.id,
                  })
                }
                className="flex items-center gap-1 rounded bg-amber-500/20 px-2.5 py-1 text-[10px] font-semibold text-amber-200 ring-1 ring-amber-500/40 hover:bg-amber-500/30"
              >
                <Crosshair className="h-3 w-3" />
                Run ViT + Connector & Emit 2D &lt;point x y&gt;
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Messages Stream */}
      <div className="flex-1 space-y-3 overflow-y-auto p-3">
        {messages.length === 0 ? (
          <div className="flex h-full flex-col items-center justify-center rounded-lg border border-dashed border-slate-800 bg-slate-950/40 p-5 text-center">
            <Sparkles className="mb-2 h-7 w-7 text-pink-400/80" />
            <div className="text-sm font-medium text-slate-200">
              Test {architecture.name} Against Your Live 3D Weight State
            </div>
            <p className="mt-1 max-w-md text-xs text-slate-400">
              Edit weights in the middle panel (LoRA, SFT Backprop, Scaling, Ablation, Quantization)
              or switch between <strong className="text-pink-300">Chat</strong>,{' '}
              <strong className="text-indigo-300">Code FIM Lab</strong>, and{' '}
              <strong className="text-amber-300">Vision 2D Pointing</strong> above.
            </p>
            <div className="mt-4 flex flex-wrap justify-center gap-1.5">
              {QUICK_TEST_PROMPTS.map((qp) => (
                <button
                  key={qp.label}
                  onClick={() => onSendPrompt(qp.prompt)}
                  className="flex items-center gap-1 rounded-full border border-slate-700 bg-slate-900 px-3 py-1 text-[11px] text-cyan-300 transition hover:border-cyan-500/50 hover:bg-slate-800"
                >
                  {qp.label}
                  <ArrowRight className="h-3 w-3" />
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((msg) => {
            if (msg.role === 'user') {
              return (
                <div key={msg.id} className="flex items-start justify-end gap-2">
                  <div className="max-w-[85%] rounded-xl rounded-tr-sm border border-cyan-500/30 bg-cyan-950/35 px-3 py-2 text-xs text-slate-100">
                    {msg.text}
                  </div>
                  <div className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-cyan-500/20 text-cyan-300">
                    <User className="h-3.5 w-3.5" />
                  </div>
                </div>
              );
            }

            const trace = msg.trace;
            const showTraceDetails = expandedTraceMsgId === msg.id;

            return (
              <div key={msg.id} className="flex items-start gap-2">
                <div className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-pink-500/20 text-pink-300">
                  <Bot className="h-3.5 w-3.5" />
                </div>

                <div className="min-w-0 flex-1 space-y-2 rounded-xl rounded-tl-sm border border-slate-800 bg-slate-950/80 p-3">
                  <div className="flex flex-wrap items-center justify-between gap-2 text-[10px]">
                    <div className="flex flex-wrap items-center gap-1.5">
                      <span className="font-mono font-semibold text-pink-300">
                        {architecture.name.split(' ')[0]}
                      </span>
                      <span className="rounded bg-slate-900 px-1.5 py-0.5 font-mono text-slate-400">
                        {msg.presetSnapshot}
                      </span>
                      {trace?.activeAdapters.map((ad) => (
                        <span
                          key={ad}
                          className="rounded bg-emerald-500/15 px-1.5 py-0.5 font-mono text-emerald-300"
                        >
                          +{ad}
                        </span>
                      ))}
                    </div>
                    {trace && (
                      <div className="flex items-center gap-2 font-mono text-slate-400">
                        <span>PPL {trace.avgPerplexity}</span>
                        <button
                          onClick={() =>
                            setExpandedTraceMsgId(showTraceDetails ? null : msg.id)
                          }
                          className="flex items-center gap-1 rounded bg-slate-900 px-1.5 py-0.5 text-cyan-300 hover:bg-slate-800"
                        >
                          <BarChart3 className="h-3 w-3" />
                          {showTraceDetails ? 'Hide Trace' : 'Logits & Trace'}
                        </button>
                      </div>
                    )}
                  </div>

                  {trace?.stabilityWarning && (
                    <div className="flex items-center gap-2 rounded-md border border-rose-500/40 bg-rose-950/40 px-2.5 py-1.5 text-[11px] text-rose-200">
                      <AlertTriangle className="h-3.5 w-3.5 shrink-0 text-rose-400" />
                      <span>{trace.stabilityWarning}</span>
                    </div>
                  )}

                  {/* Multimodal Cross-Modal Heatmap & 2D Pointing Pill Summary */}
                  {trace?.visionTrace && (
                    <div className="flex items-center gap-3 rounded-lg border border-amber-500/30 bg-amber-950/15 p-2">
                      <div className="grid h-14 w-14 shrink-0 grid-cols-8 gap-[1px] rounded border border-slate-700 bg-slate-950 p-0.5">
                        {trace.visionTrace.heatmap8x8.map((attn, idx) => (
                          <div
                            key={idx}
                            style={{
                              backgroundColor: `rgba(245, 158, 11, ${Math.min(1, attn)})`,
                            }}
                          />
                        ))}
                      </div>
                      <div className="min-w-0 flex-1 font-mono text-[10px]">
                        <div className="text-amber-300">
                          Cross-Modal Attention Map ({trace.visionTrace.rawPatches} ViT Patches →{' '}
                          {trace.visionTrace.pooledTokens} Pooled)
                        </div>
                        <div className="mt-1 flex flex-wrap gap-1.5">
                          {trace.visionTrace.pointingCoords.map((pt) => (
                            <span
                              key={pt.label}
                              className="rounded bg-slate-900 px-1.5 py-0.5 text-cyan-300 ring-1 ring-cyan-500/30"
                            >
                              🎯 {pt.label}: ({pt.xPercent}%, {pt.yPercent}%)
                            </span>
                          ))}
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Output Text (Side-by-side or Single) */}
                  {compareBase && trace && trace.outputText !== trace.baseOutputText ? (
                    <div className="grid grid-cols-1 gap-2 lg:grid-cols-2">
                      <div className="rounded-lg border border-emerald-500/30 bg-emerald-950/15 p-2.5">
                        <div className="mb-1 font-mono text-[10px] font-semibold text-emerald-300">
                          LIVE WEIGHTS OUTPUT
                        </div>
                        <pre className="whitespace-pre-wrap font-sans text-xs leading-relaxed text-slate-100">
                          {trace.outputText}
                        </pre>
                      </div>
                      <div className="rounded-lg border border-slate-800 bg-slate-900/50 p-2.5">
                        <div className="mb-1 font-mono text-[10px] font-semibold text-slate-400">
                          BASE CHECKPOINT
                        </div>
                        <pre className="whitespace-pre-wrap font-sans text-xs leading-relaxed text-slate-400">
                          {trace.baseOutputText}
                        </pre>
                      </div>
                    </div>
                  ) : (
                    <pre className="whitespace-pre-wrap font-sans text-xs leading-relaxed text-slate-100">
                      {msg.text}
                    </pre>
                  )}

                  {/* Expandable Logits, Speculative Decoding & Layer Trace */}
                  {showTraceDetails && trace && (
                    <div className="space-y-2.5 border-t border-slate-800 pt-2.5">
                      <div>
                        <div className="mb-1 font-mono text-[10px] text-slate-400">
                          Input Subword & Modality Tokens ({trace.inputTokens.length}):
                        </div>
                        <div className="flex flex-wrap gap-1">
                          {trace.inputTokens.map((t, i) => (
                            <span
                              key={i}
                              className={`rounded border px-1.5 py-0.5 font-mono text-[10px] ${
                                t.modality === 'fim'
                                  ? 'border-indigo-500/50 bg-indigo-950/50 text-indigo-200'
                                  : t.modality === 'vision'
                                  ? 'border-amber-500/50 bg-amber-950/50 text-amber-200'
                                  : 'border-slate-800 bg-slate-900 text-cyan-200'
                              }`}
                            >
                              {t.text}{' '}
                              <span className="text-[9px] text-slate-500">#{t.id}</span>
                            </span>
                          ))}
                        </div>
                      </div>

                      <div>
                        <div className="mb-1 flex items-center justify-between font-mono text-[10px] text-slate-400">
                          <span>Output Token Confidence (Click token for Softmax ΔP):</span>
                          {trace.tokenTraces.some((t) => t.speculativeAccepted !== undefined) && (
                            <span className="text-indigo-300">
                              ⚡ Green = Draft Speculative Token Accepted
                            </span>
                          )}
                        </div>
                        <div className="flex flex-wrap gap-1">
                          {trace.tokenTraces.map((tt, idx) => {
                            const deltaP = tt.prob - tt.baseProb;
                            const isSel = selectedTokenTrace === tt;
                            return (
                              <button
                                key={idx}
                                onClick={() => setSelectedTokenTrace(tt)}
                                className={`rounded px-1.5 py-0.5 font-mono text-[10px] transition ${
                                  isSel
                                    ? 'bg-pink-500 text-white'
                                    : tt.speculativeAccepted
                                    ? 'bg-emerald-500/20 text-emerald-200 ring-1 ring-emerald-500/40'
                                    : deltaP > 0.05
                                    ? 'bg-cyan-500/20 text-cyan-200 ring-1 ring-cyan-500/30'
                                    : 'bg-slate-900 text-slate-300'
                                }`}
                              >
                                {tt.token}{' '}
                                <span className="text-[9px] opacity-75">
                                  {(tt.prob * 100).toFixed(0)}%
                                </span>
                              </button>
                            );
                          })}
                        </div>
                      </div>

                      {selectedTokenTrace && (
                        <div className="rounded-md border border-slate-800 bg-slate-900/75 p-2">
                          <div className="mb-1.5 flex items-center justify-between font-mono text-[10px]">
                            <span className="text-pink-300">
                              Softmax Candidates at &ldquo;{selectedTokenTrace.token}&rdquo; (ID #{selectedTokenTrace.tokenId})
                            </span>
                            <span className="text-slate-400">Live P vs Base P</span>
                          </div>
                          <div className="space-y-1">
                            {selectedTokenTrace.topCandidates.map((cand, ci) => (
                              <div key={ci} className="flex items-center gap-2 font-mono text-[10px]">
                                <span className="w-20 truncate text-slate-200">{cand.token}</span>
                                <div className="h-2 flex-1 overflow-hidden rounded-full bg-slate-800">
                                  <div
                                    className="h-full rounded-full bg-gradient-to-r from-cyan-500 to-pink-500"
                                    style={{ width: `${Math.min(100, cand.prob * 100)}%` }}
                                  />
                                </div>
                                <span className="w-24 text-right text-slate-300">
                                  {(cand.prob * 100).toFixed(0)}%{' '}
                                  <span className="text-slate-500">
                                    (base {(cand.baseProb * 100).toFixed(0)}%)
                                  </span>
                                </span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      <div>
                        <div className="mb-1 font-mono text-[10px] text-slate-400">
                          {architecture.numLayers}-Layer Forward Pass Energy (L00 → L{architecture.numLayers - 1}):
                        </div>
                        <div className="flex h-10 items-end gap-[2px] rounded bg-slate-900/70 p-1">
                          {trace.layerTraces.map((lt) => {
                            const h = Math.min(100, Math.max(10, (lt.attnEnergy + lt.mlpEnergy) * 26));
                            const color =
                              lt.status === 'ablated'
                                ? 'bg-rose-500/80'
                                : lt.status === 'amplified'
                                ? 'bg-amber-400'
                                : lt.deltaShift > 0.08
                                ? 'bg-emerald-400'
                                : 'bg-cyan-500/75';
                            return (
                              <div
                                key={lt.layerIndex}
                                title={`L${lt.layerIndex}: Attn ${lt.attnEnergy} | MLP ${lt.mlpEnergy} | Status: ${lt.status}`}
                                style={{ height: `${h}%` }}
                                className={`flex-1 rounded-t-[2px] ${color}`}
                              />
                            );
                          })}
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            );
          })
        )}
        <div ref={scrollEndRef} />
      </div>

      {/* Prompt Input & Quick Preset Pills */}
      <div className="border-t border-slate-800 bg-slate-950/80 p-2.5">
        {messages.length > 0 && inferMode === 'chat' && (
          <div className="mb-2 flex flex-wrap gap-1">
            {QUICK_TEST_PROMPTS.map((qp) => (
              <button
                key={qp.label}
                onClick={() => onSendPrompt(qp.prompt)}
                className="rounded border border-slate-800 bg-slate-900 px-2 py-0.5 text-[10px] text-slate-300 hover:border-cyan-500/40 hover:text-cyan-300"
              >
                {qp.label}
              </button>
            ))}
          </div>
        )}
        <form onSubmit={handleSubmit} className="flex items-center gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={
              inferMode === 'vision'
                ? `Ask about ${activeScene.title} (or click Infer for default 2D pointing)...`
                : inferMode === 'fim'
                ? 'Click Synthesize above or type custom FIM instruction...'
                : `Ask ${architecture.name.split(' ')[0]} anything to test your edited 3D weights...`
            }
            className="flex-1 rounded-lg border border-slate-700/80 bg-slate-900 px-3 py-2 text-xs text-slate-100 placeholder:text-slate-500 focus:border-pink-500 focus:outline-none"
          />
          <button
            type="submit"
            disabled={inferMode === 'chat' && !input.trim()}
            className="flex items-center gap-1.5 rounded-lg bg-pink-600 px-3.5 py-2 text-xs font-semibold text-white transition hover:bg-pink-500 disabled:opacity-40"
          >
            <Send className="h-3.5 w-3.5" />
            Infer
          </button>
        </form>
      </div>
    </div>
  );
};
