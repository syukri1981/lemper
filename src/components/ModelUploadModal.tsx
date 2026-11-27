import React, { useRef, useState } from 'react';
import {
  CheckCircle2,
  Code2,
  Cpu,
  Download,
  Eye,
  FileCode,
  HardDriveUpload,
  Network,
  Sparkles,
  Upload,
  X,
} from 'lucide-react';
import { ModelArchitectureProfile } from '../types/olmo';
import { BUILTIN_MODELS } from '../engine/modelRegistry';
import {
  generateSampleSafetensorsFile,
  ParsedModelUploadResult,
  parseUploadedModelFile,
} from '../engine/modelFileParser';

interface ModelUploadModalProps {
  isOpen: boolean;
  onClose: () => void;
  currentArch: ModelArchitectureProfile;
  onSelectBuiltinModel: (arch: ModelArchitectureProfile) => void;
  onApplyParsedUpload: (result: ParsedModelUploadResult) => void;
}

export const ModelUploadModal: React.FC<ModelUploadModalProps> = ({
  isOpen,
  onClose,
  currentArch,
  onSelectBuiltinModel,
  onApplyParsedUpload,
}) => {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isParsing, setIsParsing] = useState(false);
  const [parsedResult, setParsedResult] = useState<ParsedModelUploadResult | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  if (!isOpen) return null;

  const handleFile = async (file: File) => {
    setIsParsing(true);
    setErrorMsg(null);
    try {
      const res = await parseUploadedModelFile(file);
      setParsedResult(res);
      onApplyParsedUpload(res);
    } catch (err) {
      setErrorMsg(
        err instanceof Error ? err.message : 'Failed to parse uploaded model file.'
      );
    } finally {
      setIsParsing(false);
    }
  };

  const handleGenerateAndLoadSample = async (variant: 'coder' | 'multimodal' | 'olmo') => {
    const sampleFile = generateSampleSafetensorsFile(variant);
    await handleFile(sampleFile);
  };

  const handleDownloadSampleToDisk = (variant: 'coder' | 'multimodal' | 'olmo') => {
    const sampleFile = generateSampleSafetensorsFile(variant);
    const url = URL.createObjectURL(sampleFile);
    const a = document.createElement('a');
    a.href = url;
    a.download = sampleFile.name;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 p-4 backdrop-blur-md">
      <div className="relative flex max-h-[90vh] w-full max-w-4xl flex-col overflow-hidden rounded-2xl border border-slate-700/80 bg-[#0b111e] shadow-2xl">
        {/* Modal Header */}
        <div className="flex items-center justify-between border-b border-slate-800 bg-slate-950/80 px-5 py-3.5">
          <div className="flex items-center gap-2.5">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-cyan-500 to-pink-500">
              <HardDriveUpload className="h-4 w-4 text-white" />
            </div>
            <div>
              <h2 className="text-sm font-bold text-white">
                Upload AI Model Checkpoint or Switch Open Architecture
              </h2>
              <p className="font-mono text-[11px] text-slate-400">
                Supports .safetensors, .gguf, HuggingFace config.json, .onnx & .pt/.bin — Rendered in Three.js 3D Engine
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="rounded-lg p-1.5 text-slate-400 hover:bg-slate-800 hover:text-white"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Modal Body */}
        <div className="flex-1 space-y-5 overflow-y-auto p-5">
          {/* 1. Drag & Drop Model File Uploader */}
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-12">
            <div
              onDragOver={(e) => {
                e.preventDefault();
                setIsDragging(true);
              }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={(e) => {
                e.preventDefault();
                setIsDragging(false);
                const f = e.dataTransfer.files?.[0];
                if (f) handleFile(f);
              }}
              onClick={() => fileInputRef.current?.click()}
              className={`flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed p-5 text-center transition lg:col-span-7 ${
                isDragging
                  ? 'border-cyan-400 bg-cyan-950/30'
                  : 'border-slate-700/80 bg-slate-950/60 hover:border-cyan-500/60 hover:bg-slate-900/60'
              }`}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".safetensors,.gguf,.json,.onnx,.pt,.bin,.pth"
                className="hidden"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) handleFile(f);
                }}
              />
              <div className="mb-2 flex h-11 w-11 items-center justify-center rounded-full bg-cyan-500/15 text-cyan-300 ring-1 ring-cyan-500/30">
                <Upload className="h-5 w-5" />
              </div>
              <div className="text-xs font-semibold text-slate-100">
                {isParsing
                  ? 'Parsing Binary Tensor Headers & Extracting Weights...'
                  : 'Drop your AI Model File here, or Click to Browse'}
              </div>
              <p className="mt-1 max-w-md text-[11px] text-slate-400">
                Reads <code className="text-cyan-300">.safetensors</code> 8-byte uint64 JSON header & raw FP32/FP16/BF16 weight slices,{' '}
                <code className="text-pink-300">.gguf</code> magic headers, or{' '}
                <code className="text-emerald-300">config.json</code>.
              </p>
              <div className="mt-3 flex flex-wrap justify-center gap-1.5 font-mono text-[10px]">
                <span className="rounded bg-slate-900 px-2 py-0.5 text-cyan-300 ring-1 ring-slate-800">
                  .safetensors
                </span>
                <span className="rounded bg-slate-900 px-2 py-0.5 text-pink-300 ring-1 ring-slate-800">
                  .gguf
                </span>
                <span className="rounded bg-slate-900 px-2 py-0.5 text-emerald-300 ring-1 ring-slate-800">
                  config.json
                </span>
                <span className="rounded bg-slate-900 px-2 py-0.5 text-amber-300 ring-1 ring-slate-800">
                  .onnx / .pt
                </span>
              </div>
            </div>

            {/* Instant Binary .safetensors Generator & Downloader for Testing */}
            <div className="flex flex-col justify-between rounded-xl border border-slate-800 bg-slate-950/70 p-4 lg:col-span-5">
              <div>
                <div className="flex items-center gap-1.5 text-xs font-semibold text-white">
                  <FileCode className="h-4 w-4 text-emerald-400" />
                  Don&apos;t have a local checkpoint handy?
                </div>
                <p className="mt-1 text-[11px] text-slate-400">
                  Synthesize a real spec-compliant binary <code className="text-emerald-300">.safetensors</code> file
                  (with 8-byte uint64 header + JSON tensor map + FP32 weight payload) to test our parser or download to disk:
                </p>
              </div>

              <div className="mt-3 space-y-2">
                {[
                  {
                    variant: 'coder' as const,
                    title: 'Qwen2.5-Coder-7B-GQA-FIM.safetensors',
                    badge: 'Coding AI (28L GQA)',
                    cls: 'border-indigo-500/40 bg-indigo-950/25 text-indigo-200',
                  },
                  {
                    variant: 'multimodal' as const,
                    title: 'Molmo-7B-O-Vision-Language.safetensors',
                    badge: 'Multimodal VLM (ViT+LLM)',
                    cls: 'border-amber-500/40 bg-amber-950/25 text-amber-200',
                  },
                  {
                    variant: 'olmo' as const,
                    title: 'OLMo-7B-Dolma-OpenWeights.safetensors',
                    badge: 'General Open LLM (32L)',
                    cls: 'border-cyan-500/40 bg-cyan-950/25 text-cyan-200',
                  },
                ].map((item) => (
                  <div
                    key={item.variant}
                    className={`flex items-center justify-between gap-2 rounded-lg border px-2.5 py-2 ${item.cls}`}
                  >
                    <div className="min-w-0">
                      <div className="truncate font-mono text-[10px] font-semibold">
                        {item.title}
                      </div>
                      <div className="text-[10px] opacity-80">{item.badge}</div>
                    </div>
                    <div className="flex shrink-0 items-center gap-1">
                      <button
                        onClick={() => handleGenerateAndLoadSample(item.variant)}
                        className="rounded bg-emerald-600 px-2 py-1 text-[10px] font-semibold text-white hover:bg-emerald-500"
                        title="Parse & Load directly into 3D Engine"
                      >
                        Parse & Load
                      </button>
                      <button
                        onClick={() => handleDownloadSampleToDisk(item.variant)}
                        className="rounded bg-slate-900 p-1 text-slate-300 hover:bg-slate-800 hover:text-white"
                        title="Download .safetensors file to disk so you can drag & drop it"
                      >
                        <Download className="h-3.5 w-3.5" />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {errorMsg && (
            <div className="rounded-lg border border-rose-500/40 bg-rose-950/40 p-3 text-xs text-rose-200">
              {errorMsg}
            </div>
          )}

          {/* Parsed File Telemetry Inspector */}
          {(parsedResult || currentArch.sourceFileInfo) && (
            <div className="rounded-xl border border-emerald-500/40 bg-emerald-950/15 p-4">
              {(() => {
                const info = parsedResult?.summary || currentArch.sourceFileInfo!;
                return (
                  <div className="space-y-2.5">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <div className="flex items-center gap-2">
                        <CheckCircle2 className="h-4 w-4 text-emerald-400" />
                        <span className="font-mono text-xs font-bold text-emerald-300">
                          Loaded Uploaded Model File: {info.fileName}
                        </span>
                      </div>
                      <span className="rounded bg-emerald-500/20 px-2 py-0.5 font-mono text-[10px] text-emerald-200">
                        {info.fileFormat.toUpperCase()} • {(info.fileSizeBytes / 1024).toFixed(1)} KB inspected
                      </span>
                    </div>

                    <div className="grid grid-cols-2 gap-2 font-mono text-[11px] sm:grid-cols-4">
                      <div className="rounded bg-slate-950/80 p-2">
                        <div className="text-[9px] text-slate-500">DETECTED TOPOLOGY</div>
                        <div className="truncate text-cyan-300">{info.detectedArchitecture}</div>
                      </div>
                      <div className="rounded bg-slate-950/80 p-2">
                        <div className="text-[9px] text-slate-500">PARSED TENSORS</div>
                        <div className="text-white">{info.parsedTensorCount} Matrices</div>
                      </div>
                      <div className="rounded bg-slate-950/80 p-2">
                        <div className="text-[9px] text-slate-500">EXTRACTED SCALARS</div>
                        <div className="text-emerald-300">
                          {info.realWeightsExtracted.toLocaleString()} floats
                        </div>
                      </div>
                      <div className="rounded bg-slate-950/80 p-2">
                        <div className="text-[9px] text-slate-500">DTYPE / QUANT</div>
                        <div className="text-amber-300">{info.quantizationType}</div>
                      </div>
                    </div>

                    <div className="rounded bg-slate-950/90 p-2 font-mono text-[10px] text-slate-400">
                      <span className="text-slate-500">Binary Header Preview: </span>
                      <span className="break-all text-slate-300">{info.rawHeaderPreview}</span>
                    </div>
                  </div>
                );
              })()}
            </div>
          )}

          {/* 2. Pre-Configured Reference Open-Source AI Models */}
          <div>
            <div className="mb-2.5 flex items-center justify-between">
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-300">
                Or Switch Reference Open-Source AI Model Architecture
              </span>
              <span className="font-mono text-[10px] text-slate-400">
                All rendered with Three.js 3D InstancedMesh Voxels
              </span>
            </div>

            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
              {BUILTIN_MODELS.map((m) => {
                const isSelected = currentArch.id === m.id;
                const Icon =
                  m.family === 'coding-llm'
                    ? Code2
                    : m.family === 'multimodal-vlm'
                    ? Eye
                    : m.family === 'moe-llm'
                    ? Network
                    : Cpu;

                return (
                  <button
                    key={m.id}
                    onClick={() => {
                      onSelectBuiltinModel(m);
                      onClose();
                    }}
                    className={`flex flex-col justify-between rounded-xl border p-3.5 text-left transition ${
                      isSelected
                        ? 'border-cyan-500/70 bg-cyan-950/25 ring-1 ring-cyan-500/40'
                        : 'border-slate-800 bg-slate-950/60 hover:border-slate-700 hover:bg-slate-900/50'
                    }`}
                  >
                    <div>
                      <div className="flex items-start justify-between gap-2">
                        <div className="flex items-center gap-2">
                          <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-slate-900 text-cyan-300 ring-1 ring-slate-700">
                            <Icon className="h-4 w-4" />
                          </div>
                          <div>
                            <div className="text-xs font-bold text-white">{m.name}</div>
                            <div className="text-[10px] text-slate-400">{m.organization}</div>
                          </div>
                        </div>
                        <span className="rounded bg-pink-500/15 px-2 py-0.5 font-mono text-[10px] font-medium text-pink-300">
                          {m.badge}
                        </span>
                      </div>

                      <div className="mt-2.5 flex flex-wrap gap-1.5 font-mono text-[10px] text-slate-300">
                        <span className="rounded bg-slate-900 px-1.5 py-0.5">
                          {m.numLayers}L • d={m.dModel}
                        </span>
                        <span className="rounded bg-slate-900 px-1.5 py-0.5">
                          Heads {m.numHeads}Q:{m.numKvHeads}KV
                        </span>
                        <span className="rounded bg-slate-900 px-1.5 py-0.5">
                          MLP {m.mlpIntermediate.toLocaleString()}
                        </span>
                        {m.visionConfig && (
                          <span className="rounded bg-amber-500/20 px-1.5 py-0.5 text-amber-300">
                            +ViT-L/14 & 2×2 Pool
                          </span>
                        )}
                        {m.codingConfig && (
                          <span className="rounded bg-indigo-500/20 px-1.5 py-0.5 text-indigo-300">
                            +FIM & 1M RoPE
                          </span>
                        )}
                      </div>
                    </div>

                    <div className="mt-2.5 flex items-center justify-between border-t border-slate-800/80 pt-2 text-[10px]">
                      <span className="text-slate-400">
                        Signature: {m.signatureAlgorithms.map((a) => a.name.split(' ')[0]).join(' • ')}
                      </span>
                      <span className="flex items-center gap-1 font-semibold text-cyan-300">
                        <Sparkles className="h-3 w-3" />
                        {isSelected ? 'Active' : 'Load 3D'}
                      </span>
                    </div>
                  </button>
                );
              })}
            </div>
          </div>
        </div>

        {/* Modal Footer */}
        <div className="flex items-center justify-between border-t border-slate-800 bg-slate-950/90 px-5 py-3">
          <span className="font-mono text-[11px] text-slate-400">
            Active Model: <strong className="text-cyan-300">{currentArch.name}</strong>
          </span>
          <button
            onClick={onClose}
            className="rounded-lg bg-cyan-600 px-4 py-1.5 text-xs font-semibold text-white hover:bg-cyan-500"
          >
            Explore in 3D Studio
          </button>
        </div>
      </div>
    </div>
  );
};
