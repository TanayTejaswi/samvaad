import React, { useState, useEffect } from 'react'

export default function App() {
  const [status, setStatus] = useState('idle')
  const [device, setDevice] = useState('npu')
  const [latency, setLatency] = useState(0)
  const [currentCaption, setCurrentCaption] = useState('Ready to transcribe speech on Qualcomm Hexagon NPU...')
  const [history, setHistory] = useState([])

  return (
    <div className="min-h-screen bg-[#0F172A] text-[#F8FAFC] flex flex-col font-sans">
      {/* Header */}
      <header className="border-b border-[#334155] bg-[#1E293B]/80 backdrop-blur-md px-6 py-4 sticky top-0 z-50 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="h-9 w-9 rounded-[8px] bg-[#6366F1] flex items-center justify-center font-bold text-white shadow-samvaad-soft">
            सं
          </div>
          <div>
            <h1 className="text-xl font-display font-bold tracking-tight text-[#F8FAFC]">
              SAMVAAD <span className="text-xs font-mono font-normal uppercase tracking-wider text-[#14B8A6] ml-2 px-2 py-0.5 rounded border border-[#14B8A6]/30 bg-[#14B8A6]/10">Shravan Core</span>
            </h1>
            <p className="text-xs text-[#94A3B8]">Offline NPU-Accelerated Speech Perception</p>
          </div>
        </div>

        {/* Status Indicator */}
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2 px-3 py-1.5 rounded-[8px] border border-[#334155] bg-[#1E293B] text-xs font-mono">
            <span className="h-2 w-2 rounded-full bg-[#14B8A6] animate-pulse"></span>
            <span className="text-[#94A3B8]">DEVICE:</span>
            <span className="text-[#F8FAFC] font-semibold uppercase">{device} (Hexagon HTP)</span>
          </div>

          <div className="flex items-center space-x-2 px-3 py-1.5 rounded-[8px] border border-[#334155] bg-[#1E293B] text-xs font-mono">
            <span className="text-[#94A3B8]">STATUS:</span>
            <span className="text-[#14B8A6] font-semibold uppercase">{status}</span>
          </div>
        </div>
      </header>

      {/* Main Content Viewport */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-6 grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Real-time Caption Main Stage (2 cols) */}
        <div className="lg:col-span-2 flex flex-col space-y-6">
          <div className="rounded-[12px] border border-[#334155] bg-[#1E293B] p-8 shadow-samvaad-soft flex-1 flex flex-col justify-between min-h-[380px]">
            <div>
              <div className="flex items-center justify-between pb-4 border-b border-[#334155]/60 mb-6">
                <span className="text-xs font-mono text-[#94A3B8] uppercase tracking-wider">Live Caption Stream</span>
                <span className="text-xs font-mono px-2 py-1 rounded bg-[#0F172A] border border-[#334155] text-[#94A3B8]">
                  Latency: <span className="text-[#F8FAFC]">{latency} ms</span>
                </span>
              </div>
              <div className="text-2xl md:text-4xl font-display font-medium leading-relaxed text-[#F8FAFC] transition-all">
                {currentCaption}
              </div>
            </div>

            <div className="pt-6 border-t border-[#334155]/60 flex items-center justify-between text-xs text-[#94A3B8] font-mono">
              <div className="flex items-center space-x-2">
                <span className="h-2 w-2 rounded-full bg-[#6366F1]"></span>
                <span>Microphone: 16kHz Mono (Local Ring Buffer)</span>
              </div>
              <span>No Cloud Calls • 100% Offline</span>
            </div>
          </div>

          {/* Performance Telemetry Bar */}
          <div className="grid grid-cols-3 gap-4">
            <div className="rounded-[8px] border border-[#334155] bg-[#1E293B] p-4">
              <div className="text-xs text-[#94A3B8]">Inference Latency</div>
              <div className="text-xl font-mono font-bold text-[#F8FAFC] mt-1">{latency} ms</div>
            </div>
            <div className="rounded-[8px] border border-[#334155] bg-[#1E293B] p-4">
              <div className="text-xs text-[#94A3B8]">Execution Provider</div>
              <div className="text-xl font-mono font-bold text-[#14B8A6] mt-1">QNN HTP</div>
            </div>
            <div className="rounded-[8px] border border-[#334155] bg-[#1E293B] p-4">
              <div className="text-xs text-[#94A3B8]">Fallback Status</div>
              <div className="text-xl font-mono font-bold text-[#6366F1] mt-1">Strict (No CPU)</div>
            </div>
          </div>
        </div>

        {/* Sidebar: Transcript History & Notes (1 col) */}
        <div className="rounded-[12px] border border-[#334155] bg-[#1E293B] p-6 shadow-samvaad-soft flex flex-col h-[520px]">
          <h2 className="text-sm font-mono uppercase tracking-wider text-[#94A3B8] pb-3 border-b border-[#334155]">
            Transcript History
          </h2>
          <div className="flex-1 overflow-y-auto mt-4 space-y-3 pr-2">
            {history.length === 0 ? (
              <div className="text-center py-12 text-sm text-[#94A3B8]">
                No previous utterances recorded yet.
              </div>
            ) : (
              history.map((item, idx) => (
                <div key={idx} className="p-3 rounded-[8px] bg-[#0F172A] border border-[#334155] text-sm">
                  <div className="text-[10px] font-mono text-[#94A3B8] mb-1">{item.timestamp}</div>
                  <div className="text-[#F8FAFC]">{item.text}</div>
                </div>
              ))
            )}
          </div>
        </div>
      </main>
    </div>
  )
}
