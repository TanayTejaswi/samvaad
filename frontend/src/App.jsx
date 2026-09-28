import React, { useState, useEffect, useRef } from 'react';
import { Mic, MicOff, Settings, Activity, BookOpen, Clock, Cpu } from 'lucide-react';

export default function App() {
  const [transcripts, setTranscripts] = useState([]);
  const [status, setStatus] = useState('offline'); // offline, idle, transcribing
  const [history, setHistory] = useState([]);
  const wsRef = useRef(null);
  const endOfMessagesRef = useRef(null);

  // Fetch initial history
  useEffect(() => {
    fetch('http://localhost:8000/api/history')
      .then(res => res.json())
      .then(data => {
        setHistory(data);
      })
      .catch(err => console.error("Failed to load history:", err));
  }, []);

  // WebSocket Connection
  useEffect(() => {
    const connectWs = () => {
      // Connect to the fastapi backend
      const ws = new WebSocket('ws://localhost:8000/captions');
      
      ws.onopen = () => {
        setStatus('idle');
      };
      
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'STATUS') {
          setStatus(data.status);
        } else if (data.type === 'TRANSCRIPT') {
          setTranscripts(prev => [...prev, data]);
          setHistory(prev => [data, ...prev]);
        }
      };
      
      ws.onclose = () => {
        setStatus('offline');
        // Try to reconnect after 3 seconds
        setTimeout(connectWs, 3000);
      };
      
      wsRef.current = ws;
    };

    connectWs();
    return () => {
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  // Auto-scroll to newest transcript
  useEffect(() => {
    endOfMessagesRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [transcripts]);

  // Derived state for Notes Panel (Primitive Extractive Summary mockup)
  const extractKeywords = (textList) => {
    const words = textList.join(" ").toLowerCase().replace(/[^a-z0-9 ]/g, "").split(" ");
    const stops = new Set(["the", "and", "is", "in", "it", "to", "of", "for", "on", "that", "this", "with"]);
    const counts = {};
    words.forEach(w => {
      if (w.length > 3 && !stops.has(w)) counts[w] = (counts[w] || 0) + 1;
    });
    return Object.entries(counts).sort((a, b) => b[1] - a[1]).slice(0, 5).map(x => x[0]);
  };
  
  const keywords = extractKeywords(history.slice(0, 50).map(t => t.text));

  return (
    <div className="min-h-screen bg-samvaad-bgPrimary text-samvaad-textPrimary font-sans flex flex-col">
      {/* HEADER */}
      <header className="h-16 bg-samvaad-bgSecondary border-b border-samvaad-border flex items-center justify-between px-6 shrink-0">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-control bg-samvaad-accentPrimary flex items-center justify-center shadow-samvaad-soft">
            <Mic size={18} className="text-white" />
          </div>
          <h1 className="font-display font-semibold text-xl tracking-tight">Samvaad <span className="text-samvaad-textMuted text-sm font-normal ml-2">by Shravan</span></h1>
        </div>
        
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 bg-samvaad-bgPrimary px-3 py-1.5 rounded-control border border-samvaad-border">
            {status === 'transcribing' ? (
              <><Activity size={16} className="text-samvaad-accentSecondary animate-pulse" /> <span className="text-sm font-medium">Listening...</span></>
            ) : status === 'idle' ? (
              <><Mic size={16} className="text-samvaad-textMuted" /> <span className="text-sm font-medium text-samvaad-textMuted">Idle</span></>
            ) : (
              <><MicOff size={16} className="text-red-400" /> <span className="text-sm font-medium text-red-400">Offline</span></>
            )}
          </div>
          <button className="p-2 text-samvaad-textMuted hover:text-samvaad-textPrimary transition-colors rounded-control hover:bg-samvaad-bgPrimary">
            <Settings size={20} />
          </button>
        </div>
      </header>

      {/* MAIN CONTENT GRID */}
      <main className="flex-1 flex overflow-hidden">
        
        {/* LEFT PANEL: Live Captions */}
        <section className="flex-1 flex flex-col p-6 overflow-hidden">
          <h2 className="font-display text-lg font-medium mb-4 text-samvaad-textMuted flex items-center gap-2">
            <Activity size={18} /> Live Transcript
          </h2>
          
          <div className="flex-1 bg-samvaad-bgSecondary rounded-card border border-samvaad-border p-6 overflow-y-auto shadow-samvaad-soft flex flex-col gap-4 relative">
            {transcripts.length === 0 ? (
              <div className="absolute inset-0 flex flex-col items-center justify-center text-samvaad-textMuted opacity-50">
                <Mic size={48} className="mb-4" />
                <p>Waiting for speech...</p>
              </div>
            ) : (
              transcripts.map((t, i) => (
                <div key={i} className="animate-fade-in bg-samvaad-bgPrimary p-4 rounded-control border border-samvaad-border shadow-sm">
                  <p className="text-lg leading-relaxed">{t.text}</p>
                  <div className="mt-2 flex items-center gap-3 text-xs text-samvaad-textMuted font-mono">
                    <span className="flex items-center gap-1"><Clock size={12} /> {new Date(t.timestamp).toLocaleTimeString()}</span>
                    <span className="flex items-center gap-1"><Cpu size={12} /> {t.device.toUpperCase()} ({t.latency_ms}ms)</span>
                  </div>
                </div>
              ))
            )}
            <div ref={endOfMessagesRef} />
          </div>
        </section>

        {/* RIGHT PANEL: History & Notes */}
        <aside className="w-96 border-l border-samvaad-border bg-samvaad-bgSecondary flex flex-col">
          {/* Notes Section */}
          <div className="p-6 border-b border-samvaad-border">
            <h2 className="font-display text-lg font-medium mb-4 text-samvaad-textMuted flex items-center gap-2">
              <BookOpen size={18} /> Offline Notes
            </h2>
            <div className="bg-samvaad-bgPrimary rounded-control p-4 border border-samvaad-border min-h-[120px]">
              <h3 className="text-sm font-semibold mb-2 text-samvaad-textMuted uppercase tracking-wider">Key Topics Detected</h3>
              {keywords.length > 0 ? (
                <div className="flex flex-wrap gap-2">
                  {keywords.map(kw => (
                    <span key={kw} className="px-2 py-1 bg-samvaad-accentPrimary/20 text-samvaad-accentPrimary text-xs rounded-full font-medium border border-samvaad-accentPrimary/30">
                      {kw}
                    </span>
                  ))}
                </div>
              ) : (
                <p className="text-sm text-samvaad-textMuted">Not enough data to extract topics.</p>
              )}
            </div>
          </div>
          
          {/* History Section */}
          <div className="flex-1 p-6 overflow-hidden flex flex-col">
            <h2 className="font-display text-lg font-medium mb-4 text-samvaad-textMuted flex items-center gap-2">
              <Clock size={18} /> History
            </h2>
            <div className="flex-1 overflow-y-auto pr-2 space-y-3">
              {history.map((t, i) => (
                <div key={i} className="text-sm border-l-2 border-samvaad-border pl-3 py-1 opacity-70 hover:opacity-100 transition-opacity">
                  <p className="text-samvaad-textPrimary line-clamp-2">{t.text}</p>
                  <span className="text-xs text-samvaad-textMuted mt-1 block font-mono">{new Date(t.timestamp).toLocaleTimeString()}</span>
                </div>
              ))}
            </div>
          </div>
        </aside>
      </main>
    </div>
  );
}
