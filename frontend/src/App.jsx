import React, { useState, useEffect, useRef } from 'react';
import { Mic, MicOff, Sparkles, BookOpen, Clock, Activity, Zap } from 'lucide-react';

export default function App() {
  const [transcripts, setTranscripts] = useState([]);
  const [status, setStatus] = useState('offline'); 
  const [history, setHistory] = useState([]);
  const [isRecording, setIsRecording] = useState(false);
  
  const wsRef = useRef(null);
  const endOfMessagesRef = useRef(null);
  const audioContextRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const processorRef = useRef(null);

  useEffect(() => {
    fetch('/api/history')
      .then(res => res.json())
      .then(data => setHistory(data))
      .catch(err => console.error("History error:", err));
  }, []);

  useEffect(() => {
    const connectWs = () => {
      const wsUrl = window.location.protocol === 'https:' ? 'wss://' : 'ws://' + window.location.host + '/captions';
      const ws = new WebSocket(wsUrl);
      
      ws.onopen = () => setStatus('idle');
      ws.onmessage = (event) => {
        if (typeof event.data === 'string') {
          const data = JSON.parse(event.data);
          if (data.type === 'STATUS') {
            setStatus(data.status);
          } else if (data.type === 'TRANSCRIPT') {
            setTranscripts(prev => [...prev, data]);
            setHistory(prev => [data, ...prev]);
          }
        }
      };
      ws.onclose = () => {
        setStatus('offline');
        if (isRecording) stopRecording();
        setTimeout(connectWs, 3000);
      };
      wsRef.current = ws;
    };
    connectWs();
    return () => {
      if (isRecording) stopRecording();
      wsRef.current?.close();
    };
  }, []);

  useEffect(() => {
    endOfMessagesRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [transcripts]);

  const simulateSpeech = () => {
    fetch('/api/simulate', { method: 'POST' });
  };

  const startRecording = async () => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      alert("WebSocket is not connected.");
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaStreamRef.current = stream;
      
      const audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
      audioContextRef.current = audioCtx;
      
      const source = audioCtx.createMediaStreamSource(stream);
      const processor = audioCtx.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;
      
      processor.onaudioprocess = (e) => {
        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
        const channelData = e.inputBuffer.getChannelData(0);
        // Send a copy of the actual underlying data
        wsRef.current.send(new Float32Array(channelData).buffer);
      };

      source.connect(processor);
      processor.connect(audioCtx.destination);
      
      setIsRecording(true);
      setStatus('listening');
    } catch (err) {
      console.error("Microphone error:", err);
      alert("Could not access microphone.");
    }
  };

  const stopRecording = () => {
    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(track => track.stop());
      mediaStreamRef.current = null;
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    setIsRecording(false);
    setStatus('idle');
  };

  const toggleRecording = () => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  };

  const extractKeywords = (textList) => {
    const words = textList.join(" ").toLowerCase().replace(/[^a-z0-9 ]/g, "").split(" ");
    const stops = new Set(["the", "and", "is", "in", "it", "to", "of", "for", "on", "that", "this", "with", "mock", "npu"]);
    const counts = {};
    words.forEach(w => {
      if (w.length > 3 && !stops.has(w)) counts[w] = (counts[w] || 0) + 1;
    });
    return Object.entries(counts).sort((a, b) => b[1] - a[1]).slice(0, 5).map(x => x[0]);
  };
  
  const keywords = extractKeywords(history.slice(0, 50).map(t => t.text));

  return (
    <div className="min-h-screen bg-samvaad-bgPrimary text-samvaad-textPrimary font-sans flex flex-col selection:bg-samvaad-accentPrimary selection:text-white">
      
      {/* HEADER */}
      <header className="h-20 bg-samvaad-bgSecondary/80 backdrop-blur-md border-b border-samvaad-border flex items-center justify-between px-8 sticky top-0 z-50 shadow-glow">
        <div className="flex items-center gap-4">
          <div className="w-10 h-10 rounded-full bg-samvaad-accentPrimary flex items-center justify-center text-white shadow-glow animate-fade-in">
            <Sparkles size={20} />
          </div>
          <div>
            <h1 className="font-display font-bold text-2xl tracking-tighter text-white">Samvaad</h1>
            <p className="text-xs font-medium text-samvaad-accentPrimary uppercase tracking-widest">Hexagon NPU Engine</p>
          </div>
        </div>
        
        <div className="flex items-center gap-6 animate-fade-in" style={{ animationDelay: '0.1s' }}>
          
          <button 
            onClick={toggleRecording}
            className={`flex items-center gap-2 border px-5 py-2.5 rounded-button text-sm font-semibold transition-all hover:-translate-y-0.5 active:translate-y-0 ${
              isRecording 
                ? 'bg-red-500/10 text-red-500 border-red-500/30 hover:shadow-glow' 
                : 'bg-samvaad-accentPrimary text-white border-samvaad-accentPrimary hover:shadow-glow-hover'
            }`}
          >
            {isRecording ? <MicOff size={16} /> : <Mic size={16} />} 
            {isRecording ? 'Stop Mic' : 'Start Mic'}
          </button>

          <button 
            onClick={simulateSpeech}
            className="flex items-center gap-2 bg-samvaad-bgSecondary text-white border border-samvaad-border hover:border-samvaad-textMuted shadow-sm hover:shadow-glow px-4 py-2.5 rounded-button text-sm font-semibold transition-all hover:-translate-y-0.5"
          >
            <Zap size={16} /> Simulate
          </button>

          <div className="flex items-center gap-3 bg-samvaad-bgPrimary px-4 py-2.5 rounded-button border border-samvaad-border">
            <div className="relative flex h-3 w-3">
              {(status === 'transcribing' || status === 'listening') && <span className="animate-pulse-glow absolute inline-flex h-full w-full rounded-full bg-samvaad-accentPrimary opacity-60"></span>}
              <span className={`relative inline-flex rounded-full h-3 w-3 ${status === 'offline' ? 'bg-red-500' : 'bg-samvaad-accentPrimary'}`}></span>
            </div>
            <span className="text-sm font-bold uppercase tracking-wider text-samvaad-textMuted">
              {status}
            </span>
          </div>
        </div>
      </header>

      {/* MAIN CONTENT */}
      <main className="flex-1 flex max-w-7xl w-full mx-auto p-8 gap-8">
        
        {/* LIVE TRANSCRIPT FEED */}
        <section className="flex-1 flex flex-col relative">
          <div className="flex items-center justify-between mb-6 animate-fade-in" style={{ animationDelay: '0.2s' }}>
            <h2 className="font-display text-3xl font-bold tracking-tight text-white">Live Captions</h2>
            <Activity size={24} className="text-samvaad-textMuted" />
          </div>
          
          <div className="flex-1 bg-samvaad-bgSecondary rounded-card border border-samvaad-border shadow-glow p-8 overflow-y-auto relative flex flex-col gap-6">
            {transcripts.length === 0 ? (
              <div className="absolute inset-0 flex flex-col items-center justify-center text-samvaad-textMuted opacity-50 animate-pulse-glow">
                <Mic size={64} className="mb-6 stroke-1 text-samvaad-accentPrimary" />
                <p className="text-xl font-display font-medium text-white">Awaiting Audio Input...</p>
                <p className="text-sm mt-2">Click "Start Mic" to stream real-time.</p>
              </div>
            ) : (
              transcripts.map((t, i) => (
                <div key={i} className="animate-slide-up group">
                  <p className="text-2xl leading-snug font-medium text-white">{t.text}</p>
                  <div className="mt-3 flex items-center gap-4 text-sm text-samvaad-textMuted font-mono opacity-60 group-hover:opacity-100 transition-opacity">
                    <span className="flex items-center gap-1.5"><Clock size={14} /> {new Date(t.timestamp).toLocaleTimeString()}</span>
                    <span className="flex items-center gap-1.5 bg-samvaad-bgPrimary px-2 py-0.5 rounded border border-samvaad-border text-samvaad-accentSecondary"><Cpu size={14} /> {t.device.toUpperCase()} </span>
                    <span className="text-samvaad-accentPrimary font-semibold">{t.latency_ms}ms</span>
                  </div>
                </div>
              ))
            )}
            <div ref={endOfMessagesRef} />
          </div>
        </section>

        {/* SIDEBAR */}
        <aside className="w-[380px] flex flex-col gap-8 animate-fade-in" style={{ animationDelay: '0.3s' }}>
          
          {/* KEYWORDS */}
          <div className="bg-gradient-to-br from-samvaad-accentPrimary to-samvaad-accentSecondary text-white rounded-card shadow-glow p-6">
            <h2 className="font-display text-lg font-bold mb-6 flex items-center gap-2">
              <BookOpen size={20} /> Extracted Topics
            </h2>
            {keywords.length > 0 ? (
              <div className="flex flex-wrap gap-2">
                {keywords.map(kw => (
                  <span key={kw} className="px-3 py-1.5 bg-black/20 hover:bg-black/30 transition-colors cursor-default text-sm rounded-button font-medium border border-white/20">
                    {kw}
                  </span>
                ))}
              </div>
            ) : (
              <p className="text-white/80 text-sm font-medium">Topics will appear here as the conversation progresses.</p>
            )}
          </div>
          
          {/* HISTORY */}
          <div className="flex-1 bg-samvaad-bgSecondary rounded-card border border-samvaad-border shadow-lg p-6 flex flex-col overflow-hidden">
            <h2 className="font-display text-lg font-bold mb-6 text-white flex items-center gap-2">
              <Clock size={20} className="text-samvaad-textMuted" /> History Log
            </h2>
            <div className="flex-1 overflow-y-auto pr-2 space-y-6">
              {history.length === 0 && <p className="text-sm text-samvaad-textMuted">No previous transcripts.</p>}
              {history.map((t, i) => (
                <div key={i} className="group cursor-default border-l-2 border-transparent hover:border-samvaad-accentPrimary pl-3 transition-colors">
                  <p className="text-sm text-samvaad-textMuted line-clamp-3 leading-relaxed group-hover:text-white transition-colors">{t.text}</p>
                  <span className="text-xs text-samvaad-accentPrimary/70 mt-2 block font-mono">{new Date(t.timestamp).toLocaleTimeString()}</span>
                </div>
              ))}
            </div>
          </div>
        </aside>
      </main>
    </div>
  );
}

function Cpu(props) {
  return (
    <svg {...props} xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="4" y="4" width="16" height="16" rx="2" ry="2"></rect>
      <rect x="9" y="9" width="6" height="6"></rect>
      <line x1="9" y1="1" x2="9" y2="4"></line>
      <line x1="15" y1="1" x2="15" y2="4"></line>
      <line x1="9" y1="20" x2="9" y2="23"></line>
      <line x1="15" y1="20" x2="15" y2="23"></line>
      <line x1="20" y1="9" x2="23" y2="9"></line>
      <line x1="20" y1="14" x2="23" y2="14"></line>
      <line x1="1" y1="9" x2="4" y2="9"></line>
      <line x1="1" y1="14" x2="4" y2="14"></line>
    </svg>
  );
}
