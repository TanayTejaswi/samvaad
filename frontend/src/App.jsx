import React, { useState, useEffect, useRef } from 'react';
import { Mic, MicOff, Settings, Sparkles, BookOpen, Clock, Activity, Zap } from 'lucide-react';

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
      // Use relative URL so it works anywhere
      const wsUrl = window.location.protocol === 'https:' ? 'wss://' : 'ws://' + window.location.host + '/captions';
      const ws = new WebSocket(wsUrl);
      
      ws.onopen = () => setStatus('idle');
      ws.onmessage = (event) => {
        // Only parse JSON messages
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
      
      // We use createScriptProcessor for broad compatibility, chunk size 4096
      const processor = audioCtx.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;
      
      processor.onaudioprocess = (e) => {
        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
        const channelData = e.inputBuffer.getChannelData(0);
        // Send Float32Array directly as binary buffer
        wsRef.current.send(channelData.buffer);
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
    const stops = new Set(["the", "and", "is", "in", "it", "to", "of", "for", "on", "that", "this", "with"]);
    const counts = {};
    words.forEach(w => {
      if (w.length > 3 && !stops.has(w)) counts[w] = (counts[w] || 0) + 1;
    });
    return Object.entries(counts).sort((a, b) => b[1] - a[1]).slice(0, 5).map(x => x[0]);
  };
  
  const keywords = extractKeywords(history.slice(0, 50).map(t => t.text));

  return (
    <div className="min-h-screen bg-mono-bg text-mono-textMain font-sans flex flex-col selection:bg-mono-accent selection:text-white transition-colors duration-500">
      
      {/* MINIMALIST HEADER */}
      <header className="h-20 bg-mono-surface/80 backdrop-blur-md border-b border-mono-border flex items-center justify-between px-8 sticky top-0 z-50">
        <div className="flex items-center gap-4">
          <div className="w-10 h-10 rounded-full bg-mono-accent flex items-center justify-center text-white shadow-shiny animate-fade-in">
            <Sparkles size={20} />
          </div>
          <div>
            <h1 className="font-display font-bold text-2xl tracking-tighter">Samvaad</h1>
            <p className="text-xs font-medium text-mono-textMuted uppercase tracking-widest">Live Transcriptions</p>
          </div>
        </div>
        
        <div className="flex items-center gap-6 animate-fade-in" style={{ animationDelay: '0.1s' }}>
          
          <button 
            onClick={toggleRecording}
            className={`flex items-center gap-2 border shadow-sm px-4 py-2 rounded-button text-sm font-semibold transition-all hover:-translate-y-0.5 active:translate-y-0 ${
              isRecording 
                ? 'bg-red-50 text-red-600 border-red-200 hover:shadow-shiny' 
                : 'bg-mono-accent text-white border-mono-accent hover:shadow-shiny-hover'
            }`}
          >
            {isRecording ? <MicOff size={16} /> : <Mic size={16} />} 
            {isRecording ? 'Stop Mic' : 'Start Mic'}
          </button>

          <button 
            onClick={simulateSpeech}
            className="flex items-center gap-2 bg-white text-mono-textMain border border-mono-border shadow-sm hover:shadow-shiny-hover px-4 py-2 rounded-button text-sm font-semibold transition-all hover:-translate-y-0.5"
          >
            <Zap size={16} /> Simulate
          </button>

          <div className="flex items-center gap-3 bg-mono-bg px-4 py-2 rounded-button border border-mono-border shadow-inner">
            <div className="relative flex h-3 w-3">
              {(status === 'transcribing' || status === 'listening') && <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-mono-accent opacity-40"></span>}
              <span className={`relative inline-flex rounded-full h-3 w-3 ${status === 'offline' ? 'bg-red-500' : 'bg-mono-accent'}`}></span>
            </div>
            <span className="text-sm font-bold uppercase tracking-wider">
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
            <h2 className="font-display text-3xl font-bold tracking-tight">Live Captions</h2>
            <Activity size={24} className="text-mono-textMuted" />
          </div>
          
          <div className="flex-1 bg-white rounded-card border border-mono-border shadow-shiny p-8 overflow-y-auto relative flex flex-col gap-6">
            {transcripts.length === 0 ? (
              <div className="absolute inset-0 flex flex-col items-center justify-center text-mono-textMuted opacity-50 animate-pulse-slow">
                <Mic size={64} className="mb-6 stroke-1" />
                <p className="text-xl font-display font-medium">Waiting for speech...</p>
                <p className="text-sm mt-2">Click "Start Mic" to stream from your browser!</p>
              </div>
            ) : (
              transcripts.map((t, i) => (
                <div key={i} className="animate-slide-up group">
                  <p className="text-2xl leading-snug font-medium text-mono-textMain">{t.text}</p>
                  <div className="mt-3 flex items-center gap-4 text-sm text-mono-textMuted font-mono opacity-60 group-hover:opacity-100 transition-opacity">
                    <span className="flex items-center gap-1.5"><Clock size={14} /> {new Date(t.timestamp).toLocaleTimeString()}</span>
                    <span className="flex items-center gap-1.5 bg-mono-bg px-2 py-0.5 rounded border border-mono-border"><Cpu size={14} /> {t.device.toUpperCase()} </span>
                    <span className="text-mono-accent font-semibold">{t.latency_ms}ms</span>
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
          <div className="bg-mono-accent text-white rounded-card shadow-shiny p-6">
            <h2 className="font-display text-lg font-bold mb-6 flex items-center gap-2">
              <BookOpen size={20} /> Extracted Topics
            </h2>
            {keywords.length > 0 ? (
              <div className="flex flex-wrap gap-2">
                {keywords.map(kw => (
                  <span key={kw} className="px-3 py-1.5 bg-white/10 hover:bg-white/20 transition-colors cursor-default text-sm rounded-button font-medium border border-white/10">
                    {kw}
                  </span>
                ))}
              </div>
            ) : (
              <p className="text-white/60 text-sm">Topics will appear here as the conversation progresses.</p>
            )}
          </div>
          
          {/* HISTORY */}
          <div className="flex-1 bg-white rounded-card border border-mono-border shadow-shiny p-6 flex flex-col overflow-hidden">
            <h2 className="font-display text-lg font-bold mb-6 text-mono-textMain flex items-center gap-2">
              <Clock size={20} className="text-mono-textMuted" /> History Log
            </h2>
            <div className="flex-1 overflow-y-auto pr-2 space-y-6">
              {history.length === 0 && <p className="text-sm text-mono-textMuted">No previous transcripts.</p>}
              {history.map((t, i) => (
                <div key={i} className="group cursor-default">
                  <p className="text-sm text-mono-textMain line-clamp-3 leading-relaxed group-hover:text-mono-accent transition-colors">{t.text}</p>
                  <span className="text-xs text-mono-textMuted mt-2 block font-mono">{new Date(t.timestamp).toLocaleTimeString()}</span>
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
