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
    <div className="min-h-screen bg-samvaad-bgPrimary text-samvaad-textPrimary font-sans flex flex-col selection:bg-samvaad-accentPrimary selection:text-black">
      
      {/* HEADER */}
      <header className="h-20 bg-samvaad-bgPrimary border-b border-samvaad-border flex items-center justify-between px-8 sticky top-0 z-50">
        <div className="flex items-center gap-4">
          <div className="w-10 h-10 rounded-full bg-black flex items-center justify-center text-samvaad-accentPrimary shadow-glow animate-fade-in">
            <Sparkles size={20} />
          </div>
          <div>
            <h1 className="font-display font-black text-2xl tracking-tighter text-black">Samvaad</h1>
            <p className="text-xs font-bold text-samvaad-textMuted uppercase tracking-widest">Hexagon NPU Engine</p>
          </div>
        </div>
        
        <div className="flex items-center gap-6 animate-fade-in" style={{ animationDelay: '0.1s' }}>
          
          <button 
            onClick={toggleRecording}
            className={`flex items-center gap-2 border px-6 py-2.5 rounded-full text-sm font-bold transition-all hover:-translate-y-0.5 active:translate-y-0 ${
              isRecording 
                ? 'bg-red-50 text-red-600 border-red-200 shadow-sm' 
                : 'bg-black text-samvaad-accentPrimary border-black hover:shadow-glow-hover'
            }`}
          >
            {isRecording ? <MicOff size={16} /> : <Mic size={16} />} 
            {isRecording ? 'Stop Mic' : 'Start Mic'}
          </button>

          <button 
            onClick={simulateSpeech}
            className="flex items-center gap-2 bg-white text-black border border-samvaad-border hover:border-black shadow-sm px-4 py-2.5 rounded-full text-sm font-bold transition-all hover:-translate-y-0.5"
          >
            <Zap size={16} className="text-samvaad-accentPrimary" /> Simulate
          </button>

          <div className="flex items-center gap-3 bg-samvaad-bgSecondary px-5 py-2.5 rounded-full border border-samvaad-border">
            <div className="relative flex h-3 w-3">
              {(status === 'transcribing' || status === 'listening') && <span className="animate-pulse-glow absolute inline-flex h-full w-full rounded-full bg-samvaad-accentPrimary opacity-80"></span>}
              <span className={`relative inline-flex rounded-full h-3 w-3 ${status === 'offline' ? 'bg-red-500' : 'bg-black'}`}></span>
            </div>
            <span className="text-sm font-black uppercase tracking-widest text-black">
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
            <h2 className="font-display text-4xl font-black tracking-tighter text-black">Live Captions</h2>
            <Activity size={28} className="text-samvaad-textMuted opacity-30" />
          </div>
          
          <div className="flex-1 bg-samvaad-bgSecondary rounded-card border border-samvaad-border p-8 overflow-y-auto relative flex flex-col gap-6 shadow-sm">
            {transcripts.length === 0 ? (
              <div className="absolute inset-0 flex flex-col items-center justify-center text-samvaad-textMuted opacity-60 animate-pulse-glow">
                <Mic size={64} className="mb-6 stroke-1 text-black" />
                <p className="text-2xl font-display font-bold text-black">Awaiting Audio Input</p>
                <p className="text-base mt-2 font-medium">Click "Start Mic" to stream real-time.</p>
              </div>
            ) : (
              transcripts.map((t, i) => (
                <div key={i} className="animate-slide-up group bg-white p-6 rounded-card border border-samvaad-border shadow-sm hover:shadow-md transition-shadow">
                  <p className="text-3xl leading-snug font-bold text-black">{t.text}</p>
                  <div className="mt-4 flex items-center gap-4 text-sm text-samvaad-textMuted font-mono">
                    <span className="flex items-center gap-1.5"><Clock size={14} /> {new Date(t.timestamp).toLocaleTimeString()}</span>
                    <span className="flex items-center gap-1.5 bg-samvaad-bgSecondary px-2.5 py-1 rounded-full border border-samvaad-border text-black font-bold"><Cpu size={14} /> {t.device.toUpperCase()} </span>
                    <span className="text-samvaad-accentPrimary font-black bg-black px-2.5 py-1 rounded-full">{t.latency_ms}ms</span>
                  </div>
                </div>
              ))
            )}
            <div ref={endOfMessagesRef} />
          </div>
        </section>

        {/* SIDEBAR */}
        <aside className="w-[400px] flex flex-col gap-8 animate-fade-in" style={{ animationDelay: '0.3s' }}>
          
          {/* KEYWORDS */}
          <div className="bg-black text-white rounded-card shadow-glow p-8">
            <h2 className="font-display text-2xl font-black mb-6 flex items-center gap-3 text-samvaad-accentPrimary">
              <BookOpen size={24} /> Topics
            </h2>
            {keywords.length > 0 ? (
              <div className="flex flex-wrap gap-2.5">
                {keywords.map(kw => (
                  <span key={kw} className="px-4 py-2 bg-white/10 hover:bg-white/20 transition-colors cursor-default text-sm rounded-full font-bold border border-samvaad-accentPrimary/30 text-samvaad-accentPrimary">
                    {kw}
                  </span>
                ))}
              </div>
            ) : (
              <p className="text-white/60 text-sm font-medium">Topics will appear here as you speak.</p>
            )}
          </div>
          
          {/* HISTORY */}
          <div className="flex-1 bg-white rounded-card border border-samvaad-border shadow-sm p-8 flex flex-col overflow-hidden">
            <h2 className="font-display text-2xl font-black mb-6 text-black flex items-center gap-3">
              <Clock size={24} className="text-samvaad-textMuted opacity-50" /> History
            </h2>
            <div className="flex-1 overflow-y-auto pr-2 space-y-6">
              {history.length === 0 && <p className="text-sm text-samvaad-textMuted font-medium">No previous transcripts.</p>}
              {history.map((t, i) => (
                <div key={i} className="group cursor-default border-l-4 border-samvaad-border hover:border-samvaad-accentPrimary pl-4 py-1 transition-colors">
                  <p className="text-base font-semibold text-samvaad-textMuted line-clamp-3 leading-relaxed group-hover:text-black transition-colors">{t.text}</p>
                  <span className="text-xs text-black mt-2 block font-mono font-bold">{new Date(t.timestamp).toLocaleTimeString()}</span>
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
