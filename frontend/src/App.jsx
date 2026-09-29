import React, { useState, useEffect, useRef } from 'react';
import { Mic, MicOff, Sparkles, BookOpen, Clock, Activity, Zap, Video, VideoOff, Hand } from 'lucide-react';

export default function App() {
  const [transcripts, setTranscripts] = useState([]);
  const [status, setStatus] = useState('offline'); 
  const [signStatus, setSignStatus] = useState('idle');
  const [isRecording, setIsRecording] = useState(false);
  const [isVideoOn, setIsVideoOn] = useState(false);
  
  const wsRef = useRef(null);
  const videoWsRef = useRef(null);
  const endOfMessagesRef = useRef(null);
  const audioContextRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const videoStreamRef = useRef(null);
  const processorRef = useRef(null);
  
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const requestRef = useRef(null);

  useEffect(() => {
    // Audio WebSocket
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
            setTranscripts(prev => [...prev, {...data, source: 'speech'}]);
          }
        }
      };
      ws.onclose = () => setStatus('offline');
      wsRef.current = ws;
    };
    
    // Video WebSocket
    const connectVideoWs = () => {
      const wsUrl = window.location.protocol === 'https:' ? 'wss://' : 'ws://' + window.location.host + '/video';
      const ws = new WebSocket(wsUrl);
      
      ws.onopen = () => console.log("Video WS Connected");
      ws.onmessage = (event) => {
        if (typeof event.data === 'string') {
          const data = JSON.parse(event.data);
          if (data.type === 'STATUS') {
            setSignStatus(data.status);
          } else if (data.type === 'SIGN_RECOGNIZED') {
            setTranscripts(prev => [...prev, {...data, source: 'sign', text: data.gloss}]);
          }
        }
      };
      videoWsRef.current = ws;
    };

    connectWs();
    connectVideoWs();
    
    return () => {
      wsRef.current?.close();
      videoWsRef.current?.close();
      if (mediaStreamRef.current) mediaStreamRef.current.getTracks().forEach(track => track.stop());
      if (videoStreamRef.current) videoStreamRef.current.getTracks().forEach(track => track.stop());
      if (audioContextRef.current) audioContextRef.current.close();
      cancelAnimationFrame(requestRef.current);
    };
  }, []);

  useEffect(() => {
    endOfMessagesRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [transcripts]);

  const toggleRecording = async () => {
    if (isRecording) {
      setIsRecording(false);
      mediaStreamRef.current?.getTracks().forEach(track => track.stop());
      audioContextRef.current?.close();
      setStatus('idle');
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
      alert("Microphone error: " + err.message);
    }
  };

  const toggleVideo = async () => {
    if (isVideoOn) {
      setIsVideoOn(false);
      videoStreamRef.current?.getTracks().forEach(track => track.stop());
      cancelAnimationFrame(requestRef.current);
      return;
    }
    
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
      videoStreamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      setIsVideoOn(true);
      sendFrames();
    } catch (err) {
      alert("Camera error: " + err.message);
    }
  };

  const sendFrames = () => {
    if (!videoRef.current || !canvasRef.current || !isVideoOn) return;
    
    const ctx = canvasRef.current.getContext('2d');
    ctx.drawImage(videoRef.current, 0, 0, 320, 240); // Resize to 320x240 for faster transmission
    
    // Send as JPEG
    canvasRef.current.toBlob(blob => {
      if (blob && videoWsRef.current?.readyState === WebSocket.OPEN) {
        blob.arrayBuffer().then(buffer => {
           videoWsRef.current.send(buffer);
        });
      }
    }, 'image/jpeg', 0.5);
    
    // Stream at ~10 fps for now
    setTimeout(() => {
        requestRef.current = requestAnimationFrame(sendFrames);
    }, 100);
  };

  return (
    <div className="min-h-screen bg-samvaad-bgPrimary text-samvaad-textPrimary font-sans flex flex-col">
      {/* HEADER */}
      <header className="h-20 bg-samvaad-bgPrimary border-b border-samvaad-border flex items-center justify-between px-8 sticky top-0 z-50">
        <div className="flex items-center gap-4">
          <div className="w-10 h-10 rounded-full bg-black flex items-center justify-center text-samvaad-accentPrimary">
            <Sparkles size={20} />
          </div>
          <div>
            <h1 className="font-display font-black text-2xl tracking-tighter text-black">Samvaad (Two-Way Bridge)</h1>
            <p className="text-xs font-bold text-samvaad-textMuted uppercase tracking-widest">Hexagon NPU Engine</p>
          </div>
        </div>
        
        <div className="flex items-center gap-6">
          <button onClick={toggleVideo} className={`flex items-center gap-2 border px-6 py-2.5 rounded-full text-sm font-bold ${isVideoOn ? 'bg-red-50 text-red-600' : 'bg-black text-samvaad-accentPrimary'}`}>
            {isVideoOn ? <VideoOff size={16} /> : <Video size={16} />} 
            {isVideoOn ? 'Stop Camera' : 'Start Camera'}
          </button>
          
          <button onClick={toggleRecording} className={`flex items-center gap-2 border px-6 py-2.5 rounded-full text-sm font-bold ${isRecording ? 'bg-red-50 text-red-600' : 'bg-black text-samvaad-accentPrimary'}`}>
            {isRecording ? <MicOff size={16} /> : <Mic size={16} />} 
            {isRecording ? 'Stop Mic' : 'Start Mic'}
          </button>
        </div>
      </header>

      {/* MAIN CONTENT */}
      <main className="flex-1 flex max-w-7xl w-full mx-auto p-8 gap-8">
        
        {/* CAMERA PREVIEW */}
        <aside className="w-[450px] flex flex-col gap-6">
          <div className="bg-black text-white rounded-card shadow-glow p-6 relative overflow-hidden h-[340px] flex items-center justify-center">
             {!isVideoOn && <div className="text-center text-white/50"><Video size={48} className="mx-auto mb-4"/>Start camera for ISL recognition</div>}
             <video ref={videoRef} autoPlay playsInline muted className={`absolute inset-0 w-full h-full object-cover transform -scale-x-100 ${isVideoOn ? 'opacity-100' : 'opacity-0'}`} />
             <canvas ref={canvasRef} width="320" height="240" className="hidden" />
          </div>
          
          <div className="bg-white rounded-card border border-samvaad-border p-6 text-center">
             <h3 className="font-bold text-sm text-samvaad-textMuted uppercase tracking-wider mb-2">Sign Engine Status</h3>
             <div className="flex items-center justify-center gap-2">
                <span className={`relative inline-flex rounded-full h-3 w-3 ${signStatus === 'signing' ? 'bg-samvaad-accentPrimary animate-pulse-glow' : 'bg-gray-300'}`}></span>
                <span className="font-black text-xl text-black uppercase">{signStatus}</span>
             </div>
          </div>
        </aside>

        {/* CONVERSATION FEED */}
        <section className="flex-1 flex flex-col bg-samvaad-bgSecondary rounded-card border border-samvaad-border p-8 overflow-y-auto relative shadow-sm">
          <h2 className="font-display text-2xl font-black tracking-tighter text-black mb-6">Conversation</h2>
          
          <div className="flex flex-col gap-4">
            {transcripts.map((t, i) => (
              <div key={i} className={`flex ${t.source === 'sign' ? 'justify-end' : 'justify-start'}`}>
                <div className={`max-w-[80%] p-5 rounded-2xl ${t.source === 'sign' ? 'bg-black text-white rounded-tr-none shadow-glow' : 'bg-white border border-samvaad-border text-black rounded-tl-none shadow-sm'}`}>
                  <div className="flex items-center gap-2 mb-2 opacity-70">
                    {t.source === 'sign' ? <Hand size={14} /> : <Mic size={14} />}
                    <span className="text-xs font-bold uppercase tracking-wider">{t.source === 'sign' ? 'Sign Language' : 'Speech'}</span>
                  </div>
                  <p className="text-2xl font-bold">{t.text}</p>
                </div>
              </div>
            ))}
            <div ref={endOfMessagesRef} />
          </div>
        </section>
      </main>
    </div>
  );
}
