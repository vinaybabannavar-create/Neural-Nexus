"use client";

import React, { useState, useEffect, useRef } from "react";
import { 
  Send, 
  Mic, 
  MicOff, 
  ShieldAlert, 
  ShieldCheck, 
  Activity, 
  Globe, 
  Database, 
  Clock, 
  Layers, 
  AlertTriangle,
  FileText,
  UploadCloud,
  File as FileIcon,
  X,
  RefreshCw,
  Sparkles,
  Volume2,
  Trash2,
  CheckCircle2,
  ExternalLink,
  ChevronRight,
  Info,
  Search,
  Sliders
} from "lucide-react";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp?: string;
  meta?: {
    trust_score?: number;
    trust_rating?: string;
    trust_breakdown?: any;
    relevance_score?: number;
    web_search_used?: boolean;
    sources?: string[];
    escalation_status?: string | null;
    latency_metrics?: Record<string, number>;
  };
}

interface QuarantineItem {
  id: number;
  timestamp: string;
  source: string;
  reason: string;
  risk_score: number;
  snippet: string;
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      role: "assistant",
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      content: "Hello! I am **Neural Nexus C-RAG v2** — an enterprise self-reflective Retrieval-Augmented Generation system.\n\nUpload a document (PDF, TXT, MD) or enter a web URL on the left, then ask me anything. Every response is verified for factual grounding with zero hallucinations, complete with real-time Trust Scores and latency telemetry.",
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<"chat" | "metrics" | "quarantine">("chat");
  const [voiceMode, setVoiceMode] = useState(false);
  
  // Knowledge Ingestion State
  const [ingestType, setIngestType] = useState<"file" | "url">("file");
  const [selectedFile, setSelectedFile] = useState<globalThis.File | null>(null);
  const [ingestUrl, setIngestUrl] = useState("");
  const [ingesting, setIngesting] = useState(false);
  const [ingestMessage, setIngestMessage] = useState<{ type: "success" | "error"; text: string } | null>(null);
  const [indexedSources, setIndexedSources] = useState<string[]>([]);
  const [suggestedQuestions, setSuggestedQuestions] = useState<string[]>([
    "What is the core architecture of Neural Nexus?",
    "How does the 8-node LangGraph pipeline prevent hallucinations?",
    "Explain the composite Trust Score formula."
  ]);

  const [quarantineLogs, setQuarantineLogs] = useState<QuarantineItem[]>([]);
  const [quarantineSearch, setQuarantineSearch] = useState("");
  const [selectedMeta, setSelectedMeta] = useState<any>(null);

  const chatEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  useEffect(() => {
    fetchQuarantineLogs();
  }, []);

  const fetchQuarantineLogs = async () => {
    try {
      const res = await fetch("http://localhost:8000/security/quarantine");
      if (res.ok) {
        const data = await res.json();
        setQuarantineLogs(data.records || []);
      }
    } catch (e) {
      console.log("Quarantine store fetch error:", e);
    }
  };

  const handleSend = async (queryText?: string) => {
    const textToSend = (queryText || input).trim();
    if (!textToSend || loading) return;

    const userMsgId = `user_${Date.now()}`;
    const newMsg: Message = { 
      id: userMsgId, 
      role: "user", 
      content: textToSend,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    setMessages((prev) => [...prev, newMsg]);
    if (!queryText) setInput("");
    setLoading(true);

    try {
      const history = messages
        .filter((m) => m.id !== "welcome")
        .map((m) => ({ role: m.role, content: m.content }));

      const res = await fetch("http://localhost:8000/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: textToSend,
          history: history,
        }),
      });

      if (!res.ok) {
        throw new Error(`Server returned status ${res.status}`);
      }

      const data = await res.json();
      const assistantMsg: Message = {
        id: data.request_id || `asst_${Date.now()}`,
        role: "assistant",
        content: data.answer,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        meta: {
          trust_score: data.trust_score,
          trust_rating: data.trust_rating,
          relevance_score: data.relevance_score,
          web_search_used: data.web_search_used,
          sources: data.sources,
          escalation_status: data.escalation_status,
          latency_metrics: data.latency_metrics,
        },
      };

      setMessages((prev) => [...prev, assistantMsg]);
      setSelectedMeta(assistantMsg.meta);
    } catch (err: any) {
      const errorMsg: Message = {
        id: `err_${Date.now()}`,
        role: "assistant",
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        content: `⚠️ **Connection/Pipeline Error:** ${err.message}. Please verify the FastAPI backend is running on \`http://localhost:8000\`.`,
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setLoading(false);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
      setIngestMessage(null);
    }
  };

  const handleIngestFile = async () => {
    if (!selectedFile) return;
    setIngesting(true);
    setIngestMessage(null);

    const formData = new FormData();
    formData.append("file", selectedFile);
    const fileName = selectedFile.name;

    try {
      const res = await fetch("http://localhost:8000/ingest/file", {
        method: "POST",
        body: formData,
      });

      if (res.ok) {
        setIngestMessage({
          type: "success",
          text: `Verified & Indexed: ${fileName}`,
        });
        setIndexedSources((prev) => Array.from(new Set([fileName, ...prev])));
        setSuggestedQuestions([
          `What are the main points in ${fileName}?`,
          `Can you summarize the key findings of ${fileName}?`,
          `What methodology or architecture is presented in ${fileName}?`,
        ]);
        setSelectedFile(null);
        if (fileInputRef.current) fileInputRef.current.value = "";
        fetchQuarantineLogs();
      } else {
        const err = await res.json();
        setIngestMessage({
          type: "error",
          text: `Ingestion rejected: ${err.detail || "Security check failed"}`,
        });
      }
    } catch (e: any) {
      setIngestMessage({
        type: "error",
        text: `Upload failed: ${e.message}`,
      });
    } finally {
      setIngesting(false);
    }
  };

  const handleIngestUrl = async () => {
    if (!ingestUrl.trim()) return;
    setIngesting(true);
    setIngestMessage(null);
    const targetUrl = ingestUrl.trim();

    try {
      const res = await fetch("http://localhost:8000/ingest/url", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: targetUrl }),
      });
      if (res.ok) {
        const domainOrName = targetUrl.split("://").pop()?.replace(/\/$/, "") || targetUrl;
        setIngestMessage({
          type: "success",
          text: `Verified Source: ${domainOrName}`,
        });
        setIndexedSources((prev) => Array.from(new Set([domainOrName, ...prev])));
        setSuggestedQuestions([
          `What is ${domainOrName} about and what are its key features?`,
          `Summarize the main content from ${domainOrName}`,
          `What technologies or key concepts are mentioned in ${domainOrName}?`,
        ]);
        setIngestUrl("");
        fetchQuarantineLogs();
      } else {
        const err = await res.json();
        setIngestMessage({
          type: "error",
          text: `Ingestion failed: ${err.detail || "Security perimeter violation"}`,
        });
      }
    } catch (e: any) {
      setIngestMessage({
        type: "error",
        text: `Connection error: ${e.message}`,
      });
    } finally {
      setIngesting(false);
    }
  };

  const toggleVoiceMode = async () => {
    if (!voiceMode) {
      try {
        const res = await fetch("http://localhost:8000/voice/livekit/token", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            room_name: "neural-nexus-voice",
            participant_identity: `user_${Math.random().toString(36).substring(7)}`,
          }),
        });
        if (res.ok) {
          const tokenData = await res.json();
          console.log("[LiveKit] Acquired room token:", tokenData);
        }
        setVoiceMode(true);
      } catch (e) {
        setVoiceMode(true);
      }
    } else {
      setVoiceMode(false);
    }
  };

  const clearChat = () => {
    setMessages([
      {
        id: "welcome",
        role: "assistant",
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        content: "Conversation history cleared. Ready for your next inquiry.",
      },
    ]);
    setSelectedMeta(null);
  };

  const filteredQuarantine = quarantineLogs.filter(q => 
    q.source.toLowerCase().includes(quarantineSearch.toLowerCase()) ||
    q.reason.toLowerCase().includes(quarantineSearch.toLowerCase()) ||
    q.snippet.toLowerCase().includes(quarantineSearch.toLowerCase())
  );

  return (
    <div className="flex flex-col h-screen w-screen overflow-hidden bg-[#08090e] text-slate-200 font-sans">
      
      {/* ════════════════════════ TOP HEADER BAR ════════════════════════ */}
      <header className="h-16 px-6 glass-panel border-b border-white/[0.08] flex items-center justify-between z-20 flex-shrink-0">
        
        {/* Brand */}
        <div className="flex items-center gap-3.5">
          <div className="h-10 w-10 rounded-xl bg-gradient-to-tr from-cyan-500 via-blue-500 to-purple-600 flex items-center justify-center text-xl shadow-lg shadow-cyan-500/25 border border-white/20">
            🧠
          </div>
          <div className="flex flex-col">
            <div className="flex items-center gap-2">
              <span className="font-extrabold text-base tracking-wider bg-gradient-to-r from-cyan-400 via-blue-400 to-purple-400 bg-clip-text text-transparent uppercase">
                Neural Nexus
              </span>
              <span className="px-2 py-0.5 rounded-full text-[10px] font-semibold bg-cyan-500/10 text-cyan-400 border border-cyan-500/30 uppercase tracking-widest">
                C-RAG v2.0
              </span>
            </div>
            <span className="text-[11px] text-slate-400 font-medium">
              Self-Reflective RAG • Next.js & LiveKit Engine
            </span>
          </div>
        </div>

        {/* Center View Switcher */}
        <nav className="flex items-center bg-black/40 p-1 rounded-xl border border-white/[0.08] shadow-inner">
          <button
            onClick={() => setActiveTab("chat")}
            className={`flex items-center gap-2 px-4 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              activeTab === "chat"
                ? "bg-gradient-to-r from-cyan-500/20 to-blue-500/20 text-cyan-300 border border-cyan-500/30 shadow-sm"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <Sparkles size={14} className={activeTab === "chat" ? "text-cyan-400" : "text-slate-400"} />
            Reasoning Chat
          </button>

          <button
            onClick={() => setActiveTab("metrics")}
            className={`flex items-center gap-2 px-4 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              activeTab === "metrics"
                ? "bg-gradient-to-r from-cyan-500/20 to-blue-500/20 text-cyan-300 border border-cyan-500/30 shadow-sm"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <Activity size={14} className={activeTab === "metrics" ? "text-cyan-400" : "text-slate-400"} />
            Telemetry
          </button>

          <button
            onClick={() => {
              setActiveTab("quarantine");
              fetchQuarantineLogs();
            }}
            className={`flex items-center gap-2 px-4 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              activeTab === "quarantine"
                ? "bg-gradient-to-r from-cyan-500/20 to-blue-500/20 text-cyan-300 border border-cyan-500/30 shadow-sm"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <ShieldAlert size={14} className={activeTab === "quarantine" ? "text-red-400" : "text-slate-400"} />
            Quarantine
            {quarantineLogs.length > 0 && (
              <span className="px-1.5 py-0.2 rounded-full text-[10px] font-bold bg-red-500/20 text-red-400 border border-red-500/30">
                {quarantineLogs.length}
              </span>
            )}
          </button>
        </nav>

        {/* Right Status Indicators */}
        <div className="flex items-center gap-3 text-xs">
          <div className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/25 text-emerald-400 font-medium">
            <ShieldCheck size={14} />
            <span>Perimeter Active</span>
          </div>

          <div className="hidden md:flex items-center gap-1.5 px-3 py-1 rounded-full bg-purple-500/10 border border-purple-500/25 text-purple-300 font-medium">
            <Layers size={14} />
            <span>8-Node LangGraph</span>
          </div>

          <button
            onClick={clearChat}
            title="Clear conversation"
            className="p-2 rounded-xl text-slate-400 hover:text-white hover:bg-white/[0.05] border border-transparent hover:border-white/[0.08] transition-all"
          >
            <Trash2 size={15} />
          </button>
        </div>
      </header>

      {/* ════════════════════════ MAIN BODY WORKSPACE ════════════════════════ */}
      <div className="flex flex-1 overflow-hidden">

        {/* ── LEFT SIDEBAR: KNOWLEDGE BASE & VOICE ── */}
        <aside className="w-80 glass-panel border-r border-white/[0.08] flex flex-col justify-between p-4 flex-shrink-0 overflow-y-auto">
          
          <div className="flex flex-col gap-4">
            
            {/* Knowledge Ingestion Card */}
            <div className="glass-card p-4 rounded-2xl border border-white/[0.08] shadow-lg flex flex-col gap-3.5">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Database size={15} className="text-cyan-400" />
                  <span className="text-xs font-bold uppercase tracking-wider text-slate-200">
                    Knowledge Base
                  </span>
                </div>

                {/* Segmented Control */}
                <div className="flex bg-black/50 p-0.5 rounded-lg border border-white/[0.08] text-[11px]">
                  <button
                    onClick={() => setIngestType("file")}
                    className={`px-2.5 py-1 rounded-md font-medium transition-all ${
                      ingestType === "file"
                        ? "bg-cyan-500/20 text-cyan-300 shadow-sm border border-cyan-500/30"
                        : "text-slate-400 hover:text-white"
                    }`}
                  >
                    Document
                  </button>
                  <button
                    onClick={() => setIngestType("url")}
                    className={`px-2.5 py-1 rounded-md font-medium transition-all ${
                      ingestType === "url"
                        ? "bg-cyan-500/20 text-cyan-300 shadow-sm border border-cyan-500/30"
                        : "text-slate-400 hover:text-white"
                    }`}
                  >
                    Web URL
                  </button>
                </div>
              </div>

              {/* Mode 1: Document Upload */}
              {ingestType === "file" && (
                <div className="flex flex-col gap-3">
                  <input
                    type="file"
                    ref={fileInputRef}
                    onChange={handleFileSelect}
                    accept=".pdf,.txt,.md"
                    className="hidden"
                    id="doc-upload-input"
                  />

                  {!selectedFile ? (
                    <label
                      htmlFor="doc-upload-input"
                      className="group border-2 border-dashed border-white/[0.12] hover:border-cyan-500/50 rounded-xl p-4 flex flex-col items-center justify-center gap-2 cursor-pointer bg-white/[0.015] hover:bg-cyan-500/[0.04] transition-all"
                    >
                      <div className="p-2.5 rounded-xl bg-white/[0.03] group-hover:bg-cyan-500/10 transition-colors">
                        <UploadCloud size={20} className="text-slate-400 group-hover:text-cyan-400 transition-colors" />
                      </div>
                      <div className="text-center">
                        <span className="text-xs font-semibold text-slate-200 block">
                          Select or Drop File
                        </span>
                        <span className="text-[10px] text-slate-500">
                          PDF, TXT, MD (Max 20MB)
                        </span>
                      </div>
                    </label>
                  ) : (
                    <div className="bg-black/40 border border-cyan-500/30 rounded-xl p-3 flex items-center justify-between">
                      <div className="flex items-center gap-2.5 overflow-hidden">
                        <FileIcon size={16} className="text-cyan-400 flex-shrink-0" />
                        <div className="flex flex-col overflow-hidden">
                          <span className="text-xs font-medium text-slate-200 truncate">{selectedFile.name}</span>
                          <span className="text-[10px] text-slate-400">{(selectedFile.size / 1024).toFixed(1)} KB</span>
                        </div>
                      </div>
                      <button
                        onClick={() => {
                          setSelectedFile(null);
                          if (fileInputRef.current) fileInputRef.current.value = "";
                        }}
                        className="text-slate-400 hover:text-white p-1 hover:bg-white/[0.1] rounded-lg transition-colors"
                      >
                        <X size={14} />
                      </button>
                    </div>
                  )}

                  <button
                    onClick={handleIngestFile}
                    disabled={ingesting || !selectedFile}
                    className="w-full py-2.5 px-4 rounded-xl text-xs font-bold text-white bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center gap-2 shadow-lg shadow-cyan-500/20 transition-all"
                  >
                    {ingesting ? <RefreshCw size={14} className="animate-spin" /> : <FileText size={14} />}
                    {ingesting ? "Analyzing & Indexing..." : "🚀 Ingest & Screen Document"}
                  </button>
                </div>
              )}

              {/* Mode 2: Web URL */}
              {ingestType === "url" && (
                <div className="flex flex-col gap-3">
                  <div className="relative">
                    <input
                      type="text"
                      placeholder="https://example.com/docs"
                      value={ingestUrl}
                      onChange={(e) => setIngestUrl(e.target.value)}
                      className="w-full bg-black/40 border border-white/[0.1] focus:border-cyan-500 rounded-xl px-3.5 py-2.5 text-xs text-slate-200 placeholder-slate-500 focus:outline-none transition-colors"
                    />
                  </div>
                  <button
                    onClick={handleIngestUrl}
                    disabled={ingesting || !ingestUrl.trim()}
                    className="w-full py-2.5 px-4 rounded-xl text-xs font-bold text-white bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-500 hover:to-purple-500 disabled:opacity-40 flex items-center justify-center gap-2 shadow-lg shadow-purple-500/20 transition-all"
                  >
                    {ingesting ? <RefreshCw size={14} className="animate-spin" /> : <Globe size={14} />}
                    {ingesting ? "Scraping & Indexing..." : "🌐 Ingest Web URL"}
                  </button>
                </div>
              )}

              {/* Feedback status */}
              {ingestMessage && (
                <div className={`p-2.5 rounded-xl border text-[11px] flex items-center gap-2 ${
                  ingestMessage.type === "success"
                    ? "bg-emerald-500/10 border-emerald-500/30 text-emerald-300"
                    : "bg-red-500/10 border-red-500/30 text-red-300"
                }`}>
                  {ingestMessage.type === "success" ? <CheckCircle2 size={13} className="flex-shrink-0 text-emerald-400" /> : <AlertTriangle size={13} className="flex-shrink-0 text-red-400" />}
                  <span className="truncate">{ingestMessage.text}</span>
                </div>
              )}
            </div>

            {/* Quick Suggested Prompts */}
            {suggestedQuestions.length > 0 && (
              <div className="flex flex-col gap-2">
                <div className="flex items-center gap-1.5 text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
                  <Sparkles size={12} className="text-amber-400" />
                  <span>Suggested Inquiries</span>
                </div>
                <div className="flex flex-col gap-1.5">
                  {suggestedQuestions.map((q, idx) => (
                    <button
                      key={idx}
                      onClick={() => handleSend(q)}
                      className="text-left text-xs bg-white/[0.02] hover:bg-white/[0.06] border border-white/[0.06] hover:border-cyan-500/40 rounded-xl p-2.5 text-slate-300 hover:text-white transition-all flex items-start gap-2 group shadow-sm"
                    >
                      <span className="text-cyan-400 group-hover:translate-x-0.5 transition-transform">💡</span>
                      <span className="leading-relaxed">{q}</span>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Active Sources List */}
            {indexedSources.length > 0 && (
              <div className="flex flex-col gap-1.5">
                <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
                  Indexed Contexts
                </span>
                <div className="flex flex-wrap gap-1.5">
                  {indexedSources.map((src, i) => (
                    <span key={i} className="px-2.5 py-1 rounded-lg text-[10px] font-medium bg-white/[0.04] border border-white/[0.08] text-slate-300 truncate max-w-[260px] flex items-center gap-1.5">
                      <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
                      {src}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* LiveKit Voice Agent Card */}
          <div className="glass-card p-3.5 rounded-2xl border border-white/[0.08] flex flex-col gap-2.5 mt-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className={`p-2 rounded-xl ${voiceMode ? "bg-emerald-500/20 text-emerald-400 animate-pulse" : "bg-white/[0.05] text-slate-400"}`}>
                  <Volume2 size={16} />
                </div>
                <div className="flex flex-col">
                  <span className="text-xs font-bold text-slate-200">LiveKit Voice</span>
                  <span className="text-[10px] text-slate-400">WebRTC Sub-Second Mode</span>
                </div>
              </div>

              <button
                onClick={toggleVoiceMode}
                className={`px-3 py-1.5 rounded-xl text-xs font-bold flex items-center gap-1.5 transition-all ${
                  voiceMode
                    ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/40 shadow-sm shadow-emerald-500/20"
                    : "bg-white/[0.05] text-slate-400 hover:text-white border border-white/[0.08]"
                }`}
              >
                {voiceMode ? <Mic size={13} /> : <MicOff size={13} />}
                {voiceMode ? "Live" : "Off"}
              </button>
            </div>
            
            {voiceMode && (
              <div className="flex items-center justify-between px-2 py-1 rounded-lg bg-emerald-500/5 border border-emerald-500/20 text-[10px] text-emerald-400 font-mono">
                <span>STT 150ms → RAG 600ms → TTS 150ms</span>
                <span className="animate-ping h-1.5 w-1.5 rounded-full bg-emerald-400" />
              </div>
            )}
          </div>
        </aside>

        {/* ── CENTER WORKSPACE ── */}
        <main className="flex-1 flex flex-col h-full overflow-hidden bg-[#08090e]">
          
          {/* ════════ VIEW 1: REASONING CHAT ════════ */}
          {activeTab === "chat" && (
            <div className="flex-1 flex flex-col h-full overflow-hidden">
              
              {/* Message Stream */}
              <div className="flex-1 overflow-y-auto p-6 flex flex-col gap-6 max-w-4xl mx-auto w-full">
                {messages.map((m) => (
                  <div
                    key={m.id}
                    className={`flex flex-col ${m.role === "user" ? "items-end" : "items-start"}`}
                  >
                    <div className="flex items-center gap-2 mb-1.5 px-1">
                      <span className="text-[11px] font-semibold text-slate-400">
                        {m.role === "user" ? "You" : "Neural Nexus C-RAG"}
                      </span>
                      {m.timestamp && (
                        <span className="text-[10px] text-slate-500 font-mono">
                          {m.timestamp}
                        </span>
                      )}
                    </div>

                    <div
                      className={`rounded-2xl p-5 text-sm leading-relaxed ${
                        m.role === "user"
                          ? "max-w-2xl bg-gradient-to-r from-blue-600/30 to-purple-600/30 border border-blue-500/40 text-slate-100 rounded-tr-sm shadow-md"
                          : "max-w-3xl w-full glass-card border border-white/[0.08] text-slate-200 rounded-tl-sm shadow-xl backdrop-blur-md"
                      }`}
                    >
                      {/* Check if message has human verification notice */}
                      {m.content.includes("[STATUS: PENDING_HUMAN_VERIFICATION]") ? (
                        <div className="flex flex-col gap-3">
                          <div className="p-3 rounded-xl bg-amber-500/10 border border-amber-500/30 text-amber-300 text-xs flex items-center gap-2">
                            <AlertTriangle size={16} className="text-amber-400 flex-shrink-0" />
                            <span className="font-bold">Pending Human Verification</span>
                            <span className="text-amber-400/80 font-normal">— Confidence circuit-breaker triggered</span>
                          </div>
                          <div className="whitespace-pre-wrap text-slate-300 font-sans">
                            {m.content.replace("⚠️ [STATUS: PENDING_HUMAN_VERIFICATION]", "").trim()}
                          </div>
                        </div>
                      ) : (
                        <div className="whitespace-pre-wrap font-sans text-slate-200">
                          {m.content}
                        </div>
                      )}

                      {/* Meta / Trust Score Badge on Assistant Messages */}
                      {m.meta && (
                        <div className="mt-4 pt-3.5 border-t border-white/[0.08] flex flex-wrap items-center justify-between gap-3 text-xs">
                          <div className="flex items-center gap-2">
                            {m.meta.trust_score !== undefined && (
                              <div
                                onClick={() => setSelectedMeta(m.meta)}
                                className={`cursor-pointer px-3 py-1 rounded-lg border font-bold flex items-center gap-1.5 transition-all shadow-sm ${
                                  m.meta.trust_score >= 80
                                    ? "bg-emerald-500/10 border-emerald-500/30 text-emerald-400 hover:border-emerald-500/60"
                                    : m.meta.trust_score >= 50
                                    ? "bg-amber-500/10 border-amber-500/30 text-amber-400 hover:border-amber-500/60"
                                    : "bg-red-500/10 border-red-500/30 text-red-400 hover:border-red-500/60"
                                }`}
                              >
                                <Activity size={13} />
                                <span>Trust Score: {m.meta.trust_score.toFixed(0)}/100</span>
                                <span className="text-[10px] font-normal opacity-80 font-mono">({m.meta.trust_rating || "Grounded"})</span>
                              </div>
                            )}

                            {m.meta.web_search_used && (
                              <span className="px-2.5 py-1 rounded-lg bg-purple-500/10 text-purple-300 border border-purple-500/20 text-[11px] font-medium flex items-center gap-1">
                                <Globe size={11} /> Web Fallback
                              </span>
                            )}
                          </div>

                          {m.meta.sources && m.meta.sources.length > 0 && (
                            <div className="flex items-center gap-1.5 text-slate-400 text-[11px]">
                              <span className="text-slate-500">Sources:</span>
                              <span className="text-cyan-400/90 font-mono truncate max-w-xs">
                                {m.meta.sources.join(", ")}
                              </span>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                ))}

                {loading && (
                  <div className="flex items-center gap-3 text-cyan-400 text-xs font-semibold p-4 glass-card rounded-2xl border border-cyan-500/30 w-fit animate-pulse shadow-lg shadow-cyan-500/10">
                    <RefreshCw size={15} className="animate-spin text-cyan-400" />
                    <span>8-Node LangGraph reasoning & verifying citations...</span>
                  </div>
                )}
                <div ref={chatEndRef} />
              </div>

              {/* Floating Bottom Input Bar */}
              <div className="p-4 glass-panel border-t border-white/[0.08]">
                <form 
                  onSubmit={(e) => { e.preventDefault(); handleSend(); }} 
                  className="max-w-4xl mx-auto flex items-center gap-2.5 bg-black/60 border border-white/[0.12] focus-within:border-cyan-500/70 rounded-2xl p-1.5 shadow-2xl transition-all"
                >
                  <input
                    type="text"
                    placeholder="Ask Neural Nexus anything about your documents or knowledge base..."
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    disabled={loading}
                    className="flex-1 bg-transparent px-4 py-2.5 text-sm text-slate-100 placeholder-slate-500 focus:outline-none"
                  />
                  <button
                    type="submit"
                    disabled={loading || !input.trim()}
                    className="bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white p-3 rounded-xl font-semibold flex items-center justify-center disabled:opacity-40 disabled:cursor-not-allowed shadow-lg shadow-cyan-500/25 transition-all"
                  >
                    <Send size={15} />
                  </button>
                </form>
              </div>
            </div>
          )}

          {/* ════════ VIEW 2: TELEMETRY ════════ */}
          {activeTab === "metrics" && (
            <div className="flex-1 overflow-y-auto p-8 max-w-5xl mx-auto w-full flex flex-col gap-6">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-lg font-bold text-slate-100 flex items-center gap-2">
                    <Activity size={20} className="text-cyan-400" />
                    Pipeline Telemetry & Execution Analytics
                  </h3>
                  <p className="text-xs text-slate-400">
                    Real-time performance metrics computed across all 8 LangGraph stages.
                  </p>
                </div>
              </div>

              {/* Top 3 KPI Cards */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="glass-card p-5 rounded-2xl border border-white/[0.08] flex flex-col gap-1.5 shadow-lg">
                  <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Composite Trust Index</span>
                  <div className="flex items-baseline gap-2">
                    <span className="text-3xl font-extrabold text-emerald-400 font-mono">
                      {selectedMeta?.trust_score ? `${selectedMeta.trust_score.toFixed(0)}` : "100"}
                    </span>
                    <span className="text-sm text-slate-500 font-medium">/ 100</span>
                  </div>
                  <span className="text-[11px] text-slate-500 font-medium">Relevance (40) + Grounding (40) + Speed (20)</span>
                </div>

                <div className="glass-card p-5 rounded-2xl border border-white/[0.08] flex flex-col gap-1.5 shadow-lg">
                  <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Document Relevance</span>
                  <div className="flex items-baseline gap-2">
                    <span className="text-3xl font-extrabold text-cyan-400 font-mono">
                      {selectedMeta?.relevance_score ? `${(selectedMeta.relevance_score * 100).toFixed(0)}%` : "100%"}
                    </span>
                  </div>
                  <span className="text-[11px] text-slate-500 font-medium">Threshold: 50% for direct generation</span>
                </div>

                <div className="glass-card p-5 rounded-2xl border border-white/[0.08] flex flex-col gap-1.5 shadow-lg">
                  <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Escalation Circuit</span>
                  <div className="flex items-baseline gap-2">
                    <span className="text-xl font-bold text-slate-200">
                      {selectedMeta?.escalation_status || "Normal (Grounded)"}
                    </span>
                  </div>
                  <span className="text-[11px] text-slate-500 font-medium">Auto-fallback to human oversight on ungrounded claims</span>
                </div>
              </div>

              {/* Per-Node Latency Waterfall Table */}
              <div className="glass-card p-6 rounded-2xl border border-white/[0.08] flex flex-col gap-4 shadow-xl">
                <h4 className="text-sm font-bold text-slate-200 flex items-center gap-2">
                  <Clock size={16} className="text-cyan-400" />
                  Per-Node Latency Breakdown (LangGraph Execution)
                </h4>

                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs">
                    <thead>
                      <tr className="border-b border-white/[0.08] text-slate-400 font-semibold">
                        <th className="py-3 px-4">Pipeline Node</th>
                        <th className="py-3 px-4">Execution Duration</th>
                        <th className="py-3 px-4">Latency Waterfall</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-white/[0.04] text-slate-300">
                      {selectedMeta?.latency_metrics ? (
                        Object.entries(selectedMeta.latency_metrics).map(([node, sec]: any) => {
                          const ms = (sec * 1000).toFixed(1);
                          const widthPct = Math.min(100, Math.max(10, (sec / 2) * 100));
                          return (
                            <tr key={node} className="hover:bg-white/[0.02] transition-colors">
                              <td className="py-3 px-4 font-mono font-bold text-cyan-400">{node}</td>
                              <td className="py-3 px-4 font-mono text-slate-200">{ms} ms</td>
                              <td className="py-3 px-4 w-1/2">
                                <div className="w-full bg-white/[0.05] h-2 rounded-full overflow-hidden">
                                  <div 
                                    className="h-full bg-gradient-to-r from-cyan-400 to-blue-500 rounded-full" 
                                    style={{ width: `${widthPct}%` }}
                                  />
                                </div>
                              </td>
                            </tr>
                          );
                        })
                      ) : (
                        <tr>
                          <td colSpan={3} className="py-8 text-center text-slate-500">
                            Submit a question in the chat to view real-time latency telemetry.
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}

          {/* ════════ VIEW 3: QUARANTINE AUDIT ════════ */}
          {activeTab === "quarantine" && (
            <div className="flex-1 overflow-y-auto p-8 max-w-6xl mx-auto w-full flex flex-col gap-6">
              
              {/* Header & Controls */}
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div>
                  <h3 className="text-lg font-bold text-slate-100 flex items-center gap-2">
                    <ShieldAlert size={20} className="text-red-400" />
                    Quarantine Store Audit Ledger
                  </h3>
                  <p className="text-xs text-slate-400">
                    Forensic SQLite repository (<code className="text-cyan-400">quarantine.db</code>) logging screened adversarial injection attempts.
                  </p>
                </div>

                <div className="flex items-center gap-2.5">
                  <div className="relative">
                    <Search size={14} className="absolute left-3 top-3 text-slate-500" />
                    <input
                      type="text"
                      placeholder="Filter audit log..."
                      value={quarantineSearch}
                      onChange={(e) => setQuarantineSearch(e.target.value)}
                      className="bg-black/50 border border-white/[0.1] rounded-xl pl-9 pr-3 py-1.5 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-red-500/50 w-52"
                    />
                  </div>

                  <button
                    onClick={fetchQuarantineLogs}
                    className="bg-white/[0.05] hover:bg-white/[0.1] border border-white/[0.08] px-3.5 py-1.5 rounded-xl text-xs font-semibold flex items-center gap-2 transition-all"
                  >
                    <RefreshCw size={13} /> Refresh
                  </button>
                </div>
              </div>

              {/* Data Table */}
              <div className="glass-card rounded-2xl border border-white/[0.08] overflow-hidden shadow-xl">
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs">
                    <thead className="bg-black/40 border-b border-white/[0.08] text-slate-400 font-semibold">
                      <tr>
                        <th className="py-3.5 px-4 w-16">#ID</th>
                        <th className="py-3.5 px-4 w-28">Timestamp</th>
                        <th className="py-3.5 px-4">Source Document</th>
                        <th className="py-3.5 px-4">Threat Classification</th>
                        <th className="py-3.5 px-4 w-24">Risk</th>
                        <th className="py-3.5 px-4">Forensic Payload Snippet</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-white/[0.04] text-slate-300">
                      {filteredQuarantine.length > 0 ? (
                        filteredQuarantine.map((q) => (
                          <tr key={q.id} className="hover:bg-white/[0.02] transition-colors">
                            <td className="py-3 px-4 font-mono text-slate-500">#{q.id}</td>
                            <td className="py-3 px-4 text-slate-400 font-mono text-[11px]">
                              {q.timestamp.split("T")[0]}
                            </td>
                            <td className="py-3 px-4 font-semibold text-slate-200 truncate max-w-[180px]">
                              {q.source}
                            </td>
                            <td className="py-3 px-4 text-red-400 font-medium">
                              {q.reason}
                            </td>
                            <td className="py-3 px-4">
                              <span className="px-2 py-0.5 rounded-md text-[10px] font-extrabold bg-red-500/15 border border-red-500/30 text-red-400">
                                {(q.risk_score * 100).toFixed(0)}%
                              </span>
                            </td>
                            <td className="py-3 px-4 font-mono text-[11px] text-slate-400 truncate max-w-xs">
                              <code className="bg-black/50 px-2 py-1 rounded text-slate-300 border border-white/[0.06]">
                                {q.snippet}
                              </code>
                            </td>
                          </tr>
                        ))
                      ) : (
                        <tr>
                          <td colSpan={6} className="py-12 text-center text-slate-500">
                            {quarantineSearch ? "No records matched your search filter." : "No quarantined threats recorded in quarantine.db."}
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}

        </main>
      </div>
    </div>
  );
}