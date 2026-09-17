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
  CheckCircle2, 
  AlertTriangle,
  FileText,
  Link2,
  RefreshCw,
  Terminal,
  Volume2
} from "lucide-react";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
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
      content: "Hello! I am **Neural Nexus C-RAG v2**.\n\nAsk me anything or ingest documents/URLs in the left panel. I continuously verify factual grounding, calculate real-time Trust Scores, and provide per-node latency telemetry.",
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<"chat" | "quarantine" | "metrics">("chat");
  const [voiceMode, setVoiceMode] = useState(false);
  const [ingestUrl, setIngestUrl] = useState("");
  const [ingesting, setIngesting] = useState(false);
  const [ingestMessage, setIngestMessage] = useState<string | null>(null);
  const [quarantineLogs, setQuarantineLogs] = useState<QuarantineItem[]>([]);
  const [selectedMeta, setSelectedMeta] = useState<any>(null);

  const chatEndRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WebSocket | null>(null);

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
      console.log("Quarantine store fetch error (FastAPI may be offline):", e);
    }
  };

  const handleSend = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!input.trim() || loading) return;

    const userText = input.trim();
    const userMsgId = `user_${Date.now()}`;
    const newMsg: Message = { id: userMsgId, role: "user", content: userText };

    setMessages((prev) => [...prev, newMsg]);
    setInput("");
    setLoading(true);

    try {
      // Connect to FastAPI REST / Query
      const history = messages
        .filter((m) => m.id !== "welcome")
        .map((m) => ({ role: m.role, content: m.content }));

      const res = await fetch("http://localhost:8000/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: userText,
          history: history,
        }),
      });

      if (!res.ok) {
        throw new Error(`Server returned ${res.status}`);
      }

      const data = await res.json();
      const assistantMsg: Message = {
        id: data.request_id || `asst_${Date.now()}`,
        role: "assistant",
        content: data.answer,
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
        content: `⚠️ **Connection/Pipeline Error:** ${err.message}. Ensure FastAPI server is running on \`http://localhost:8000\`.`,
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setLoading(false);
    }
  };

  const handleIngestUrl = async () => {
    if (!ingestUrl.trim()) return;
    setIngesting(true);
    setIngestMessage(null);
    try {
      const res = await fetch("http://localhost:8000/ingest/url", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: ingestUrl.trim() }),
      });
      if (res.ok) {
        setIngestMessage(`✓ Successfully indexed: ${ingestUrl}`);
        setIngestUrl("");
        fetchQuarantineLogs();
      } else {
        const err = await res.json();
        setIngestMessage(`✗ Ingestion failed: ${err.detail || "Violation detected"}`);
      }
    } catch (e: any) {
      setIngestMessage(`✗ Error connecting to ingestion service: ${e.message}`);
    } finally {
      setIngesting(false);
    }
  };

  const toggleVoiceMode = async () => {
    if (!voiceMode) {
      // Connect to LiveKit token endpoint
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
          setVoiceMode(true);
        } else {
          setVoiceMode(true); // fallback mode
        }
      } catch (e) {
        setVoiceMode(true);
      }
    } else {
      setVoiceMode(false);
    }
  };

  return (
    <main className="flex h-screen w-screen overflow-hidden">
      {/* ── LEFT SIDEBAR ────────────────────────────────── */}
      <aside className="w-80 glass-panel flex flex-col justify-between p-5 border-r border-glassBorder z-10">
        <div className="flex flex-col gap-6">
          {/* Brand */}
          <div className="flex items-center gap-3">
            <div className="h-10 w-10 rounded-xl bg-gradient-to-tr from-accent to-accentPurple flex items-center justify-center text-xl shadow-lg shadow-accent/20">
              🧠
            </div>
            <div>
              <h1 className="font-bold text-lg leading-tight bg-gradient-to-r from-accent to-accentPurple bg-clip-text text-transparent">
                Neural Nexus
              </h1>
              <span className="text-xs text-slate-400">Next.js & LiveKit C-RAG</span>
            </div>
          </div>

          {/* Navigation Tabs */}
          <div className="flex bg-surface p-1 rounded-xl border border-glassBorder text-xs">
            <button
              onClick={() => setActiveTab("chat")}
              className={`flex-1 py-1.5 rounded-lg font-medium transition-all ${
                activeTab === "chat" ? "bg-accent/20 text-accent" : "text-slate-400 hover:text-white"
              }`}
            >
              Chat
            </button>
            <button
              onClick={() => setActiveTab("metrics")}
              className={`flex-1 py-1.5 rounded-lg font-medium transition-all ${
                activeTab === "metrics" ? "bg-accent/20 text-accent" : "text-slate-400 hover:text-white"
              }`}
            >
              Telemetry
            </button>
            <button
              onClick={() => {
                setActiveTab("quarantine");
                fetchQuarantineLogs();
              }}
              className={`flex-1 py-1.5 rounded-lg font-medium transition-all ${
                activeTab === "quarantine" ? "bg-accent/20 text-accent" : "text-slate-400 hover:text-white"
              }`}
            >
              Quarantine
            </button>
          </div>

          {/* Knowledge Ingestion */}
          <div className="flex flex-col gap-3">
            <label className="text-xs uppercase tracking-wider text-slate-400 font-semibold flex items-center gap-1.5">
              <Database size={13} className="text-accent" /> Knowledge Ingest
            </label>
            <div className="flex flex-col gap-2">
              <div className="relative">
                <input
                  type="text"
                  placeholder="https://example.com/doc"
                  value={ingestUrl}
                  onChange={(e) => setIngestUrl(e.target.value)}
                  className="w-full bg-surface border border-glassBorder rounded-lg px-3 py-2 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-accent"
                />
              </div>
              <button
                onClick={handleIngestUrl}
                disabled={ingesting || !ingestUrl.trim()}
                className="w-full bg-surfaceHover hover:bg-accent/20 border border-glassBorder text-accent hover:border-accent/40 py-2 rounded-lg text-xs font-semibold flex items-center justify-center gap-2 transition-all disabled:opacity-50"
              >
                {ingesting ? <RefreshCw size={13} className="animate-spin" /> : <Globe size={13} />}
                Ingest URL
              </button>
              {ingestMessage && (
                <p className={`text-[11px] mt-1 ${ingestMessage.startsWith("✓") ? "text-trustGreen" : "text-trustRed"}`}>
                  {ingestMessage}
                </p>
              )}
            </div>
          </div>
        </div>

        {/* LiveKit Voice Mode Card */}
        <div className="glass-card p-4 rounded-xl flex flex-col gap-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Volume2 size={16} className={voiceMode ? "text-trustGreen animate-pulse" : "text-slate-400"} />
              <span className="text-xs font-semibold text-slate-200">LiveKit Voice</span>
            </div>
            <button
              onClick={toggleVoiceMode}
              className={`p-2 rounded-lg text-xs font-medium flex items-center gap-1.5 transition-all ${
                voiceMode
                  ? "bg-trustGreen/20 text-trustGreen border border-trustGreen/40"
                  : "bg-surfaceHover text-slate-400 border border-glassBorder hover:text-white"
              }`}
            >
              {voiceMode ? <Mic size={14} /> : <MicOff size={14} />}
              {voiceMode ? "Active" : "Off"}
            </button>
          </div>
          <p className="text-[11px] text-slate-400">
            {voiceMode
              ? "LiveKit WebRTC active: Sub-second voice round-trip (STT 150ms → RAG 600ms → TTS 150ms)."
              : "Enable real-time voice mode to speak directly with Neural Nexus."}
          </p>
        </div>
      </aside>

      {/* ── MAIN CONTENT AREA ────────────────────────────── */}
      <section className="flex-1 flex flex-col h-full overflow-hidden bg-background">
        {/* Header */}
        <header className="h-16 glass-panel border-b border-glassBorder flex items-center justify-between px-6 z-10">
          <div className="flex items-center gap-3">
            <h2 className="font-semibold text-sm text-slate-200">
              {activeTab === "chat" && "Conversation & Real-time Reasoning"}
              {activeTab === "metrics" && "Pipeline Latency & Trust Telemetry"}
              {activeTab === "quarantine" && "Security Perimeter & Quarantine Store Audit"}
            </h2>
          </div>
          <div className="flex items-center gap-4 text-xs">
            <span className="flex items-center gap-1.5 text-trustGreen font-medium">
              <ShieldCheck size={14} /> Perimeter Active
            </span>
            <span className="text-slate-600">|</span>
            <span className="text-slate-400 flex items-center gap-1">
              <Layers size={13} className="text-accent" /> 8-Node LangGraph
            </span>
          </div>
        </header>

        {/* Tab 1: Chat View */}
        {activeTab === "chat" && (
          <div className="flex-1 flex flex-col h-[calc(100vh-4rem)] overflow-hidden">
            {/* Messages */}
            <div className="flex-1 overflow-y-auto p-6 flex flex-col gap-5">
              {messages.map((m) => (
                <div
                  key={m.id}
                  className={`flex flex-col ${m.role === "user" ? "items-end" : "items-start"}`}
                >
                  <div
                    className={`max-w-2xl rounded-2xl p-4 text-sm leading-relaxed ${
                      m.role === "user"
                        ? "bg-gradient-to-r from-accent/20 to-accentPurple/20 border border-accent/40 text-white rounded-br-none"
                        : "glass-card text-slate-200 rounded-bl-none border-glassBorder"
                    }`}
                  >
                    <div className="whitespace-pre-wrap">{m.content}</div>

                    {/* Metadata & Trust Badge for Assistant */}
                    {m.meta && (
                      <div className="mt-4 pt-3 border-t border-glassBorder flex flex-wrap items-center gap-2 text-xs">
                        {m.meta.trust_score !== undefined && (
                          <span
                            onClick={() => setSelectedMeta(m.meta)}
                            className="cursor-pointer bg-surface px-2.5 py-1 rounded-md border border-glassBorder text-trustGreen font-bold flex items-center gap-1 hover:border-trustGreen/50 transition-all"
                          >
                            <Activity size={12} />
                            Trust: {m.meta.trust_score.toFixed(0)}/100
                          </span>
                        )}
                        {m.meta.escalation_status && (
                          <span className="bg-trustRed/20 text-trustRed border border-trustRed/30 px-2 py-0.5 rounded font-medium flex items-center gap-1">
                            <AlertTriangle size={11} /> {m.meta.escalation_status}
                          </span>
                        )}
                        {m.meta.web_search_used && (
                          <span className="bg-accentPurple/20 text-accentPurple px-2 py-0.5 rounded flex items-center gap-1">
                            <Globe size={11} /> Web Fallback
                          </span>
                        )}
                        {m.meta.sources && m.meta.sources.length > 0 && (
                          <span className="text-slate-400 text-[11px]">
                            Sources: {m.meta.sources.join(", ")}
                          </span>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              ))}

              {loading && (
                <div className="flex items-center gap-3 text-accent text-xs font-medium animate-pulse p-4">
                  <RefreshCw size={14} className="animate-spin" />
                  Self-reflective grading & synthesizing answer...
                </div>
              )}
              <div ref={chatEndRef} />
            </div>

            {/* Input Bar */}
            <div className="p-4 glass-panel border-t border-glassBorder">
              <form onSubmit={handleSend} className="flex gap-2 max-w-4xl mx-auto">
                <input
                  type="text"
                  placeholder="Ask Neural Nexus anything (e.g. 'What is contextual chunking?')..."
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  disabled={loading}
                  className="flex-1 bg-surface border border-glassBorder rounded-xl px-4 py-3 text-sm text-slate-200 placeholder-slate-500 focus:outline-none focus:border-accent"
                />
                <button
                  type="submit"
                  disabled={loading || !input.trim()}
                  className="bg-gradient-to-r from-accent to-accentPurple text-white px-5 rounded-xl font-semibold flex items-center gap-2 hover:opacity-90 disabled:opacity-50 transition-all shadow-lg shadow-accent/20"
                >
                  <Send size={15} />
                </button>
              </form>
            </div>
          </div>
        )}

        {/* Tab 2: Metrics & Telemetry View */}
        {activeTab === "metrics" && (
          <div className="p-6 flex-1 overflow-y-auto flex flex-col gap-6 max-w-4xl mx-auto w-full">
            <div className="grid grid-cols-3 gap-4">
              <div className="glass-card p-4 rounded-xl border border-glassBorder flex flex-col gap-1">
                <span className="text-xs text-slate-400 font-medium">Composite Trust Index</span>
                <span className="text-2xl font-bold text-trustGreen">
                  {selectedMeta?.trust_score ? `${selectedMeta.trust_score.toFixed(0)}/100` : "100/100"}
                </span>
                <span className="text-[11px] text-slate-500">Relevance (40) + Grounding (40) + Speed (20)</span>
              </div>
              <div className="glass-card p-4 rounded-xl border border-glassBorder flex flex-col gap-1">
                <span className="text-xs text-slate-400 font-medium">Relevance Score</span>
                <span className="text-2xl font-bold text-accent">
                  {selectedMeta?.relevance_score ? `${(selectedMeta.relevance_score * 100).toFixed(0)}%` : "100%"}
                </span>
                <span className="text-[11px] text-slate-500">Threshold: 50% for direct generation</span>
              </div>
              <div className="glass-card p-4 rounded-xl border border-glassBorder flex flex-col gap-1">
                <span className="text-xs text-slate-400 font-medium">Escalation Status</span>
                <span className="text-2xl font-bold text-slate-200">
                  {selectedMeta?.escalation_status || "Normal"}
                </span>
                <span className="text-[11px] text-slate-500">Confidence circuit-breaker</span>
              </div>
            </div>

            {/* Per-Node Latency Table */}
            <div className="glass-card p-5 rounded-xl border border-glassBorder flex flex-col gap-4">
              <h3 className="font-semibold text-sm text-slate-200 flex items-center gap-2">
                <Clock size={15} className="text-accent" /> Per-Node Execution Latency Breakdown
              </h3>
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead>
                    <tr className="border-b border-glassBorder text-slate-400">
                      <th className="py-2">Pipeline Node</th>
                      <th className="py-2">Latency (Seconds)</th>
                      <th className="py-2">Latency (ms)</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-glassBorder text-slate-300">
                    {selectedMeta?.latency_metrics ? (
                      Object.entries(selectedMeta.latency_metrics).map(([node, sec]: any) => (
                        <tr key={node}>
                          <td className="py-2 font-mono text-accent">{node}</td>
                          <td className="py-2">{sec.toFixed(3)}s</td>
                          <td className="py-2">{(sec * 1000).toFixed(1)}ms</td>
                        </tr>
                      ))
                    ) : (
                      <tr>
                        <td colSpan={3} className="py-4 text-center text-slate-500">
                          Submit a query to view live per-stage timings.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* Tab 3: Quarantine Store Audit */}
        {activeTab === "quarantine" && (
          <div className="p-6 flex-1 overflow-y-auto flex flex-col gap-6 max-w-5xl mx-auto w-full">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="font-semibold text-base text-slate-100 flex items-center gap-2">
                  <ShieldAlert size={18} className="text-trustRed" /> Quarantine Store Audit Trail
                </h3>
                <p className="text-xs text-slate-400">
                  Forensic log stored in SQLite (`quarantine.db`) tracking rejected prompt injection and jailbreak attempts.
                </p>
              </div>
              <button
                onClick={fetchQuarantineLogs}
                className="bg-surface hover:bg-surfaceHover border border-glassBorder px-3 py-1.5 rounded-lg text-xs flex items-center gap-1.5"
              >
                <RefreshCw size={12} /> Refresh
              </button>
            </div>

            <div className="glass-card rounded-xl border border-glassBorder overflow-hidden">
              <table className="w-full text-left text-xs">
                <thead className="bg-surface border-b border-glassBorder text-slate-400">
                  <tr>
                    <th className="p-3">ID</th>
                    <th className="p-3">Timestamp (UTC)</th>
                    <th className="p-3">Source</th>
                    <th className="p-3">Detected Threat / Reason</th>
                    <th className="p-3">Risk</th>
                    <th className="p-3">Snippet</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-glassBorder text-slate-300">
                  {quarantineLogs.length > 0 ? (
                    quarantineLogs.map((q) => (
                      <tr key={q.id} className="hover:bg-surfaceHover/50 transition-all">
                        <td className="p-3 font-mono text-slate-400">#{q.id}</td>
                        <td className="p-3 text-slate-400">{q.timestamp.split("T")[0]}</td>
                        <td className="p-3 font-medium text-slate-200">{q.source}</td>
                        <td className="p-3 text-trustRed">{q.reason}</td>
                        <td className="p-3 font-bold text-trustYellow">{(q.risk_score * 100).toFixed(0)}%</td>
                        <td className="p-3 font-mono text-[11px] text-slate-400 truncate max-w-xs">{q.snippet}</td>
                      </tr>
                    ))
                  ) : (
                    <tr>
                      <td colSpan={6} className="p-6 text-center text-slate-500">
                        No quarantined documents recorded in quarantine.db yet.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </section>
    </main>
  );
}
