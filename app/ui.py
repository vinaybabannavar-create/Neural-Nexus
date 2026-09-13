"""
ui.py — Refined Neural Nexus UI with Glassmorphism & Starting Animation.
"""
import streamlit as st
from pathlib import Path
import tempfile, shutil, base64
from langchain_core.messages import HumanMessage

st.set_page_config(page_title="Neural Nexus", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

def get_image_base64(path):
    try:
        with open(path, "rb") as f: return base64.b64encode(f.read()).decode()
    except: return ""

# Logo lookup
logo_path = list(Path("C:/Users/LENOVO/.gemini/antigravity/brain/5e17c66a-3682-440f-985b-4ca23a46117e/").glob("neural_nexus_logo_*.png"))
if not logo_path and Path("Neural Nexus.png").exists():
    logo_path = [Path("Neural Nexus.png")]
logo_b64 = get_image_base64(logo_path[0]) if logo_path else ""
logo_img = f'<img src="data:image/png;base64,{logo_b64}" width="140" style="border-radius:15px; margin-bottom: 20px; box-shadow: 0 8px 32px rgba(0,0,0,0.3);">' if logo_b64 else '<h1>🧠</h1>'

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

/* --- GLOBAL FONT & BACKGROUND --- */
html, body, [class*="css"] {{
    font-family: 'Outfit', sans-serif !important;
}}

.stApp {{
    background: #090910;
    background-image: 
        radial-gradient(circle at 15% 50%, rgba(79, 172, 254, 0.08), transparent 40%),
        radial-gradient(circle at 85% 30%, rgba(167, 139, 250, 0.08), transparent 40%);
    color: #e2eaf5;
}}

/* --- STARTING ANIMATION OVERLAY --- */
.starting-overlay {{
    position: fixed;
    top: 0; left: 0; width: 100vw; height: 100vh;
    background: #090910;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    z-index: 999999;
    animation: fadeOutOverlay 2.5s cubic-bezier(0.8, 0, 0.2, 1) forwards;
    pointer-events: none;
}}

.starting-logo {{
    font-size: 4rem;
    background: linear-gradient(90deg, #4facfe, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    animation: scalePulse 1.5s ease-in-out infinite alternate;
}}

.starting-sub {{
    color: #8fa3b0;
    font-size: 1.2rem;
    margin-top: 10px;
    letter-spacing: 4px;
    text-transform: uppercase;
    animation: slideUpFade 1s ease-out forwards;
}}

@keyframes scalePulse {{
    0% {{ transform: scale(0.95); opacity: 0.8; }}
    100% {{ transform: scale(1.05); opacity: 1; filter: drop-shadow(0 0 20px rgba(79,172,254,0.6)); }}
}}

@keyframes slideUpFade {{
    0% {{ opacity: 0; transform: translateY(20px); }}
    100% {{ opacity: 1; transform: translateY(0); }}
}}

@keyframes fadeOutOverlay {{
    0% {{ opacity: 1; }}
    80% {{ opacity: 1; }}
    100% {{ opacity: 0; visibility: hidden; }}
}}

/* --- SIDEBAR STYLING --- */
[data-testid="stSidebar"] {{
    background: rgba(15, 15, 25, 0.6) !important;
    backdrop-filter: blur(20px) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.05);
}}

/* --- CHAT BUBBLES --- */
.chat-user {{
    background: linear-gradient(135deg, rgba(79, 172, 254, 0.15), rgba(167, 139, 250, 0.1));
    border: 1px solid rgba(79, 172, 254, 0.3);
    border-radius: 20px 20px 5px 20px;
    padding: 1.2rem;
    margin: 1rem 0 1rem auto;
    max-width: 85%;
    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
}}

.chat-assistant {{
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(10px);
    border-radius: 20px 20px 20px 5px;
    padding: 1.2rem;
    margin: 1rem auto 1rem 0;
    max-width: 90%;
    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
}}

/* --- GLASS CARD METRICS --- */
.glass-card {{
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 15px;
    padding: 1rem;
    margin-top: 1rem;
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
}}
.glass-metric {{
    flex: 1;
    background: rgba(0,0,0,0.2);
    border-radius: 10px;
    padding: 10px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.03);
}}
.metric-val {{ font-size: 1.4rem; font-weight: 800; color: #4facfe; }}
.metric-lbl {{ font-size: 0.7rem; color: #8fa3b0; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }}

</style>

<!-- STARTING ANIMATION HTML -->
<div class="starting-overlay">
    <div class="starting-logo">Neural Nexus</div>
    <div class="starting-sub">Initializing C-RAG Engine...</div>
</div>
""", unsafe_allow_html=True)

# ── Session State Init ─────────────────────────────────────────
if "messages" not in st.session_state: st.session_state.messages = []
if "langchain_messages" not in st.session_state: st.session_state.langchain_messages = []
if "suggested_questions" not in st.session_state: st.session_state.suggested_questions = []

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"<div style='text-align:center;'>{logo_img}</div>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center; margin-bottom: 2rem;'>Neural Nexus</h3>", unsafe_allow_html=True)
    
    st.markdown("#### 📁 Knowledge Base")
    tab_file, tab_url = st.tabs(["Upload File", "Web URL"])

    with tab_file:
        uploaded = st.file_uploader("", type=["pdf", "txt", "md"], label_visibility="collapsed")
        if st.button("🚀 Ingest Document", use_container_width=True, disabled=uploaded is None):
            from app.ingest import ingest
            import os
            tmp_dir = tempfile.mkdtemp()
            tmp_path = os.path.join(tmp_dir, uploaded.name)
            with open(tmp_path, "wb") as f:
                f.write(uploaded.getbuffer())
            
            with st.spinner("Analyzing & indexing..."):
                try:
                    ingest(tmp_path)
                    st.session_state.last_ingested = uploaded.name
                    st.session_state.suggested_questions = [
                        f"What are the main points in {uploaded.name}?",
                        f"Can you summarize the key findings of {uploaded.name}?",
                        f"What is the context of {uploaded.name}?"
                    ]
                    st.success(f"Verified & Indexed: {uploaded.name}")
                except Exception as e:
                    st.error(f"Error: {e}")
                finally:
                    shutil.rmtree(tmp_dir, ignore_errors=True)

    with tab_url:
        url = st.text_input("Source URL")
        if st.button("🌐 Ingest URL", use_container_width=True, disabled=not url):
            from app.ingest import ingest
            with st.spinner("Reading URL..."):
                try:
                    ingest(url)
                    st.session_state.last_ingested = url
                    st.session_state.suggested_questions = [
                        "What is this website about?",
                        "Summarize the key takeaways from the page.",
                        "Who is the author or organization behind this?"
                    ]
                    st.success("Verified Source")
                except Exception as e:
                    st.error(f"Error: {e}")

    st.divider()
    st.markdown("#### ⚙️ Settings")
    show_debug = st.checkbox("Show Performance Metrics", value=True)
    
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.langchain_messages = []
        st.rerun()
        
    st.markdown("<br><p style='font-size:0.8rem; color: gray; text-align:center;'>Powered by Groq & DeepSeek</p>", unsafe_allow_html=True)

# ── Main Content ──────────────────────────────────────────────
st.markdown("<h1 style='font-weight:800; font-size: 2.5rem; margin-bottom: 0;'>Neural Nexus <span style='font-size:1rem; color:#4facfe; border: 1px solid #4facfe; padding: 2px 8px; border-radius: 12px; vertical-align: middle;'>C-RAG v2</span></h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 1.1rem; opacity: 0.7; margin-top: 0; margin-bottom: 2rem;'>Advanced Self-Reflective Retrieval Augmented Generation</p>", unsafe_allow_html=True)

# Suggested Questions
if st.session_state.suggested_questions:
    st.markdown(f"<p style='font-size: 0.85rem; opacity: 0.6; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 1px;'>✨ Suggested for: <b>{st.session_state.get('last_ingested', 'Document')}</b></p>", unsafe_allow_html=True)
    cols = st.columns(len(st.session_state.suggested_questions))
    for i, q in enumerate(st.session_state.suggested_questions):
        display_q = q if len(q) < 50 else q[:47] + "..."
        if cols[i].button(display_q, key=f"sq_{i}", use_container_width=True, help=q):
            st.session_state.pending_question = q

st.divider()

# ── Chat Interface ────────────────────────────────────────────
chat_container = st.container()

with chat_container:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"<div class='chat-user'>{msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-assistant'>{msg['content']}</div>", unsafe_allow_html=True)
            if show_debug and "meta" in msg:
                m = msg["meta"]
                lat_str = f"{sum(m.get('latencies', {}).values()):.2f}s" if m.get("latencies") else "—"
                meth_str = "🌐 Web Search" if m.get("web_search_used") else "📚 Vector DB"
                rel_str = f"{m.get('relevance_score', 0):.0%}"
                trust_val = m.get("trust_score", 100.0)
                trust_rating = m.get("trust_rating", "High Trust")
                escalation = m.get("escalation_status")
                
                status_badge = f"<span style='color:#ff5252; font-weight:bold;'>⚠️ ESCALATED</span>" if escalation == "pending_human_verification" else f"<span style='color:#00f5a0;'>✓ {trust_rating}</span>"
                
                st.markdown(f'''
                <div class='glass-card'>
                    <div class='glass-metric'><div class='metric-val'>{rel_str}</div><div class='metric-lbl'>Relevance</div></div>
                    <div class='glass-metric'><div class='metric-val' style='color:#00f5a0;'>{trust_val:.0f}/100</div><div class='metric-lbl'>Trust Index</div></div>
                    <div class='glass-metric'><div class='metric-val' style='color:#a78bfa;'>{meth_str}</div><div class='metric-lbl'>Source</div></div>
                    <div class='glass-metric'><div class='metric-val' style='color:#ffbd2e;'>{lat_str}</div><div class='metric-lbl'>Latency</div></div>
                </div>
                ''', unsafe_allow_html=True)
                
                with st.expander("📊 Latency & Security Telemetry"):
                    from app.trust.latency_tracer import LatencyTracer
                    breakdown_data = LatencyTracer.get_breakdown_table(m.get("latencies", {}))
                    if breakdown_data:
                        st.markdown("**Per-Stage Execution Breakdown:**")
                        st.table(breakdown_data)
                    st.markdown(f"**Escalation Status:** `{escalation or 'Normal Resolution'}` | **Trust Rating:** {status_badge}", unsafe_allow_html=True)

                if m.get("sources"):
                    with st.expander("📎 Verified Sources"):
                        for src in m["sources"]:
                            st.markdown(f"- `{src}`")

if "pending_question" in st.session_state:
    question = st.session_state.pop("pending_question")
else:
    question = st.chat_input("Ask Neural Nexus anything...")

if question:
    # Append user question
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.langchain_messages.append(HumanMessage(content=question))
    
    with chat_container:
        st.markdown(f"<div class='chat-user'>{question}</div>", unsafe_allow_html=True)
        
        with st.spinner("Analyzing context and generating response..."):
            from app.graph.pipeline import rag_graph
            from app.trust.trust_score_engine import trust_engine
            import uuid
            
            req_id = f"ui_{uuid.uuid4().hex[:10]}"
            initial_state = {
                "question": question,
                "messages": st.session_state.langchain_messages,
                "documents": [],
                "generation": None,
                "web_search_used": False,
                "retry_count": 0,
                "relevance_score": 0.0,
                "sources": [],
                "hallucination_check": "grounded",
                "node_execution_times": {},
                "request_id": req_id,
                "escalation_status": None,
                "manual_context_override": None,
            }

            try:
                result = rag_graph.invoke(initial_state)
                answer = result.get("generation", "No answer generated.")
                st.session_state.langchain_messages = result.get("messages", st.session_state.langchain_messages)
                
                trust_breakdown = trust_engine.record(req_id, result).breakdown
                
                meta = {
                    "web_search_used": result.get("web_search_used", False),
                    "relevance_score": result.get("relevance_score", 0.0),
                    "retry_count": result.get("retry_count", 0),
                    "sources": result.get("sources", []),
                    "latencies": result.get("node_execution_times", {}),
                    "trust_score": trust_breakdown.composite_score,
                    "trust_rating": trust_breakdown.rating,
                    "escalation_status": result.get("escalation_status"),
                }
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "meta": meta
                })
            except Exception as e:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Pipeline Error: {e}",
                    "meta": {}
                })
    
    st.rerun()
