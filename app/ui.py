"""
ui.py — High-Level Premium Streamlit UI for the Corrective RAG pipeline.
"""
import streamlit as st
from pathlib import Path
import tempfile
import shutil
import base64
import time

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Neural Nexus | Intelligent AI Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Image Helper ──────────────────────────────────────────────
def get_image_base64(path):
    try:
        with open(path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except:
        return ""

# Try to find the generated logo
logo_path = list(Path("C:/Users/LENOVO/.gemini/antigravity/brain/5e17c66a-3682-440f-985b-4ca23a46117e/").glob("neural_nexus_logo_*.png"))
logo_base64 = get_image_base64(logo_path[0]) if logo_path else ""

# ── Custom CSS for Premium Design ──────────────────────────────
st.markdown(f"""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}

    /* Main Background Gradient */
    .stApp {{
        background: radial-gradient(circle at top right, #1a1a2e, #0f0f1a);
        color: #e0e0e0;
    }}

    /* Sidebar Styling */
    [data-testid="stSidebar"] {{
        background-color: rgba(15, 15, 26, 0.8);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }}

    /* Header Styling */
    .main-header {{
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }}

    /* Glassmorphism Containers */
    .glass-card {{
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 2rem;
        margin-bottom: 1.5rem;
    }}

    /* Custom Chat Bubbles */
    .stChatMessage {{
        background-color: rgba(255, 255, 255, 0.02) !important;
        border-radius: 15px !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        margin-bottom: 1rem !important;
    }}

    /* Metric Styling */
    [data-testid="stMetricValue"] {{
        color: #00f2fe !important;
        font-weight: 700 !important;
    }}

    /* Sidebar Logo Header */
    .sidebar-logo {{
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 2rem;
    }}
    
    /* Buttons */
    .stButton > button {{
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }}
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0, 242, 254, 0.3);
    }}

    /* Splash Screen Animation */
    #splash-screen {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: #05050a;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        z-index: 999999;
        animation: fadeOutSplash 1.5s cubic-bezier(0.65, 0, 0.35, 1) forwards;
        animation-delay: 6s;
        overflow: hidden;
    }}

    .splash-content {{
        position: relative;
        text-align: center;
        z-index: 1;
    }}

    .splash-logo {{
        width: 350px;
        animation: neonFlicker 3s linear infinite, logoIntro 6s ease-in-out forwards;
    }}

    .boot-text {{
        margin-top: 2rem;
        font-family: 'Courier New', monospace;
        color: #00f2fe;
        font-size: 1rem;
        letter-spacing: 5px;
        text-transform: uppercase;
        animation: blink 0.8s step-end infinite;
    }}

    @keyframes neonFlicker {{
        0%, 19%, 21%, 23%, 25%, 54%, 56%, 100% {{ opacity: 1; filter: drop-shadow(0 0 15px #00f2fe); }}
        20%, 22%, 24%, 55% {{ opacity: 0.7; filter: none; }}
    }}

    @keyframes logoIntro {{
        0% {{ transform: scale(0.3); opacity: 0; }}
        15% {{ transform: scale(1.1); opacity: 1; }}
        80% {{ transform: scale(1); opacity: 1; }}
        100% {{ transform: scale(5); opacity: 0; }}
    }}

    @keyframes fadeOutSplash {{
        from {{ opacity: 1; visibility: visible; }}
        to {{ opacity: 0; visibility: hidden; }}
    }}

    @keyframes blink {{
        50% {{ opacity: 0; }}
    }}

    /* Quick Question Chips */
    .question-chip {{
        display: inline-block;
        padding: 10px 20px;
        margin: 5px;
        border-radius: 50px;
        background: rgba(0, 242, 254, 0.1);
        border: 1px solid rgba(0, 242, 254, 0.3);
        color: #00f2fe;
        cursor: pointer;
        transition: all 0.3s ease;
        font-size: 0.9rem;
    }}
    .question-chip:hover {{
        background: rgba(0, 242, 254, 0.2);
        border-color: #00f2fe;
        transform: translateY(-2px);
    }}
</style>
""", unsafe_allow_html=True)

# ── Splash Screen ─────────────────────────────────────────────
if "splash_shown" not in st.session_state:
    st.markdown(f'''
        <div id="splash-screen">
            <div class="splash-content">
                <img class="splash-logo" src="data:image/png;base64,{logo_base64}">
                <div class="boot-text">Initializing Neural Nexus...</div>
            </div>
        </div>
    ''', unsafe_allow_html=True)
    st.session_state.splash_shown = True

# ── Question Generation Logic ─────────────────────────────────
def generate_suggested_questions(text: str):
    from app.utils.llm_factory import get_generator_llm
    llm = get_generator_llm()
    prompt = (
        "Based on the following document excerpt, generate 3 clear, intriguing, "
        "and specific questions that a user might want to ask about this document. "
        "Return ONLY a numbered list of 3 questions.\n\n"
        f"TEXT:\n{text[:4000]}\n\nQUESTIONS:"
    )
    try:
        response = llm.invoke(prompt)
        questions = [q.split(". ", 1)[-1].strip() for q in response.content.strip().split("\n") if q.strip()]
        return questions[:3]
    except Exception as e:
        return ["What is the main topic of this document?", "Can you summarize the key findings?", "Who are the main entities mentioned?"]

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    if logo_base64:
        st.markdown(f'''
            <div class="sidebar-logo">
                <img src="data:image/png;base64,{logo_base64}" width="180" style="border-radius: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.5);">
            </div>
        ''', unsafe_allow_html=True)
    
    st.markdown("### 📄 Knowledge Base")
    tab_file, tab_url = st.tabs(["Upload", "Web URL"])

    with tab_file:
        uploaded = st.file_uploader("", type=["pdf", "txt", "md"], label_visibility="collapsed")
        if st.button("🚀 Ingest Document", use_container_width=True, disabled=uploaded is None):
            from app.ingest import ingest
            suffix = Path(uploaded.name).suffix
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                shutil.copyfileobj(uploaded, tmp)
                tmp_path = tmp.name
            
            with st.spinner("Analyzing & indexing..."):
                try:
                    # Read sample text for question generation
                    sample_text = ""
                    if suffix == ".txt":
                        with open(tmp_path, "r", encoding="utf-8") as f:
                            sample_text = f.read(5000)
                    else:
                        # For PDF, we just note we processed it
                        sample_text = f"A document named {uploaded.name}"
                    
                    ingest(tmp_path)
                    st.session_state.suggested_questions = generate_suggested_questions(sample_text)
                    st.session_state.last_ingested = uploaded.name
                    st.success(f"Verified: {uploaded.name}")
                except Exception as e:
                    st.error(f"Error: {e}")
                finally:
                    Path(tmp_path).unlink(missing_ok=True)

    with tab_url:
        url = st.text_input("Source URL")
        if st.button("🌐 Ingest URL", use_container_width=True, disabled=not url):
            from app.ingest import ingest
            with st.spinner("Reading URL..."):
                try:
                    ingest(url)
                    st.session_state.suggested_questions = generate_suggested_questions(f"Web content from {url}")
                    st.session_state.last_ingested = url
                    st.success("Verified Source")
                except Exception as e:
                    st.error(f"Error: {e}")

    st.divider()
    st.markdown("### ⚙️ Engine Settings")
    show_debug = st.checkbox("Show Debug Metadata", value=False)
    st.info("Active Provider: **Groq + DeepSeek**")

# ── Main Content ──────────────────────────────────────────────
st.markdown("<div class='main-header'>Neural Nexus</div>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 1.2rem; opacity: 0.8; margin-top: -1rem; margin-bottom: 2rem; color: #4facfe;'>Advanced Self-Reflective Intelligence for Precise Insights</p>", unsafe_allow_html=True)

# Suggested Questions Section
if "suggested_questions" in st.session_state and st.session_state.suggested_questions:
    st.markdown("<div style='margin-top: 2rem; margin-bottom: 1rem;'>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size: 0.9rem; opacity: 0.6; margin-left: 5px;'>✨ INSIGHTS FOR: <b>{st.session_state.get('last_ingested', 'Document')}</b></p>", unsafe_allow_html=True)
    
    # Use columns but with more control
    q_cols = st.columns(len(st.session_state.suggested_questions))
    for i, q in enumerate(st.session_state.suggested_questions):
        # Clean up question text if too long
        display_q = q if len(q) < 60 else q[:57] + "..."
        if q_cols[i].button(f"🔍 {display_q}", key=f"sq_{i}", use_container_width=True, help=q):
            st.session_state.pending_question = q
    st.markdown("</div>", unsafe_allow_html=True)

# ── Chat Interface ────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Container for chat history
chat_container = st.container()

with chat_container:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and show_debug and "meta" in msg:
                with st.expander("Engine Analytics"):
                    st.json(msg["meta"])

# Handle pending question from suggested chips
if "pending_question" in st.session_state:
    question = st.session_state.pop("pending_question")
    # Manual trigger of the chat logic
else:
    question = st.chat_input("Ask the agent anything...")

if question:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": question})
    with chat_container:
        with st.chat_message("user"):
            st.markdown(question)

    # Run pipeline
    with chat_container:
        with st.chat_message("assistant"):
            answer_placeholder = st.empty()
            
            with st.spinner("Analyzing context and generating response..."):
                from app.graph.pipeline import rag_graph

                initial_state = {
                    "question": question,
                    "documents": [],
                    "generation": None,
                    "web_search_used": False,
                    "retry_count": 0,
                    "relevance_score": 0.0,
                    "sources": [],
                    "hallucination_check": "grounded"
                }

                try:
                    result = rag_graph.invoke(initial_state)
                    answer = result.get("generation", "No answer generated.")
                    sources = result.get("sources", [])
                    web_used = result.get("web_search_used", False)
                    relevance = result.get("relevance_score", 0.0)
                    retries = result.get("retry_count", 1)

                    # Display answer
                    answer_placeholder.markdown(answer)

                    # Metadata badges in a glass card
                    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                    m_cols = st.columns(3)
                    m_cols[0].metric("Relevance", f"{relevance:.0%}")
                    m_cols[1].metric("Method", "🌐 Web Search" if web_used else "📚 Vector DB")
                    m_cols[2].metric("Refinements", retries)
                    
                    if sources:
                        with st.expander("📎 Verified Sources"):
                            for src in sources:
                                st.markdown(f"- `{src}`")
                    st.markdown("</div>", unsafe_allow_html=True)

                    meta = {
                        "web_search_used": web_used,
                        "relevance_score": relevance,
                        "retry_count": retries,
                        "sources": sources,
                    }

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "meta": meta,
                    })

                except Exception as e:
                    error_msg = f"Pipeline Error: {e}"
                    answer_placeholder.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                    })
    # Force refresh to clear the suggested questions or handle next input
    st.rerun()
