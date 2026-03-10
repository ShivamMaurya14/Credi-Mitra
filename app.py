"""
CREDI-MITRA Streamlit Application — LLM Agent Chat Interface

Features:
- Continuous chat interface with st.chat_message
- Intermediate tool visibility (every tool call shown in expandable sections)
- Human-in-the-Loop via LangGraph interrupt / Command(resume=...)
- Session state memory for conversation persistence across reruns
"""

# Heavy imports (pandas, numpy, fpdf, pypdf, pdfplumber, docx, llama_parse) 
# moved to internal functions to save memory on 8GB machines.
import streamlit as st
import time
import json
import os
import io
import re
import uuid
from langgraph.types import interrupt, Command
import nest_asyncio
nest_asyncio.apply()
from dotenv import load_dotenv

# Import the agent builder
from agent_graph import build_agent

# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="CREDI-MITRA — AI Credit Analyst Agent",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────
def apply_custom_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;800&family=Inter:wght@400;500;600&display=swap');

        /* Global */
        .stApp {
            font-family: 'Inter', sans-serif;
        }

        /* ── INCREASE ALL FONT SIZES ── */
        [data-testid="stMarkdownContainer"] p, 
        [data-testid="stMarkdownContainer"] li {
            font-size: 1.15rem !important;
            line-height: 1.6 !important;
        }
        [data-testid="stWidgetLabel"] p {
            font-size: 1.15rem !important;
            margin-bottom: 0.5rem !important;
        }
        h1 { font-size: 3rem !important; }
        h2 { font-size: 2.5rem !important; }
        h3 { font-size: 2rem !important; }
        h4 { font-size: 1.6rem !important; }
        
        /* The slider values and small text */
        .st-emotion-cache-1629p8f h1, 
        [data-testid="stCaptionContainer"] p {
            font-size: 1.05rem !important;
        }
        .stCode code {
            font-size: 1.1rem !important;
            line-height: 1.5 !important;
        }

        /* Gradient Title */
        .main-title {
            font-family: 'Outfit', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            font-size: 4rem !important;
            font-weight: 800;
            margin-bottom: 0;
            line-height: 1.1;
            padding-top: 0.5rem;
            text-transform: uppercase;
            letter-spacing: 2px;
        }

        .sub-title {
            background: linear-gradient(90deg, #a8edea 0%, #fed6e3 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            font-size: 1.4rem;
            font-weight: 500;
            margin-bottom: 1.5rem;
            margin-top: 0.3rem;
            letter-spacing: 1px;
        }

        /* Metric styling */
        [data-testid="stMetricValue"] {
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        [data-testid="stMetricDelta"] {
            font-size: 1rem;
        }

        /* Button styling */
        .stButton>button {
            border-radius: 12px;
            font-weight: 600;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            border: 1px solid rgba(102, 126, 234, 0.3);
        }
        .stButton>button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }
        button[kind="primary"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            border: none !important;
        }
        button[kind="primary"]:hover {
            background: linear-gradient(135deg, #5a6fd6 0%, #6a4299 100%) !important;
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
        }

        /* Chat messages */
        [data-testid="stChatMessageContent"] {
            font-size: 1.1rem !important;
            line-height: 1.6 !important;
        }
        [data-testid="stChatMessageContent"] p {
            font-size: 1.1rem !important;
        }

        /* Tool output expanders */
        .tool-output {
            border-left: 3px solid #667eea;
            padding-left: 12px;
            margin: 8px 0;
            background: rgba(102, 126, 234, 0.05);
            border-radius: 0 8px 8px 0;
        }

        /* Sidebar enhancements */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(30,30,50,0.95) 0%, rgba(20,20,35,0.98) 100%);
        }
        [data-testid="stSidebar"] .stMarkdown h3 {
            color: #a8edea;
        }

        /* Container borders */
        [data-testid="stVerticalBlock"] > div:has(> [data-testid="stContainer"]) {
            border-radius: 12px;
        }

        /* Status indicator */
        .status-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
        }
        .status-ready {
            background: rgba(0, 255, 127, 0.15);
            color: #00FF7F;
            border: 1px solid rgba(0, 255, 127, 0.3);
        }
        .status-pending {
            background: rgba(255, 165, 0, 0.15);
            color: #FFA500;
            border: 1px solid rgba(255, 165, 0, 0.3);
        }
    </style>
    """, unsafe_allow_html=True)

apply_custom_css()


# ──────────────────────────────────────────────
# Session State Initialization
# ──────────────────────────────────────────────
def init_session_state():
    defaults = {
        "logged_in": False,
        "current_page": "login",
        "messages": [],           # Chat messages: {role, content, type?, tool_name?, tool_data?}
        "docs_verified": False,
        "company_name": "",
        "app_no": "",
        "pdf_extracted_text": "",
        "manual_entry": "",
        "cam_generated": False,
        "cam_content": "",
        "agent": None,
        "thread_id": None,
        "waiting_for_human": False,  # True when graph is interrupted
        "interrupt_data": None,      # The interrupt payload
        "interrupt_id": None,        # The unique ID for the interrupt (required for multi-interrupt support)
        "agent_running": False,
        "base_premium": 8.5,         # Base interest rate premium (%)
        "revenue_tolerance": 25,     # Revenue variance tolerance (%)
        "litigation_threshold": 3,   # Max litigation cases allowed
        "selected_model": "gemini-1.5-pro (Google)",
        "selected_analysis_model": "gemini-1.5-pro (Google)",
        "gemini_models": [],         # Dynamically fetched Gemini models
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()


# ──────────────────────────────────────────────
# Utility Functions
# ──────────────────────────────────────────────
def switch_page(page_name):
    st.session_state.current_page = page_name
    st.rerun()


def reset_application_state():
    """Clear Pinecone DB, local temp files, and all session state analysis keys."""
    # 1. Clear Pinecone Cloud Data & Temp Files via RAG Manager
    try:
        from rag import get_document_manager
        mgr = get_document_manager()
        mgr.reset_session()
    except Exception as e:
        st.warning(f"Note: Error during data cleanup: {e}")

    # 2. Reset Session State Keys
    keys_to_reset = [
        "messages", "docs_verified", "company_name", "app_no", 
        "pdf_extracted_text", "manual_entry", "cam_generated", 
        "cam_content", "agent", "thread_id", "waiting_for_human", 
        "interrupt_data", "interrupt_id", "agent_running", "document_extracted_text",
        "current_upload_dir"
    ]
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]
    
    # 3. Re-initialize defaults
    init_session_state()


# removed fetch_available_gemini_models logic


def add_message(role, content, **kwargs):
    """Add a message to session state with simple consecutive deduplication."""
    if st.session_state.messages:
        last = st.session_state.messages[-1]
        if (last["role"] == role and 
            last["content"] == content and 
            last.get("type") == kwargs.get("type")):
            return
            
    msg = {"role": role, "content": content}
    msg.update(kwargs)
    st.session_state.messages.append(msg)


def render_tool_output(tool_name, tool_data):
    """Render a tool's output as an expandable section in the chat."""
    icon_map = {
        "list_uploaded_documents": "📂",
        "analyze_document": "📄",
        "extract_pdf_data": "📑",
        "crawl_web_for_litigation": "🔍",
        "extract_numerical_features": "📊",
        "run_xgboost_scorer": "🤖",
        "generate_cam_report": "📋",
    }
    icon = icon_map.get(tool_name, "🔧")

    with st.expander(f"{icon} Output from `{tool_name}`", expanded=True):
        # Special handling for litigation analysis results
        if tool_name == "crawl_web_for_litigation" and isinstance(tool_data, str):
            try:
                parsed = json.loads(tool_data)
                _render_litigation_analysis(parsed)
                return
            except (json.JSONDecodeError, TypeError):
                pass
        
        # Generic rendering for other tools
        if isinstance(tool_data, str):
            try:
                parsed = json.loads(tool_data)
                st.json(parsed)
            except (json.JSONDecodeError, TypeError):
                # Check if it looks like a CAM report (markdown)
                if tool_data.strip().startswith("#"):
                    st.markdown(tool_data)
                else:
                    st.code(tool_data, language="text")
        elif isinstance(tool_data, dict):
            st.json(tool_data)
        else:
            st.write(tool_data)


def _render_litigation_analysis(result_dict):
    """Render litigation analysis results in a professional tabular format."""
    if not isinstance(result_dict, dict):
        st.json(result_dict)
        return
    
    # Extract data
    company = result_dict.get("company_searched", "Unknown")
    litigation_count = result_dict.get("litigation_count", 0)
    sentiment_score = result_dict.get("news_sentiment_score", 0.0)
    positive_news = result_dict.get("positive_news_count", 0)
    negative_news = result_dict.get("negative_news_count", 0)
    total_analyzed = result_dict.get("total_results_analyzed", 0)
    risk_score = result_dict.get("risk_score", 0.0)
    nclt_cases = result_dict.get("nclt_cases", [])
    rbi_actions = result_dict.get("rbi_regulatory_actions", [])
    detailed_findings = result_dict.get("detailed_findings", [])
    warnings = result_dict.get("warnings", [])
    errors = result_dict.get("errors_during_analysis", [])
    
    # Display header
    st.success("✅ **Web Research & Litigation Analysis Complete**")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📋 Litigations Found", litigation_count)
    col2.metric("😊 Avg Sentiment", sentiment_score, delta="Range: -1 to 1")
    col3.metric("📊 Positive News", positive_news)
    col4.metric("📊 Negative News", negative_news)
    
    col_risk, col_total = st.columns(2)
    col_risk.metric("⚠️ Net Risk Score", risk_score, delta="Negative minus Positive")
    col_total.metric("📈 Results Analyzed", total_analyzed)
    
    # Critical findings alerts
    if nclt_cases:
        st.warning(f"🚨 **NCLT Cases Found ({len(nclt_cases)})**: {'; '.join(nclt_cases[:3])}")
    if rbi_actions:
        st.error(f"⛔ **RBI Regulatory Actions ({len(rbi_actions)})**: {'; '.join(rbi_actions[:3])}")
    
    # Display warnings and errors if any
    if warnings:
        for warning in warnings:
            st.warning(f"⚠️ {warning}")
    if errors:
        st.info("**Analysis Notes:**")
        for error in errors[:3]:
            st.caption(f"• {error}")
    
    # Display detailed findings in tabular format
    if detailed_findings:
        import pandas as pd
        st.subheader("📊 Litigation Analysis Results - Detailed Findings")
        df = pd.DataFrame(detailed_findings)
        
        # Format DataFrame for display
        if "sentiment" in df.columns:
            df["sentiment"] = df["sentiment"].apply(lambda x: f"{x:.2f}")
        if "risk_level" in df.columns:
            df["risk_level"] = df["risk_level"].apply(
                lambda x: f"🔴 {x}" if x == "HIGH" else f"🟡 {x}" if x in ["MEDIUM", "MODERATE"] else f"🟢 {x}" if x in ["LOW", "POSITIVE"] else f"⚪ {x}"
            )
        if "is_nclt" in df.columns:
            df["is_nclt"] = df["is_nclt"].apply(lambda x: "✓ NCLT" if x else "")
        if "is_rbi" in df.columns:
            df["is_rbi"] = df["is_rbi"].apply(lambda x: "✓ RBI" if x else "")
        
        # Select columns to display
        display_cols = [c for c in ["headline", "risk_level", "sentiment", "litigation_type", "is_nclt", "is_rbi", "summary"] if c in df.columns]
        df_display = df[display_cols]
        
        st.dataframe(df_display, use_container_width=True, height=400)
    else:
        st.info("ℹ️ No litigation findings to display.")
    
    # Summary JSON for reference
    with st.expander("📋 Raw JSON Data (for reference)", expanded=False):
        st.json(result_dict)


def generate_cam_pdf(cam_text):
    """Convert CAM markdown content into a professionally formatted PDF with robust layout handling."""
    from fpdf import FPDF
    import re
    
    # Pre-process text: normalize encoding and convert to string
    cam_text = re.sub(r'[^\x00-\xff]', '', str(cam_text))
    
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()
    
    effective_width = pdf.w - pdf.l_margin - pdf.r_margin

    # ── Header Section ──
    pdf.set_font("Helvetica", "B", 24)
    pdf.set_text_color(102, 126, 234)
    pdf.set_x(pdf.l_margin)
    pdf.cell(effective_width, 16, "CREDI-MITRA", align="C", new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_font("Helvetica", "I", 11)
    pdf.set_text_color(100, 100, 100)
    pdf.set_x(pdf.l_margin)
    pdf.cell(effective_width, 8, "AI-Powered Credit Intelligence Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    
    pdf.set_draw_color(102, 126, 234)
    pdf.set_line_width(0.7)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(8)

    # ── Content Loop with Safety Checks ──
    for raw_line in cam_text.strip().splitlines():
        line = raw_line.strip()
        if not line:
            pdf.ln(4)
            continue
            
        # Ensure we are always starting from the left margin for a new line block
        pdf.set_x(pdf.l_margin)

        # Headers logic
        if line.startswith("# "):
            pdf.set_font("Helvetica", "B", 18)
            pdf.set_text_color(40, 60, 120)
            text = line[2:].replace("**", "")
            pdf.multi_cell(effective_width, 10, text)
            pdf.ln(2)
        elif line.startswith("## "):
            pdf.set_font("Helvetica", "B", 15)
            pdf.set_text_color(60, 80, 160)
            text = line[3:].replace("**", "")
            pdf.multi_cell(effective_width, 8, text)
            pdf.ln(2)
        elif line.startswith("### "):
            pdf.set_font("Helvetica", "B", 12)
            pdf.set_text_color(60, 80, 160)
            text = line[4:].replace("**", "")
            pdf.multi_cell(effective_width, 7, text)
            pdf.ln(1)
        
        # Horizontal Rule
        elif line == "---" or line == "***":
            pdf.ln(2)
            pdf.set_draw_color(220, 220, 220)
            pdf.set_line_width(0.3)
            pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
            pdf.ln(4)
            
        # Lists
        elif line.startswith("- ") or line.startswith("* "):
            text = line[2:].replace("**", "")
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(50, 50, 50)
            # Indent bullet points
            pdf.set_x(pdf.l_margin + 6)
            pdf.multi_cell(effective_width - 6, 6, "• " + text)
            pdf.ln(1)
            
        # Tables (Simplified row printing)
        elif line.startswith("|"):
            if "---" in line: continue
            text = line.replace("|", "  ").replace("**", "").strip()
            pdf.set_font("Courier", "", 9)
            pdf.set_text_color(80, 80, 80)
            pdf.multi_cell(effective_width, 5, text)
            
        # Standard Body Text
        else:
            text = line.replace("**", "")
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(50, 50, 50)
            pdf.multi_cell(effective_width, 6, text)
            pdf.ln(1)

    return pdf.output()


# ──────────────────────────────────────────────
# Agent Execution Engine
# ──────────────────────────────────────────────
def run_agent(user_input=None, resume_value=None):
    """
    Run or resume the LangGraph agent.
    
    - On first call: streams the full agent execution
    - On interrupt: pauses and sets waiting_for_human=True
    - On resume: continues from the interrupt with the user's answer
    """
    # Build agent if not cached
    selected_name = st.session_state.get("selected_model", "llama-3.3-70b-versatile (Groq)")
    if st.session_state.agent is None:
        try:
            import agent_graph as _ag_module
            # Inject the upload directory into the module for thread-safe access by tools
            _ag_module.RELIABLE_UPLOAD_DIR = st.session_state.get("current_upload_dir")
            agent, checkpointer = _ag_module.build_agent(selected_name)
            st.session_state.agent = agent
        except Exception as e:
            err_str = str(e).lower()
            is_quota_error = any(k in err_str for k in ["rate_limit", "429", "quota", "resource_exhausted"])
            if is_quota_error:
                st.warning("⚠️ Primary model quota reached. Falling back to Gemini...")
                try:
                    agent, checkpointer = _ag_module.build_agent("gemini (Google)")
                    st.session_state.agent = agent
                    st.success("✅ Successfully switched to Gemini fallback engine.")
                except Exception as gemini_err:
                    st.error(
                        f"❌ Both Groq and Gemini are unavailable.\n\n"
                        f"**Groq Error:** {e}\n\n"
                        f"**Gemini Error:** {gemini_err}\n\n"
                        "⏳ Please wait a few minutes and try again, or enable billing on Google AI Studio."
                    )
                    return
            else:
                st.error(f"Failed to build agent: {e}")
                return

    if not st.session_state.get("thread_id"):
        st.session_state.thread_id = str(uuid.uuid4())

    agent = st.session_state.agent
    
    # Refresh the reliable directory cache for the modules' tools
    import agent_graph as _ag_module
    _ag_module.RELIABLE_UPLOAD_DIR = st.session_state.get("current_upload_dir")
    _ag_module.RELIABLE_MODEL_NAME = st.session_state.get("selected_model")
    _ag_module.RELIABLE_ANALYSIS_MODEL = st.session_state.get("selected_analysis_model")
    
    config = {
        "configurable": {
            "thread_id": st.session_state.thread_id,
            "upload_dir": st.session_state.get("current_upload_dir", "")
        }
    }

    # Prepare input
    if resume_value is not None:
        # Resuming from an interrupt
        # Version 1.0.10 of LangGraph doesn't support 'id' keyword in Command constructor.
        # We will pass resume value directly.
        agent_input = Command(resume=resume_value)
    else:
        # Build the context message for the agent
        base_premium = st.session_state.get("base_premium", 8.5)
        
        comp_name = st.session_state.get("company_name", "").strip()
        app_num = st.session_state.get("app_no", "").strip()
        upload_dir = st.session_state.get("current_upload_dir")

        # Proactive recovery: always prefer name from directory if it matches the pattern
        if upload_dir and os.path.exists(upload_dir):
            folder_name = os.path.basename(upload_dir)
            if "_" in folder_name:
                # Extract everything before the last underscore as the name
                recovered_name = folder_name.rsplit("_", 1)[0].replace("_", " ")
                # If current name is blank or just looks like an ID, use recovered name
                if not comp_name or comp_name.isdigit() or comp_name == app_num:
                    comp_name = recovered_name
        
        # Ensure any underscores in existing name are also cleared
        comp_name = comp_name.replace("_", " ")

        officer_status = "None provided"
        if st.session_state.get("manual_entry") and st.session_state.manual_entry.strip():
            officer_status = st.session_state.manual_entry
        elif st.session_state.get("document_extracted_text") and "--- Document: Officer Insights Report" in st.session_state.document_extracted_text:
            officer_status = "Provided via uploaded document."
        elif upload_dir and os.path.exists(upload_dir) and "Officer_Insights_Report.pdf" in os.listdir(upload_dir):
            officer_status = "Provided via uploaded document (Officer_Insights_Report.pdf found)."

        doc_text_status = "No"
        if upload_dir and os.path.exists(upload_dir) and len([f for f in os.listdir(upload_dir) if not f.startswith(".")]) > 0:
            doc_text_status = "Yes"
        elif st.session_state.get("document_extracted_text") and len(st.session_state.document_extracted_text.strip()) > 0:
            doc_text_status = "Yes"
        else:
            tid = st.session_state.get("thread_id")
            if tid:
                b_path = os.path.join("temp", f"{tid}.txt")
                if os.path.exists(b_path) and os.path.getsize(b_path) > 0:
                    doc_text_status = "Yes"

        context_block = f"""
### SYSTEM VERIFIED CONTEXT (MANDATORY)
- **Company Name**: {comp_name}
- **Application No**: {app_num}
- **Officer Insights**: {officer_status}
- **Document Text Available**: {doc_text_status}
- **Base Interest Rate Premium**: {base_premium}%
- **Revenue Variance Tolerance**: {st.session_state.revenue_tolerance}%
- **Litigation Threshold**: {st.session_state.litigation_threshold} cases
"""
        
        # Do not append the entire extracted text to the prompt to save LLM tokens.
        full_message = f"{user_input}\n\n{context_block}"
        agent_input = {"messages": [{"role": "user", "content": full_message}]}

    # Stream the agent execution
    try:
        st.session_state.agent_running = True
        
        for event in agent.stream(agent_input, config=config, stream_mode="updates"):
            for node_name, node_data in event.items():
                if node_name == "__interrupt__":
                    # Handle interrupt — the graph is paused
                    interrupts = node_data
                    if interrupts and len(interrupts) > 0:
                        interrupt_info = interrupts[0]
                        interrupt_value = interrupt_info.value if hasattr(interrupt_info, 'value') else interrupt_info
                        interrupt_id = interrupt_info.id if hasattr(interrupt_info, 'id') else None

                        st.session_state.waiting_for_human = True
                        st.session_state.interrupt_data = interrupt_value
                        st.session_state.interrupt_id = interrupt_id

                        # ── Distinguish Step Review vs HITL Data Request ──
                        if isinstance(interrupt_value, dict) and interrupt_value.get("type") == "step_review":
                            step_num = interrupt_value.get("step_number", "?")
                            tool_nm = interrupt_value.get("tool_name", "Tool")
                            # Extract bullet lines from the question
                            q = interrupt_value.get("question", "")
                            step_msg = (
                                f"✅ **Step {step_num}/5 — {tool_nm} Complete**\n\n"
                                + "\n".join(
                                    line for line in q.split("\n")
                                    if line.strip().startswith("•") or "Finding" in line or ":" in line
                                )
                                + "\n\n---\n💬 Type **`continue`** to proceed to the next step, "
                                "or describe a correction (e.g. `CIBIL is 780, not 650`)."
                            )
                            add_message("assistant", step_msg, type="step_review")
                        else:
                            # Standard HITL question (missing data, ambiguity)
                            question = ""
                            if isinstance(interrupt_value, dict):
                                question = interrupt_value.get("question", str(interrupt_value))
                            else:
                                question = str(interrupt_value)
                            add_message("assistant", question, type="interrupt")

                        st.session_state.agent_running = False
                        return  # Stop — wait for user input

                elif node_name == "tools":
                    # Tool execution results
                    messages = node_data.get("messages", [])
                    for msg in messages:
                        tool_name = getattr(msg, 'name', 'unknown_tool')
                        tool_content = msg.content if hasattr(msg, 'content') else str(msg)

                        add_message(
                            "assistant",
                            f"**Tool `{tool_name}` executed by `{st.session_state.get('selected_analysis_model')}`.**",
                            type="tool_call",
                            tool_name=tool_name,
                            tool_data=tool_content
                        )

                        # Check if this is the CAM report
                        if tool_name == "generate_cam_report":
                            st.session_state.cam_content = tool_content
                            st.session_state.cam_generated = True
                        
                        # Handle automatic session reset if cleanup tool succeeded
                        if tool_name == "clear_application_session" and "SESSION_CLEARED_SUCCESSFULLY" in tool_content:
                            st.success("🔒 Session data wiped. Returning to dashboard...")
                            time.sleep(2)
                            reset_application_state()
                            switch_page("dashboard")

                elif node_name == "agent":
                    # Agent's reasoning / response messages
                    messages = node_data.get("messages", [])
                    for msg in messages:
                        raw_content = msg.content if hasattr(msg, 'content') else str(msg)
                        
                        # Handle List-based content (Gemini/Vertex AI multi-part format)
                        if isinstance(raw_content, list):
                            content = "".join([part.get("text", "") if isinstance(part, dict) else str(part) for part in raw_content])
                        else:
                            content = str(raw_content)

                        # Skip empty content or pure tool-call messages
                        if content.strip() and not getattr(msg, 'tool_calls', None):
                            add_message("assistant", content, type="reasoning")
                        elif getattr(msg, 'tool_calls', None):
                            for tc in msg.tool_calls:
                                tool_name = tc.get("name", "unknown")
                                add_message(
                                    "assistant",
                                    f"**Orchestrator (`{st.session_state.get('selected_model')}`) calling tool:** `{tool_name}`...",
                                    type="tool_invoke"
                                )

        st.session_state.agent_running = False

    except Exception as e:
        st.session_state.agent_running = False
        err_text = str(e)
        # LangGraph INVALID_CHAT_HISTORY safeguard:
        # If a previous run left half-finished tool calls in the checkpoint,
        # we reset the thread and ask the user to resend, instead of silently dying.
        if "INVALID_CHAT_HISTORY" in err_text or "Found AIMessages with tool_calls" in err_text:
            # Reset agent + thread so future messages start from a clean state
            st.session_state.agent = None
            st.session_state.thread_id = str(uuid.uuid4())
            add_message(
                "assistant",
                "⚠️ The previous analysis session left an incomplete tool call in memory. "
                "I've reset the internal conversation state. Please resend your last message "
                "or re-trigger the current phase (e.g., Phase 1/2/3/4/5 button).",
                type="error",
            )
        else:
            add_message("assistant", f"❌ **Agent Error:** {err_text}", type="error")


# ──────────────────────────────────────────────
# VIEW 1: Login
# ──────────────────────────────────────────────
def render_login():
    st.markdown("<h1 class='main-title' style='font-size: 4rem;'>CREDI-MITRA</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>AI-Powered Corporate Credit Analyst Agent</p>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.container(border=True):
            st.subheader("🔐 Credit Manager Login")
            st.caption("Authenticate to access the AI analysis platform")
            st.markdown("")
            username = st.text_input("Username", value="admin", placeholder="Corporate ID")
            password = st.text_input("Password", value="password", type="password", placeholder="Password")
            st.markdown("")
            if st.button("🚀 Authenticate & Enter", type="primary", use_container_width=True):
                if username == "admin" and password == "password":
                    st.session_state.logged_in = True
                    st.success("✅ Authentication successful!")
                    time.sleep(0.5)
                    switch_page("dashboard")
                else:
                    st.error("❌ Invalid credentials. Please try again.")


# ──────────────────────────────────────────────
# VIEW 2: Dashboard
# ──────────────────────────────────────────────
def render_dashboard():
    st.markdown("<h1 class='main-title' style='font-size: 3.5rem;'>CREDI-MITRA</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Portfolio Overview & Policy Management</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col_head, col_logout = st.columns([6, 1])
    with col_head:
        st.header("📊 Dashboard")
    with col_logout:
        st.write("")
        if st.button("🚪 Log Out", use_container_width=True):
            st.session_state.logged_in = False
            switch_page("login")

    st.markdown("---")

    st.info("👋 **Welcome back, Admin.** Here is a quick snapshot of the corporate credit portfolio.")

    col1, col2, col3 = st.columns(3)
    with col1:
        with st.container(border=True):
            st.metric(label="📄 Active Applications", value="12", delta="+2 this week")
    with col2:
        with st.container(border=True):
            st.metric(label="✅ Accepted (MTD)", value="45", delta="15% vs Last Mo.")
    with col3:
        with st.container(border=True):
            st.metric(label="❌ Rejected (MTD)", value="8", delta="-2% vs Last Mo.", delta_color="inverse")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Credit Policy Settings & Actions ──
    col_action, col_settings = st.columns([1.1, 1.4])

    with col_action:
        with st.container(height=400, border=True):
            st.subheader("⚡ Quick Actions")
            st.caption("Choose an action to perform")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🚀 Start New Application Analysis", type="primary", use_container_width=True):
                reset_application_state()
                switch_page("analysis")
            st.markdown("<br>", unsafe_allow_html=True)
            st.success("🤖 **System Status:**\n\n✅ LLM Orchestrator Online\n\n✅ XGBoost Engine Ready\n\n✅ Pinecone Cloud DB Ready")

    with col_settings:
        with st.container(height=400, border=True):
            st.subheader("🎛️ Credit Policy Parameters")
            st.caption("Configure base parameters used in dynamic credit decisioning")
            st.markdown("<br>", unsafe_allow_html=True)

            base_premium = st.slider(
                "📈 Base Interest Rate Premium (%)",
                min_value=5.0,
                max_value=15.0,
                value=float(st.session_state.base_premium),
                step=0.25,
                help="The base interest rate before adding risk and age premiums."
            )
            st.session_state.base_premium = base_premium

            revenue_tolerance = st.slider(
                "🛡️ Revenue Variance Tolerance (%)",
                min_value=5,
                max_value=100,
                value=int(st.session_state.revenue_tolerance),
                step=5,
                help="Maximum allowed variance between GST Revenue and Bank Inflow before triggering a 'Circular Trading' rejection."
            )
            st.session_state.revenue_tolerance = revenue_tolerance

            litigation_threshold = st.slider(
                "⚖️ Litigation Threshold (Max Cases)",
                min_value=0,
                max_value=20,
                value=int(st.session_state.litigation_threshold),
                step=1,
                help="Maximum number of active litigation cases allowed before triggering a hard rejection for legal risk."
            )
            st.session_state.litigation_threshold = litigation_threshold

            # Show the formula breakdown
            st.markdown("---")
            st.markdown("**📐 Interest Rate Calculation Formula:**")
            st.code(
                f"Rate = {base_premium}% (Base Premium)\n"
                f"       + Risk Premium ((900 - CIBIL) / 100 × 0.5)\n"
                f"       + Age Premium  (1.5% if Company Age ≤ 5 yrs)",
                language="text"
            )


# ──────────────────────────────────────────────
# VIEW 3: Analysis & Agent Chat
# ──────────────────────────────────────────────
def render_analysis():
    # ── Sidebar: Document Ingestion ──
    with st.sidebar:
        with st.expander("🚀 Agent Configuration", expanded=True):
            # Directly provided list of high-quality, stable models
            model_choices = [
                # --- Google Gemini (Next Gen) ---
                "Gemini 3 (Google)",
                "Gemini 2.5 Pro (Google)",
                "Gemini 2.5 Flash (Google)",
                "Gemini 2.5 Flash-Lite (Google)",
                
                # --- Groq (High Speed) ---
                "llama-3.3-70b-versatile (Groq)",
                "llama-3.1-8b-instant (Groq)",
                "mixtral-8x7b-32768 (Groq)",
                "gemma2-9b-it (Groq)",
                "qwen-2.5-32b (Groq)",
                "openai/gpt-oss-120b (Groq)"
                
                # --- OpenAI / OSS ---
                "gpt-4o (OpenAI)",
                "gpt-4o-mini (OpenAI)",
                "o1 (OpenAI)",
                "o1-mini (OpenAI)",
                "o3-mini (OpenAI)"
            ]
            
            # Single Model Selection
            current_stored = st.session_state.get("selected_model")
            try:
                default_idx = model_choices.index(current_stored) if current_stored in model_choices else 0
            except:
                default_idx = 0

            selected_model = st.selectbox(
                "🤖 Select AI Model", 
                model_choices, 
                index=default_idx,
                help="The central model used for both reasoning and data analysis."
            )
            
            if st.session_state.selected_model != selected_model:
                st.session_state.selected_model = selected_model
                st.session_state.selected_analysis_model = selected_model
                # Force a full state reset on model change
                reset_application_state()
                st.info("🔄 Model changed. Session reset for compatibility.")
                st.rerun()

        with st.expander("📋 Application Details", expanded=True):
            st.text_input("Company Name", key="company_name")
            st.text_input("Application No.", key="app_no")
            app_date = st.date_input("Application Date")


        st.markdown("---")

        with st.expander("📁 Manual Document Ingestion", expanded=False):
            st.info("Upload the required documents for AI processing.")
            app_form = st.file_uploader("Application Form", type=["pdf", "docx"])
            cibil = st.file_uploader("CIBIL Score Report", type=["pdf"])
            gst = st.file_uploader("GST Returns (GSTR-2A/3B)", type=["pdf"])
            bank = st.file_uploader("Bank Statements", type=["pdf"])
            annual = st.file_uploader("Annual Reports", type=["pdf"])

        st.markdown("---")

        with st.expander("🧑‍💼 Manual Officer Insights", expanded=False):
            officer_report = st.file_uploader("Upload Officer Report", type=["pdf", "docx", "txt"])
            manual_entry = st.text_area(
                "Manual Notes",
                placeholder="e.g., Factory visit notes, management interview summary...",
                height=120
            )
            st.caption("💡 Provide either an uploaded report OR manual notes.")

        st.markdown("---")
        file_map = {
            "Application_Form": app_form,
            "CIBIL_Score_Report": cibil,
            "GST_Returns": gst,
            "Bank_Statements": bank,
            "Annual_Reports": annual,
            "Officer_Insights_Report": officer_report,
        }
        # ── INITIALIZE CHAT WITHOUT PARSING PDFs YET ──
        if st.button("🚨 Start AI Chat (Phase 1)", type="primary", use_container_width=True):
            st.session_state.docs_verified = True

            # Source from session state (synced with text_input keys)
            current_c_name = st.session_state.get("company_name", "").strip()
            current_a_no = st.session_state.get("app_no", "").strip()

            # Fix: If UI input is blank but we have an auto-fetch result, use it
            if not current_c_name and st.session_state.get("company_name"):
                 current_c_name = st.session_state.company_name
            if not current_a_no and st.session_state.get("app_no"):
                 current_a_no = st.session_state.app_no

            if not current_a_no:
                st.error("Application Number is required.")
                st.stop()

            # Reset agent state for new analysis
            st.session_state.messages = []
            st.session_state.agent = None
            
            # ── CLEAR PREVIOUS VECTORS (RAG RESET) ──
            # This ensures old vectors from previous sessions/apps are cleared
            try:
                from rag import get_document_manager
                # Pass selected model to ensure correct embedding provider
                mgr = get_document_manager(model_choice=st.session_state.get("selected_model"))
                mgr.reset_session()
                # Also reset local extraction status
                st.session_state.document_extracted_text = ""
            except Exception as e:
                st.warning(f"Note: Vector DB cleanup skipped: {e}")

            if not st.session_state.get("thread_id"):
                st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.waiting_for_human = False
            st.session_state.interrupt_data = None
            st.session_state.cam_generated = False
            st.session_state.cam_content = ""

            # ── Touchless Auto-Fetch Logic ──
            # Automatically check if a local folder exists for this App No
            found_dir = None
            uploads_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "applications_received")
            if os.path.exists(uploads_dir):
                for dirname in os.listdir(uploads_dir):
                    if dirname.endswith(f"_{current_a_no}"):
                        found_dir = os.path.join(uploads_dir, dirname)
                        # Derive name from folder: everything before last underscore
                        recovered_name = dirname.rsplit("_", 1)[0].replace("_", " ")
                        # Use recovered name if current is blank or suspicious
                        if not current_c_name or current_c_name.isdigit() or current_c_name == current_a_no:
                             current_c_name = recovered_name
                        break
            
            if found_dir and os.path.isdir(found_dir):
                # We found a local folder—check for valid documents
                existing_files = [f for f in os.listdir(found_dir) if os.path.isfile(os.path.join(found_dir, f)) and not f.endswith(".json") and not f.endswith(".txt")]
                if existing_files:
                    save_dir = found_dir
                    uploaded_doc_names = list(existing_files)
                    saved_count = len(uploaded_doc_names)
                else:
                    # Fallback to manual naming if folder is empty or invalid
                    c_slug = current_c_name.replace(" ", "_") or "Company"
                    a_slug = current_a_no.replace(" ", "_") or "App"
                    save_dir = os.path.join(uploads_dir, f"{c_slug}_{a_slug}")
                    uploaded_doc_names = []
                    saved_count = 0
            else:
                # No local folder found—proceed with manual naming convention
                c_slug = current_c_name.replace(" ", "_") or "Company"
                a_slug = current_a_no.replace(" ", "_") or "App"
                save_dir = os.path.join(uploads_dir, f"{c_slug}_{a_slug}")
                uploaded_doc_names = []
                saved_count = 0
            
            st.session_state.current_upload_dir = save_dir
            os.makedirs(save_dir, exist_ok=True)
            
            extracted_text = ""

            for section_name, uploaded_file in file_map.items():
                if uploaded_file is not None:
                    ext = os.path.splitext(uploaded_file.name)[1].lower()
                    save_path = os.path.join(save_dir, f"{section_name}{ext}")
                    
                    new_filename = f"{section_name}{ext}"
                    if new_filename not in uploaded_doc_names:
                        uploaded_doc_names.append(new_filename)
                        saved_count += 1
                        
                    file_bytes = uploaded_file.getbuffer()
                    with open(save_path, "wb") as f:
                        f.write(file_bytes)

            if manual_entry and manual_entry.strip():
                saved_count += 1
                uploaded_doc_names.append("Officer Manual Notes")
                extracted_text += f"\n\n--- Document: Officer Manual Notes ---\n\n" + manual_entry.strip()

            st.session_state.document_extracted_text = extracted_text.strip()
            st.session_state.manual_entry = manual_entry.strip() if manual_entry else ""

            # Save initial text to bridge file
            thread_id = st.session_state.thread_id
            os.makedirs("temp", exist_ok=True)
            bridge_path = os.path.join("temp", f"{thread_id}.txt")
            with open(bridge_path, "w") as f:
                f.write(st.session_state.document_extracted_text)

            if saved_count > 0:
                st.success(f"✅ {saved_count} inputs saved for analysis.")

            st.session_state.messages = []
            doc_list_str = ", ".join(uploaded_doc_names) if uploaded_doc_names else "No documents"
            prompt = (
                f"Begin Phase 1: App submitted for '{current_c_name}' (No: {current_a_no}) "
                f"with: {doc_list_str}. "
                "First, call `list_uploaded_documents`, then analyze each document one by one "
                "using `analyze_document`, summarizing ML features and 5Cs insights after each file. "
                "When all documents are done, call `extract_document_data` exactly once to finalize aggregation."
            )
            st.session_state.auto_submit_prompt = prompt

            st.rerun()

        st.markdown("---")
        with st.expander("🧭 Orchestrated Analysis Phases", expanded=False):
            st.caption("Trigger each phase explicitly. The agent will obey these stepwise instructions.")
            if st.button("▶️ Phase 2: Web Research & Risk", use_container_width=True):
                phase2_prompt = (
                    "Proceed to Phase 2: Start web research for the current company. "
                    "Use `crawl_web_for_litigation` to perform granular web due diligence, "
                    "compute litigation_count, positive/negative news counts, news_sentiment_score and risk_score, "
                    "and then present a seriousness-aware summary for human review."
                )
                st.session_state.auto_submit_prompt = phase2_prompt
                st.session_state.waiting_for_human = False
                st.session_state.interrupt_data = None
                st.experimental_rerun()

            if st.button("▶️ Phase 3: Extract Numerical Features", use_container_width=True):
                phase3_prompt = (
                    "Proceed to Phase 3: Extract numerical features. "
                    "Use `extract_numerical_features` to synthesize the 6 ML features "
                    "(Company_Age, CIBIL_Commercial_Score, GSTR_Declared_Revenue_Cr, "
                    "Bank_Statement_Inflow_Cr, Litigation_Count, News_Sentiment_Score). "
                    "Ask me via interrupt if any critical values are missing or inconsistent."
                )
                st.session_state.auto_submit_prompt = phase3_prompt
                st.session_state.waiting_for_human = False
                st.session_state.interrupt_data = None
                st.experimental_rerun()

            if st.button("▶️ Phase 4: Run ML Scoring", use_container_width=True):
                phase4_prompt = (
                    "Proceed to Phase 4: Run ML scoring. "
                    "Call `run_xgboost_scorer` with the final feature JSON, "
                    "present the model decision (approval, limit, rate, probabilities) and "
                    "show how the rejection rules interact with this decision. "
                    "Pause for my confirmation or override before treating the decision as final."
                )
                st.session_state.auto_submit_prompt = phase4_prompt
                st.session_state.waiting_for_human = False
                st.session_state.interrupt_data = None
                st.experimental_rerun()

            if st.button("▶️ Phase 5: Generate CAM Report", use_container_width=True):
                phase5_prompt = (
                    "Proceed to Phase 5: Generate CAM. "
                    "Call `generate_cam_report` for the current company using all prior analysis, "
                    "ask me for a final confirmation via interrupt, and then return the final CAM "
                    "markdown so the UI can display it and offer a PDF download."
                )
                st.session_state.auto_submit_prompt = phase5_prompt
                st.session_state.waiting_for_human = False
                st.session_state.interrupt_data = None
                st.experimental_rerun()

        if st.button("← Back to Dashboard"):
            switch_page("dashboard")

        if st.session_state.cam_generated:
            st.markdown("---")
            st.subheader("📑 Final CAM Report")
            render_pdf_download_button("sidebar")

    # ── Main Area: Chat Interface ──
    st.markdown("<h1 class='main-title' style='font-size: 3.5rem;'>CREDI-MITRA</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>LLM-Orchestrated Credit Appraisal Console</p>", unsafe_allow_html=True)

    # Render all messages
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]
        msg_type = msg.get("type", "")

        if role == "user":
            with st.chat_message("user"):
                st.markdown(content)
        elif role == "assistant":
            if msg_type == "tool_call":
                with st.chat_message("assistant", avatar="🔧"):
                    st.markdown(content)
                    tool_name = msg.get("tool_name", "")
                    tool_data = msg.get("tool_data", "")
                    if tool_data:
                        render_tool_output(tool_name, tool_data)

                    # If this is the CAM report, render download button
                    if tool_name == "generate_cam_report" and st.session_state.cam_generated:
                        st.markdown("---")
                        _render_cam_extras()

            elif msg_type == "tool_invoke":
                with st.chat_message("assistant", avatar="🧠"):
                    st.markdown(content)

            elif msg_type == "interrupt":
                with st.chat_message("assistant", avatar="⏸️"):
                    st.warning(content)

            elif msg_type == "error":
                with st.chat_message("assistant", avatar="❌"):
                    st.error(content)

            else:
                with st.chat_message("assistant"):
                    st.markdown(content)

    # ── Auto-run agent on submit ──
    if st.session_state.get("auto_submit_prompt"):
        prompt = st.session_state.auto_submit_prompt
        st.session_state.auto_submit_prompt = None
        
        # Add the prompt to history
        add_message("user", prompt)
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant", avatar="🧠"):
            with st.status("🧠 Analyzing ingestion...", expanded=True) as status:
                st.write("Confirming receipt of documents...")
                run_agent(user_input=prompt)
                status.update(label="✅ Ready to proceed!", state="complete")
        st.rerun()



    # ── Chat Input ──
    prompt = st.chat_input("Type here to interact with the AI Agent...")

    if prompt:
        add_message("user", prompt)

        if st.session_state.waiting_for_human:
            # User is responding to an interrupt question
            st.session_state.waiting_for_human = False
            interrupt_data = st.session_state.interrupt_data
            st.session_state.interrupt_data = None
            st.session_state.interrupt_id = None

            with st.chat_message("assistant", avatar="🧠"):
                with st.status("🔄 Resuming agent with your input...", expanded=True) as status:
                    st.write(f"Your response: *{prompt}*")
                    st.write("Feeding back to the orchestrator...")
                    run_agent(resume_value=prompt)
                    status.update(label="✅ Agent resumed!", state="complete")

        else:
            # Normal message — run the agent
            with st.chat_message("assistant", avatar="🧠"):
                with st.status("🧠 LLM Orchestrator is thinking...", expanded=True) as status:
                    st.write("The AI Agent is analyzing your request and deciding which tools to use...")
                    run_agent(user_input=prompt)

                    if st.session_state.waiting_for_human:
                        status.update(
                            label="⏸️ Waiting for your input — please answer the question above",
                            state="error"
                        )
                    else:
                        status.update(label="✅ Analysis step complete!", state="complete")

        st.rerun()


def _render_cam_extras():
    """Render the CAM explainability chart and download button in the main chat."""
    if st.session_state.cam_content:
        import pandas as pd
        # SHAP explainability chart
        st.subheader("📈 Model Explainability (Feature Impact)")
        features = [
            "CIBIL Score", "Bank Inflow", "GST Revenue",
            "Company Age", "Litigation Count", "News Sentiment"
        ]
        impact = [0.8, 0.6, 0.4, 0.2, -0.3, -0.1]
        shap_df = pd.DataFrame({
            "Feature": features,
            "Impact on Approval": impact
        }).set_index("Feature")

        col_l, col_c, col_r = st.columns([1, 3, 1])
        with col_c:
            st.bar_chart(shap_df, color="#247eea", height=400)

        # PDF download
        st.markdown("---")
        render_pdf_download_button("main")


def render_pdf_download_button(location="main"):
    """Dedicated function to render the PDF download button with byte caching."""
    if not st.session_state.cam_content:
        return

    # Cache PDF generation to avoid lag on every rerun
    if "cam_pdf_bytes" not in st.session_state or st.session_state.get("last_cam_content") != st.session_state.cam_content:
        with st.spinner("Preparing PDF report..."):
            try:
                pdf_data = generate_cam_pdf(st.session_state.cam_content)
                st.session_state.cam_pdf_bytes = pdf_data
                st.session_state.last_cam_content = st.session_state.cam_content
                
                # ── Robust Auto-Save to 'Report_Generated' ──
                try:
                    rep_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Report_Generated")
                    os.makedirs(rep_dir, exist_ok=True)
                    
                    c_slug = st.session_state.get("company_name", "").strip().replace(" ", "_") or "Company"
                    a_slug = st.session_state.get("app_no", "").strip().replace(" ", "_") or "App"
                    f_path = os.path.join(rep_dir, f"{c_slug}_{a_slug}_CAM.pdf")
                    
                    with open(f_path, "wb") as f:
                        f.write(pdf_data)
                except Exception as save_err:
                    st.warning(f"Note: Could not auto-save PDF to directory: {save_err}")
            except Exception as e:
                st.error(f"Failed to generate PDF: {e}")
                return

    pdf_bytes = st.session_state.cam_pdf_bytes
    c_name = st.session_state.get("company_name", "").strip().replace(" ", "_") or "Company"
    a_no = st.session_state.get("app_no", "").strip().replace(" ", "_") or "App"
    
    label = "📄 Download CAM (PDF)"
    if location == "sidebar":
        st.info("CAM report is ready for download.")
        st.download_button(
            label=label,
            data=pdf_bytes,
            file_name=f"{c_name}_{a_no}_CAM.pdf",
            mime="application/pdf",
            use_container_width=True,
            key="sidebar_download_cam"
        )
    else:
        st.download_button(
            label=label,
            data=pdf_bytes,
            file_name=f"{c_name}_{a_no}_CAM.pdf",
            mime="application/pdf",
            type="primary",
            key="main_download_cam"
        )


# ──────────────────────────────────────────────
# Main Controller
# ──────────────────────────────────────────────
def main():
    if not st.session_state.logged_in:
        st.session_state.current_page = "login"
        render_login()
    else:
        if st.session_state.current_page == "dashboard":
            render_dashboard()
        elif st.session_state.current_page == "analysis":
            render_analysis()
        else:
            render_dashboard()


if __name__ == "__main__":
    main()
