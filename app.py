"""
CREDI-MITRA Streamlit Application — LLM Agent Chat Interface

Features:
- Continuous chat interface with st.chat_message
- Intermediate tool visibility (every tool call shown in expandable sections)
- Human-in-the-Loop via LangGraph interrupt / Command(resume=...)
- Session state memory for conversation persistence across reruns
"""

import streamlit as st
import time
import json
import uuid
import os
import io
import re
import pandas as pd
import numpy as np
from fpdf import FPDF
from dotenv import load_dotenv
from pypdf import PdfReader
from langgraph.types import interrupt, Command

load_dotenv()

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
        "agent_running": False,
        "base_premium": 8.5,         # Base interest rate premium (%)
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


def add_message(role, content, **kwargs):
    """Add a message to session state."""
    msg = {"role": role, "content": content}
    msg.update(kwargs)
    st.session_state.messages.append(msg)


def render_tool_output(tool_name, tool_data):
    """Render a tool's output as an expandable section in the chat."""
    icon_map = {
        "extract_pdf_data": "📄",
        "crawl_web_for_litigation": "🔍",
        "extract_numerical_features": "📊",
        "run_xgboost_scorer": "🤖",
        "generate_cam_report": "📋",
    }
    icon = icon_map.get(tool_name, "🔧")

    with st.expander(f"{icon} Tool Output: `{tool_name}`", expanded=True):
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


def generate_cam_pdf(cam_text):
    """Convert CAM markdown content into a professionally formatted PDF."""
    cam_text = re.sub(r'[^\x00-\xff]', '', cam_text)
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(102, 126, 234)
    pdf.cell(0, 14, "CREDI-MITRA", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, "AI-Powered Credit Intelligence", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(4)
    pdf.set_draw_color(102, 126, 234)
    pdf.set_line_width(0.6)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(6)

    for raw_line in cam_text.strip().splitlines():
        line = raw_line.strip()
        if not line:
            pdf.ln(3)
            continue
        if line.startswith("####") or line.startswith("###") or line.startswith("##") or line.startswith("# "):
            heading = line.lstrip("#").strip().replace("**", "")
            pdf.set_font("Helvetica", "B", 13)
            pdf.set_text_color(60, 80, 160)
            pdf.multi_cell(0, 8, heading)
            pdf.ln(2)
        elif line == "---":
            pdf.ln(2)
            pdf.set_draw_color(180, 180, 180)
            pdf.set_line_width(0.3)
            pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
            pdf.ln(4)
        elif line.startswith("- ") or line.startswith("* "):
            text = line[2:].replace("**", "")
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(50, 50, 50)
            pdf.set_x(pdf.l_margin + 4)
            pdf.multi_cell(pdf.w - pdf.l_margin - pdf.r_margin - 4, 6, "  " + text)
            pdf.ln(1)
        elif line.startswith("|"):
            text = line.replace("**", "")
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(70, 70, 70)
            pdf.multi_cell(0, 5, text)
        else:
            text = line.replace("**", "")
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(50, 50, 50)
            pdf.multi_cell(0, 6, text)
            pdf.ln(1)

    return bytes(pdf.output())


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
    if st.session_state.agent is None:
        agent, checkpointer = build_agent()
        st.session_state.agent = agent
        st.session_state.thread_id = str(uuid.uuid4())

    agent = st.session_state.agent
    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    # Prepare input
    if resume_value is not None:
        # Resuming from an interrupt
        agent_input = Command(resume=resume_value)
    else:
        # Build the context message for the agent
        base_premium = st.session_state.get("base_premium", 8.5)
        context = (
            f"Company Name: {st.session_state.company_name}\n"
            f"Application No: {st.session_state.app_no}\n"
            f"Officer Insights: {st.session_state.manual_entry or 'None provided'}\n"
            f"PDF Text Available: {'Yes' if st.session_state.pdf_extracted_text else 'No'}\n"
            f"Base Interest Rate Premium: {base_premium}%\n"
        )
        if st.session_state.pdf_extracted_text:
            # Truncate for the message but full text goes to the tool
            context += f"PDF Text Length: {len(st.session_state.pdf_extracted_text)} characters\n"

        full_message = f"{user_input}\n\nContext:\n{context}"
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

                        st.session_state.waiting_for_human = True
                        st.session_state.interrupt_data = interrupt_value

                        # Display the question from the tool
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
                            f"🔧 **Tool `{tool_name}` executed.**",
                            type="tool_call",
                            tool_name=tool_name,
                            tool_data=tool_content
                        )

                        # Check if this is the CAM report
                        if tool_name == "generate_cam_report":
                            st.session_state.cam_content = tool_content
                            st.session_state.cam_generated = True

                elif node_name == "agent":
                    # Agent's reasoning / response messages
                    messages = node_data.get("messages", [])
                    for msg in messages:
                        content = msg.content if hasattr(msg, 'content') else str(msg)
                        # Skip empty content or pure tool-call messages
                        if content and not getattr(msg, 'tool_calls', None):
                            add_message("assistant", content, type="reasoning")
                        elif getattr(msg, 'tool_calls', None):
                            for tc in msg.tool_calls:
                                tool_name = tc.get("name", "unknown")
                                add_message(
                                    "assistant",
                                    f"🧠 **Calling tool:** `{tool_name}`...",
                                    type="tool_invoke"
                                )

        st.session_state.agent_running = False

    except Exception as e:
        st.session_state.agent_running = False
        add_message("assistant", f"❌ **Agent Error:** {str(e)}", type="error")


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
            st.caption("Initiate a multi-agent AI analysis powered by LLM Orchestrator")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🚀 Start New Application Analysis", type="primary", use_container_width=True):
                switch_page("analysis")
            st.markdown("<br>", unsafe_allow_html=True)
            st.success("🤖 **System Status:**\n\n✅ LLM Orchestrator Online\n\n✅ XGBoost Engine Ready")

    with col_settings:
        with st.container(height=400, border=True):
            st.subheader("🎛️ Credit Policy Parameters")
            st.caption("Configure base parameters used in dynamic credit decisioning")
            st.markdown("<br>", unsafe_allow_html=True)

            base_premium = st.slider(
                "📈 Base Interest Rate Premium (%)",
                min_value=5.0,
                max_value=15.0,
                value=st.session_state.base_premium,
                step=0.25,
                help="The base interest rate before adding risk and age premiums."
            )
            st.session_state.base_premium = base_premium

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
        with st.expander("📋 Application Details", expanded=True):
            company_name = st.text_input("Company Name")
            app_no = st.text_input("Application No.")
            app_date = st.date_input("Application Date")
            if st.button("🔄 Auto-Fetch (Coming Soon)"):
                st.info("Integration pending — will connect to core banking system.")

        st.markdown("---")

        with st.expander("📁 Document Ingestion", expanded=True):
            st.info("Upload the required documents for AI processing.")
            app_form = st.file_uploader("Application Form", type=["pdf", "docx"])
            cibil = st.file_uploader("CIBIL Score Report", type=["pdf"])
            gst = st.file_uploader("GST Returns (GSTR-2A/3B)", type=["pdf", "csv", "xlsx"])
            bank = st.file_uploader("Bank Statements", type=["pdf", "csv", "xlsx"])
            annual = st.file_uploader("Annual Reports", type=["pdf"])

        st.markdown("---")

        with st.expander("🧑‍💼 Officer Insights", expanded=True):
            officer_report = st.file_uploader("Upload Officer Report", type=["pdf", "docx", "txt"])
            manual_entry = st.text_area(
                "Manual Notes",
                placeholder="e.g., Factory visit notes, management interview summary...",
                height=120
            )
            st.caption("💡 Provide either an uploaded report OR manual notes.")

        st.markdown("---")

        # Submit button
        if st.button("🚨 Submit to Agent", type="primary", use_container_width=True):
            st.session_state.docs_verified = True
            st.session_state.company_name = company_name
            st.session_state.app_no = app_no

            # Reset agent state for new analysis
            st.session_state.messages = []
            st.session_state.agent = None
            st.session_state.thread_id = None
            st.session_state.waiting_for_human = False
            st.session_state.interrupt_data = None
            st.session_state.cam_generated = False
            st.session_state.cam_content = ""

            # Save uploaded files
            c_name = company_name.strip().replace(" ", "_") or "Company"
            a_no = app_no.strip().replace(" ", "_") or "App"
            folder_name = f"{c_name}_{a_no}"
            save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads", folder_name)
            os.makedirs(save_dir, exist_ok=True)

            file_map = {
                "Application_Form": app_form,
                "CIBIL_Score_Report": cibil,
                "GST_Returns": gst,
                "Bank_Statements": bank,
                "Annual_Reports": annual,
                "Officer_Insights_Report": officer_report,
            }

            saved_count = 0
            extracted_text = ""
            uploaded_doc_names = []
            for section_name, uploaded_file in file_map.items():
                if uploaded_file is not None:
                    uploaded_doc_names.append(section_name.replace("_", " "))
                    ext = os.path.splitext(uploaded_file.name)[1]
                    save_path = os.path.join(save_dir, f"{section_name}{ext}")
                    file_bytes = uploaded_file.getbuffer()
                    with open(save_path, "wb") as f:
                        f.write(file_bytes)
                    if ext.lower() == ".pdf":
                        try:
                            pdf_reader = PdfReader(io.BytesIO(file_bytes))
                            text = "".join([page.extract_text() or "" for page in pdf_reader.pages])
                            extracted_text += f"\n\n--- Document: {section_name} (PDF) ---\n\n" + text
                        except Exception as e:
                            st.error(f"Failed to read {section_name}: {e}")
                    elif ext.lower() == ".csv":
                        try:
                            df = pd.read_csv(io.BytesIO(file_bytes))
                            text = df.to_string(index=False)
                            extracted_text += f"\n\n--- Document: {section_name} (CSV) ---\n\n" + text
                        except Exception as e:
                            st.error(f"Failed to read {section_name}: {e}")
                    elif ext.lower() in [".xlsx", ".xls"]:
                        try:
                            df = pd.read_excel(io.BytesIO(file_bytes))
                            text = df.to_string(index=False)
                            extracted_text += f"\n\n--- Document: {section_name} (Excel) ---\n\n" + text
                        except Exception as e:
                            st.error(f"Failed to read {section_name}: {e}")
                    saved_count += 1

            st.session_state.pdf_extracted_text = extracted_text
            st.session_state.manual_entry = manual_entry.strip() if manual_entry else ""

            if manual_entry and manual_entry.strip():
                save_path = os.path.join(save_dir, "Officer_Manual_Notes.txt")
                with open(save_path, "w") as f:
                    f.write(manual_entry.strip())
                saved_count += 1
                uploaded_doc_names.append("Officer Manual Notes")

            if saved_count > 0:
                st.success(f"✅ {saved_count} document(s) uploaded and extracted.")

            # Automatically trigger the agent to acknowledge receipt instead of fake message
            st.session_state.messages = []
            doc_list_str = ", ".join(uploaded_doc_names) if uploaded_doc_names else "No documents"
            prompt = f"I have submitted the application for '{company_name}' (App No: {app_no}). I uploaded the following documents: {doc_list_str}. \n\nPlease confirm receipt of these documents, list them back to me, and ask if you should proceed with the analysis."
            st.session_state.auto_submit_prompt = prompt

            st.rerun()

        st.markdown("---")
        if st.button("← Back to Dashboard"):
            switch_page("dashboard")

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
    if prompt := st.chat_input("Type here to interact with the AI Agent..."):
        add_message("user", prompt)

        if st.session_state.waiting_for_human:
            # User is responding to an interrupt question
            st.session_state.waiting_for_human = False
            interrupt_data = st.session_state.interrupt_data
            st.session_state.interrupt_data = None

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
    """Render the CAM download button and explainability chart."""
    if st.session_state.cam_content:
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
            st.bar_chart(shap_df, color="#667eea", height=400)

        # PDF download
        pdf_bytes = generate_cam_pdf(st.session_state.cam_content)
        c_name = st.session_state.get("company_name", "").strip().replace(" ", "_") or "Company"
        a_no = st.session_state.get("app_no", "").strip().replace(" ", "_") or "App"
        st.download_button(
            label="📄 Download CAM Report (PDF)",
            data=pdf_bytes,
            file_name=f"{c_name}_{a_no}_CAM.pdf",
            mime="application/pdf",
            type="primary"
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
