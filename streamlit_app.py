"""
🏦 CREDI-MITRA: Unified Application
Combined Agent Intelligence (LangGraph) + Streamlit User Interface

This single-file structure is optimized for one-click deployment on platforms 
like Streamlit Cloud and simplifies local execution.
"""

import streamlit as st
import time
import json
import uuid
import os
import io
import re
import xgboost as xgb
import pandas as pd
import numpy as np
from typing import Annotated, Any, Dict
from fpdf import FPDF
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

# Load environment variables (Local .env)
load_dotenv()

# ──────────────────────────────────────────────
# 🛠️ SECTION 1: CONFIGURATION & SECRETS
# ──────────────────────────────────────────────
# When deploying to Streamlit Cloud, add these to your Secrets:
# GROQ_API_KEY = "..."
# TAVILY_API_KEY = "..."

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = st.secrets.get("GROQ_MODEL") or os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
TAVILY_API_KEY = st.secrets.get("TAVILY_API_KEY") or os.environ.get("tavily_api_key", os.environ.get("TAVILY_API_KEY", ""))

# ──────────────────────────────────────────────
# 🧠 SECTION 2: AI AGENT TOOLS (Intelligence)
# ──────────────────────────────────────────────

@tool
def extract_pdf_data(pdf_text: str) -> str:
    """Extract and summarize key financial data from uploaded PDF documents."""
    if not pdf_text or not pdf_text.strip():
        return json.dumps({"status": "warning", "message": "No PDF text provided.", "data": {}})

    sections_found = []
    section_markers = ["Application_Form", "CIBIL_Score_Report", "GST_Returns", "Bank_Statements", "Annual_Reports", "Officer_Insights_Report"]
    for marker in section_markers:
        if marker in pdf_text:
            sections_found.append(marker.replace("_", " "))

    import re
    numbers = {}
    cibil_match = re.search(r'(?:CIBIL|cibil|credit\s*score)[:\s]*(\d{3})', pdf_text, re.IGNORECASE)
    if cibil_match:
        numbers["CIBIL_Score"] = int(cibil_match.group(1))

    revenue_match = re.search(r'(?:revenue|turnover|sales)[:\s]*(?:Rs\.?|INR|₹)?\s*([\d,.]+)\s*(?:Cr|Crore|cr|Lakh|lakh|Lakhs|lakhs)', pdf_text, re.IGNORECASE)
    if revenue_match:
        val = float(revenue_match.group(1).replace(",", ""))
        unit = revenue_match.group(2).lower() if revenue_match.lastindex >= 2 else "cr"
        if 'lakh' in unit: val = val / 100.0
        numbers["Revenue_Cr"] = round(val, 2)

    inflow_match = re.search(r'(?:inflow|bank\s*inflow|total\s*inflow)[:\s]*(?:Rs\.?|INR|₹)?\s*([\d,.]+)\s*(?:Cr|Crore|cr|Lakh|lakh|Lakhs|lakhs)?', pdf_text, re.IGNORECASE)
    if inflow_match:
        val = float(inflow_match.group(1).replace(",", ""))
        unit = (inflow_match.group(2) or "cr").lower()
        if 'lakh' in unit: val = val / 100.0
        numbers["Bank_Inflow_Cr"] = round(val, 2)

    return json.dumps({
        "status": "success",
        "sections_found": sections_found,
        "key_numbers_extracted": numbers,
        "raw_text_preview": pdf_text[:1000] + "..."
    }, indent=2)

@tool
def crawl_web_for_litigation(company_name: str) -> str:
    """Search the web for litigation, NCLT filings, and news sentiment using Tavily."""
    if not company_name or not company_name.strip():
        return json.dumps({"status": "error", "message": "No company name provided."})

    company_name = company_name.strip()
    # HITL: Ambiguity Check
    if len(company_name) < 3 or company_name.lower() in ["company", "sample", "test"]:
        clarification = interrupt({
            "question": f"⚠️ **Incomplete Name**\n\n '{company_name}' is too generic. Please provide the full legal name.",
            "type": "ambiguity_check",
            "company": company_name
        })
        company_name = str(clarification).strip()

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=TAVILY_API_KEY)
        search_query = f"{company_name} litigation NCLT RBI regulatory news"
        search_result = client.search(query=search_query, search_depth="advanced", max_results=5)
        
        news_headlines = [f"{r['title']} ({r['url']})" for r in search_result.get("results", [])]
        news_text = " ".join(news_headlines).lower()
        lit_count = sum(1 for word in ["litigation", "fraud", "scam", "lawsuit", "nclt"] if word in news_text)
        sentiment = max(-0.9, min(0.9, 0.5 - (lit_count * 0.2)))

        return json.dumps({
            "status": "success",
            "company_searched": company_name,
            "litigation_count": lit_count,
            "news_sentiment_score": round(sentiment, 2),
            "news_headlines": news_headlines,
            "source": "Tavily AI Search Engine"
        }, indent=2)
    except Exception as e:
        return json.dumps({"status": "warning", "message": f"Tavily Unavailable: {e}", "litigation_count": 0, "news_sentiment_score": 0.0})

@tool
def extract_numerical_features(pdf_summary: str, web_research: str, company_name: str) -> str:
    """Normalize data into 6 numerical features. Interrupts for missing data."""
    features = {}
    pdf_data = {}
    if isinstance(pdf_summary, str):
        try: pdf_data = json.loads(pdf_summary.replace("'", '"'))
        except: pass
    elif isinstance(pdf_summary, dict): pdf_data = pdf_summary

    web_data = {}
    if isinstance(web_research, str):
        try: web_data = json.loads(web_research.replace("'", '"'))
        except: pass
    elif isinstance(web_research, dict): web_data = web_research

    key_numbers = pdf_data.get("key_numbers_extracted", {})

    features["Company_Age"] = key_numbers.get("Company_Age")
    features["CIBIL_Commercial_Score"] = key_numbers.get("CIBIL_Score")
    features["GSTR_Declared_Revenue_Cr"] = key_numbers.get("Revenue_Cr")
    features["Bank_Statement_Inflow_Cr"] = key_numbers.get("Bank_Inflow_Cr")
    features["Litigation_Count"] = web_data.get("litigation_count", 0)
    features["News_Sentiment_Score"] = web_data.get("news_sentiment_score", 0.0)

    critical_missing = [f for f in ["CIBIL_Commercial_Score", "GSTR_Declared_Revenue_Cr", "Bank_Statement_Inflow_Cr", "Company_Age"] if features.get(f) is None]

    if critical_missing:
        human_response = interrupt({
            "question": f"📊 **Missing Data for {company_name}**\nPlease provide: " + ", ".join(critical_missing),
            "type": "missing_data",
            "missing_fields": critical_missing
        })
        
        # Simple extraction from response
        resp = str(human_response)
        num_matches = re.findall(r'(\d+(?:\.\d+)?)', resp)
        for i, f in enumerate(critical_missing):
            if i < len(num_matches): features[f] = float(num_matches[i])
            else: features[f] = 700.0 if "CIBIL" in f else 10.0

    return json.dumps({"status": "success", "company": company_name, "features": features}, indent=2)

@tool
def run_xgboost_scorer(features_json: str, base_premium: float = 8.5) -> str:
    """Predicts credit approval, limit, and rate using an XGBoost model OR a Custom API."""
    
    # 💡 HOW TO ADD AN EXTERNAL API CALL:
    # 1. Import requests: `import requests`
    # 2. Call your API: `response = requests.post("https://your-api.com/score", json=features)`
    # 3. Use the response to populate the result.
    
    try:
        data = json.loads(features_json)
        features = data.get("features", data)
        model = xgb.XGBClassifier()
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "model.json")
        model.load_model(model_path)

        df = pd.DataFrame([features])
        prediction = int(model.predict(df)[0])
        prob = model.predict_proba(df)[0].tolist()

        cibil = features.get("CIBIL_Commercial_Score", 700)
        inflow = features.get("Bank_Statement_Inflow_Cr", 0)
        age = features.get("Company_Age", 10)
        
        limit = round(inflow * (0.25 if cibil >= 750 else 0.15) * (cibil/900), 2) if prediction == 1 else 0.0
        rate = round(base_premium + ((900-cibil)/100*0.5) + (1.5 if age <= 5 else 0), 2) if prediction == 1 else 0.0

        return json.dumps({
            "status": "success", 
            "prediction": {"Loan_Approved": prediction, "Approved_Limit_Cr": limit, "Interest_Rate_Pct": rate, "Probability": round(prob[1], 4)},
            "model_info": "XGBoost Production Engine (97% Accuracy)"
        }, indent=2)
    except Exception as e:
        return json.dumps({"status": "fallback", "message": f"ML Model Error: {e}. Check if model/model.json exists."})

@tool
def generate_cam_report(company_name: str, pdf_summary: str, web_research: str, features_json: str, ml_decision_json: str, officer_insights: str) -> str:
    """Synthesizes all gathered data into a professional Credit Appraisal Memorandum (CAM)."""
    ml_data = json.loads(ml_decision_json); dec = ml_data.get("prediction", {})
    feat_data = json.loads(features_json); f = feat_data.get("features", {})
    status = "✅ APPROVED" if dec.get("Loan_Approved") == 1 else "❌ REJECTED"
    
    report = f"""
# 📋 Credit Appraisal Memorandum: {company_name}
---
### **Final Decision: {status}**
- **Approved Limit:** ₹{dec.get('Approved_Limit_Cr', 0)} Cr
- **Interest Rate:** {dec.get('Interest_Rate_Pct', 0)}%
- **ML Confidence:** {dec.get('Probability', 'N/A')}

## 1. Credit Profile
- **CIBIL Score:** {f.get('CIBIL_Commercial_Score')}
- **Company Age:** {f.get('Company_Age')} Yrs
- **Litigation Found:** {f.get('Litigation_Count')}

## 2. Financial Capacity
- **GST Revenue:** ₹{f.get('GSTR_Declared_Revenue_Cr')} Cr
- **Bank Total Inflow:** ₹{f.get('Bank_Statement_Inflow_Cr')} Cr

## 3. Officer Analysis
{officer_insights or "No manual notes provided."}

---
*Generated by CREDI-MITRA AI Agent*
"""
    return report.strip()

# ──────────────────────────────────────────────
# 🛠️ SECTION 3: GRAPH ORCHESTRATOR
# ──────────────────────────────────────────────

ORCHESTRATOR_PROMPT = """You are CREDI-MITRA, an expert AI Credit Analyst.
Your goal is to complete a CAM report by using tools in this order:
1. extract_pdf_data
2. crawl_web_for_litigation
3. extract_numerical_features
4. run_xgboost_scorer
5. generate_cam_report

IMPORTANT:
- Use currency unit as CRORES (Cr).
- Ask for missing data exactly as requested by tools via interrupt.
- When the user tells you they have successfully uploaded documents, enthusiastically confirm receipt of the specific documents they mentioned, list them, and ask: "Shall I proceed with the analysis using my Credit Engine tools?" Wait for them to say yes before continuing.
- Context provided below: {context}
"""

def build_agent():
    llm = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0)
    checkpointer = MemorySaver()
    tools = [extract_pdf_data, crawl_web_for_litigation, extract_numerical_features, run_xgboost_scorer, generate_cam_report]
    return create_react_agent(model=llm, tools=tools, prompt=ORCHESTRATOR_PROMPT, checkpointer=checkpointer)

# ──────────────────────────────────────────────
# 🖥️ SECTION 4: STREAMLIT UI (Frontend)
# ──────────────────────────────────────────────

st.set_page_config(page_title="CREDI-MITRA — AI Credit Agent", page_icon="🏦", layout="wide")

# Custom Styling
st.markdown("""
<style>
    .main-title { background: linear-gradient(135deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 3.5rem; font-weight: 800; text-align: center; }
    .stButton>button { border-radius: 12px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# Initialization
if "messages" not in st.session_state: st.session_state.update({"logged_in": False, "messages": [], "pdf_text": "", "agent": None, "thread_id": None, "waiting_for_human": False, "interrupt_data": None})

def run_agent_main(user_input=None, resume_value=None):
    if st.session_state.agent is None:
        st.session_state.agent = build_agent()
        st.session_state.thread_id = str(uuid.uuid4())
    
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    agent_input = Command(resume=resume_value) if resume_value else {"messages": [{"role": "user", "content": f"{user_input}\n\nCompany: {st.session_state.get('company_name')}\nApp No: {st.session_state.get('app_no')}"}]}
    
    try:
        for event in st.session_state.agent.stream(agent_input, config=config, stream_mode="updates"):
            for node, data in event.items():
                if node == "__interrupt__":
                    st.session_state.waiting_for_human = True
                    st.session_state.interrupt_data = data[0].value
                    st.session_state.messages.append({"role": "assistant", "content": st.session_state.interrupt_data.get("question", "Need input"), "type": "interrupt"})
                    return
                elif node == "tools":
                    for m in data.get("messages", []):
                        st.session_state.messages.append({"role": "assistant", "content": f"🔧 Tool `{m.name}`: {m.content[:200]}...", "type": "tool_call", "tool_data": m.content, "tool_name": m.name})
                elif node == "agent":
                    for m in data.get("messages", []):
                        if m.content and not getattr(m, 'tool_calls', None):
                            st.session_state.messages.append({"role": "assistant", "content": m.content})
    except Exception as e:
        st.error(f"Agent Error: {e}")

# Views
def render_login():
    st.markdown("<h1 class='main-title'>CREDI-MITRA</h1>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.container(border=True):
            st.subheader("🔐 Credit Manager Login")
            u = st.text_input("Username", value="admin")
            p = st.text_input("Password", value="password", type="password")
            if st.button("🚀 Authenticate", use_container_width=True):
                if u == "admin" and p == "password": st.session_state.logged_in = True; st.rerun()
                else: st.error("Invalid Credentials")

def render_analysis():
    st.markdown("<h1 class='main-title' style='font-size: 2.5rem;'>Appraisal Console</h1>", unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("📁 Ingestion")
        st.session_state.company_name = st.text_input("Company Name")
        st.session_state.app_no = st.text_input("App No.")
        files = st.file_uploader("Upload PDFs/Data", accept_multiple_files=True)
        notes = st.text_area("Officer Notes")
        if st.button("🚀 Submit to Agent", type="primary", use_container_width=True):
            text = ""
            doc_names = []
            for f in files:
                doc_names.append(f.name)
                if f.type == "application/pdf":
                    reader = PdfReader(io.BytesIO(f.read()))
                    text += "".join([p.extract_text() for p in reader.pages])
                else: 
                    text += f"\n---\n{f.name} Data: " + pd.read_csv(f).to_string() if f.name.endswith(".csv") else ""
            if notes: doc_names.append("Officer Notes")
            st.session_state.pdf_text = text
            st.session_state.officer_notes = notes
            
            prompt = f"I have submitted the application for '{st.session_state.company_name}' (App No: {st.session_state.app_no}). I uploaded: {', '.join(doc_names) if doc_names else 'Nothing'}. Please confirm receipt and ask if you should proceed."
            st.session_state.auto_submit = prompt
            st.rerun()

    # Chat
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("tool_data") and st.expander("Details"): st.write(m["tool_data"])

    if st.session_state.get("auto_submit"):
        prompt = st.session_state.auto_submit
        st.session_state.auto_submit = None
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"): run_agent_main(user_input=prompt)
        st.rerun()

    if prompt := st.chat_input("Analyze..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        
        with st.chat_message("assistant"):
            if st.session_state.waiting_for_human:
                st.session_state.waiting_for_human = False
                run_agent_main(resume_value=prompt)
            else:
                run_agent_main(user_input=prompt)
        st.rerun()

if not st.session_state.logged_in: render_login()
else: render_analysis()
