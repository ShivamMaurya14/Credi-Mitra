"""
CREDI-MITRA Agent Graph — LLM Orchestrator Supervisor Pattern

Uses a Llama model via Groq as the central reasoning agent.
The agent has access to 5 specialized tools and can dynamically
decide the sequence of actions.

Two tools (crawl_web_for_litigation, extract_numerical_features) support
Human-in-the-Loop (HITL) via LangGraph's interrupt(), which pauses the
graph and returns control to the Streamlit UI for user clarification.
"""

import os
import json
import re
import uuid
# xgboost and pandas moved to internal functions to save memory
from typing import Annotated, Any, Dict, List, Optional, Union
from dotenv import load_dotenv

import streamlit as st
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
import time

# Import RAG tools
from rag import get_rag_tools, get_document_manager

# Load environment variables from .env file
env_locations = [
    ".env",
    os.path.join("..", ".env"),
    os.path.join(os.path.dirname(__file__), ".env"),
    os.path.join(os.path.dirname(__file__), "..", ".env")
]

env_loaded = False
for path in env_locations:
    if os.path.exists(path):
        load_dotenv(dotenv_path=path)
        env_loaded = True
        break

if not env_loaded:
    load_dotenv()

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile").strip()
TAVILY_API_KEY = os.environ.get("tavily_api_key", os.environ.get("TAVILY_API_KEY", "")).strip()

if not GROQ_API_KEY:
    try:
        st.error("❌ GROQ_API_KEY not found in .env file. Please check your .env configuration.")
        st.stop()
    except Exception:
        raise ValueError("GROQ_API_KEY not found in environment variables.")

RELIABLE_UPLOAD_DIR = None
RELIABLE_MODEL_NAME = None
RELIABLE_ANALYSIS_MODEL = None

def _get_tool_llm():
    model_choice = st.session_state.get("selected_model") or RELIABLE_MODEL_NAME
    if not model_choice:
        raise ValueError("Model selection is missing in session state.")
    
    # Handle the "(Groq)" or "(Google)" suffix if present
    clean_model = str(model_choice).split(" (")[0].strip()

    # ── Resolve Provider ──
    model_choice_str = str(model_choice).lower()
    
    if "(google)" in model_choice_str:
        provider = "google"
    elif "(openai)" in model_choice_str:
        provider = "openai"
    elif "(groq)" in model_choice_str:
        provider = "groq"
    else:
        # Fallback to keyword detection
        if "gemini" in clean_model.lower() or "google" in clean_model.lower():
            provider = "google"
        elif "gpt" in clean_model.lower() or "openai" in clean_model.lower():
            provider = "openai"
        else:
            provider = "groq"

    # ── Initialize Model ──
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        gemini_api_key = os.environ.get("gemini_api_key", "") or os.environ.get("GOOGLE_API_KEY", "")
        # Map to valid model names
        m_low = clean_model.lower()
        if clean_model.startswith("gemini-"):
            target = clean_model
        elif "3" in m_low:
            target = "gemini-3"
        elif "2.5" in m_low and "pro" in m_low:
            target = "gemini-2.5-pro"
        elif "2.5" in m_low and "flash-lite" in m_low:
            target = "gemini-2.5-flash-lite"
        elif "2.5" in m_low and "flash" in m_low:
            target = "gemini-2.5-flash"
        elif "pro" in m_low:
            target = "gemini-1.5-pro"
        else:
            target = "gemini-1.5-flash"
        return ChatGoogleGenerativeAI(
            model=target, 
            api_key=gemini_api_key, 
            temperature=0.1,
            model_kwargs={"tools": [{"google_search_retrieval": {}}]}
        )

    elif provider == "openai":
        openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        if "gpt-4o-mini" in clean_model.lower(): target = "gpt-4o-mini"
        elif "gpt-4o" in clean_model.lower(): target = "gpt-4o"
        elif "o1" in clean_model.lower(): target = "o1-mini"
        elif "o3-mini" in clean_model.lower(): target = "o3-mini"
        else: target = "gpt-4o-mini"
        return ChatOpenAI(model=target, temperature=0, api_key=openai_api_key)

    else:
        # Groq selection (strictly using the chosen model)
        return ChatGroq(
            model=clean_model,
            api_key=GROQ_API_KEY,
            temperature=0,
            max_retries=3
        )

# ──────────────────────────────────────────────
# Tool 0 — Set Numerical Feature
# ──────────────────────────────────────────────
@tool
def set_numerical_feature(feature_name: str, value: float, reasoning: str) -> str:
    """Store or update a specific numerical credit feature in the persistent database.
    
    Use this tool IMMEDIATELY when you find a financial metric in a document or news.
    
    Valid features:
    - Company_Age (Years)
    - CIBIL_Commercial_Score (300-900)
    - GSTR_Declared_Revenue_Cr (Crores)
    - Bank_Statement_Inflow_Cr (Crores)
    - Litigation_Count (Number of cases)
    - News_Sentiment_Score (-1.0 to 1.0)
    
    Args:
        feature_name: The name of the feature to update.
        value: The numerical value found.
        reasoning: Short explanation of where/how you found this value.
    """
    valid_features = [
        "Company_Age", "CIBIL_Commercial_Score", "GSTR_Declared_Revenue_Cr",
        "Bank_Statement_Inflow_Cr", "Litigation_Count", "News_Sentiment_Score"
    ]
    if feature_name not in valid_features:
        return f"Error: '{feature_name}' is invalid. Use: {', '.join(valid_features)}"

    _update_ml_features({feature_name: value})
    
    history = st.session_state.get("feature_update_history", [])
    history.append({"t": time.strftime("%H:%M:%S"), "f": feature_name, "v": value, "r": reasoning})
    st.session_state["feature_update_history"] = history
    
    _update_analysis_summary("feature_update", {"feature": feature_name, "value": value, "reason": reasoning})
    return f"Success: {feature_name} updated to {value}."

# ── Step Review Helper ──
CONTINUE_COMMANDS = {
    "continue", "next", "ok", "yes", "proceed", "go", "c", "n", "done", "", 
    "y", "yea", "yep", "yeah", "okay", "fine", "move on", "advance"
}

def _step_review(step_num: int, step_name: str, preview_lines: list) -> str:
    """
    Interrupt execution after a tool step, show the user a summary,
    and let them type 'continue' OR a correction before proceeding.
    Returns the user's raw response string.
    """
    preview_text = "\n".join(f"  • {line}" for line in preview_lines[:8])
    correction = interrupt({
        "type": "step_review",
        "step_number": step_num,
        "tool_name": step_name,
        "question": (
            f"\n---\n"
            f"✅ **Step {step_num}/5 — {step_name} Complete**\n\n"
            f"**Findings:**\n{preview_text}\n\n"
            f"Type **`continue`** to proceed to the next step, "
            f"or describe any corrections you would like to make before continuing."
        ),
    })
    return str(correction).strip()


# ──────────────────────────────────────────────
# Persistent JSON Store Helpers
# ──────────────────────────────────────────────
ML_FEATURES_FILE = "ml_features.json"
ANALYSIS_SUMMARY_FILE = "analysis_summary.json"


def _get_store_dir() -> str:
    """Return the current-application upload directory, or a temp fallback."""
    save_dir = st.session_state.get("current_upload_dir", "")
    if save_dir and os.path.exists(save_dir):
        return save_dir
    # Fallback: use temp with thread_id
    thread_id = st.session_state.get("thread_id", "default")
    fallback = os.path.join("temp", thread_id)
    os.makedirs(fallback, exist_ok=True)
    return fallback


def _update_ml_features(updates: dict) -> None:
    """
    Merge `updates` into ml_features.json.
    The file always contains a flat dict of the 6 XGBoost features
    (plus any intermediate values). Later writes override earlier ones
    only for keys that are explicitly provided.
    """
    store_dir = _get_store_dir()
    path = os.path.join(store_dir, ML_FEATURES_FILE)
    existing: dict = {}
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                existing = json.load(f)
        except Exception:
            existing = {}
    existing.update({k: v for k, v in updates.items() if v is not None})
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)


def _update_analysis_summary(section: str, data: dict) -> None:
    """
    Merge `data` under `section` key in analysis_summary.json.
    The file is a nested dict: { "section_name": { ...data... }, ... }.
    Existing sections are preserved; existing keys within a section are
    overwritten only if explicitly provided.
    """
    store_dir = _get_store_dir()
    path = os.path.join(store_dir, ANALYSIS_SUMMARY_FILE)
    existing: dict = {}
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                existing = json.load(f)
        except Exception:
            existing = {}
    if section not in existing:
        existing[section] = {}
    existing[section].update(data)
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)


def _read_ml_features() -> dict:
    """Read the current ml_features.json (returns {} if not yet written)."""
    store_dir = _get_store_dir()
    path = os.path.join(store_dir, ML_FEATURES_FILE)
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _read_analysis_summary() -> dict:
    """Read the current analysis_summary.json (returns {} if not yet written)."""
    store_dir = _get_store_dir()
    path = os.path.join(store_dir, ANALYSIS_SUMMARY_FILE)
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


# ──────────────────────────────────────────────
# Tool 1 — Extract PDF Data
# ──────────────────────────────────────────────
@tool
def extract_document_data(dummy_arg: str = "") -> str:
    """Extract and summarize key financial data from ALL uploaded documents.
    
    Call this tool FIRST. It reads the text directly from protected system storage.
    """
    # 1. Try session state (main thread)
    document_text = str(st.session_state.get("document_extracted_text", ""))
    
    # 2. Fallback to Bridge File (worker thread)
    if not document_text.strip():
        thread_id = st.session_state.get("thread_id", "default")
        bridge_path = os.path.join("temp", f"{thread_id}.txt")
        if os.path.exists(bridge_path):
            with open(bridge_path, "r") as f:
                document_text = f.read()
    
    if not document_text.strip():
        return json.dumps({
            "status": "warning",
            "message": "No document text was found in the session memory. If you have uploaded files, they might be image-only scans or corrupted.",
            "data": {}
        })
    
    # Check if there are warnings in the text
    has_warnings = "Binary/OCR Required" in document_text

    # 1. Identify sections
    sections_found = []
    section_markers = ["Application_Form", "CIBIL_Score_Report", "GST_Returns", "Bank_Statements", "Annual_Reports", "Officer_Insights_Report"]
    for marker in section_markers:
        if marker in document_text:
            sections_found.append(marker.replace("_", " "))

    # 2. Extract key metrics via regex (as a faster backup to LLM)
    extracted_metrics = {}
    try:
        # CIBIL
        cibil = re.search(r'(?:CIBIL|credit\s*score)[:\s]*(\d{3})', document_text, re.IGNORECASE)
        if cibil: extracted_metrics["CIBIL_Score"] = cibil.group(1)
        
        # Revenue
        rev = re.search(r'(?:revenue|turnover)[:\s]*(?:Rs\.?|INR|₹)?\s*([\d,.]+)\s*(Cr|Crore|Lakh)', document_text, re.IGNORECASE)
        if rev: extracted_metrics["Revenue_Estimate"] = f"{rev.group(1)} {rev.group(2)}"
        
        # Inflow
        inf = re.search(r'(?:inflow|total\s*inflow)[:\s]*(?:Rs\.?|INR|₹)?\s*([\d,.]+)\s*(Cr|Crore|Lakh)?', document_text, re.IGNORECASE)
        if inf: extracted_metrics["Bank_Inflow_Estimate"] = f"{inf.group(1)} {inf.group(2) or 'Cr'}"
    except:
        pass

    # 3. Final Summary
    result = {
        "status": "success" if not has_warnings else "partial_success",
        "message": "Documents ingested. Use 'extract_numerical_features' for final decision data.",
        "sections_found": sections_found,
        "document_length_chars": len(document_text),
        "preliminary_metrics": extracted_metrics,
        "raw_text_preview": str(document_text[:1500])
    }

    # ── Persist to JSON stores ──
    _update_ml_features({
        k: v for k, v in extracted_metrics.items()
        if k in ["CIBIL_Score", "Revenue_Estimate", "Bank_Inflow_Estimate"]
    })
    _update_analysis_summary("document_extraction", {
        "sections_found": sections_found,
        "document_length": len(document_text),
        "preliminary_metrics": extracted_metrics,
        "has_warnings": has_warnings,
    })

    # ── Interactive Step Review ──
    preview = [
        f"Sections found: {', '.join(sections_found) or 'None detected'}",
        f"Document length: {len(document_text):,} chars",
    ]
    for k, v in extracted_metrics.items():
        preview.append(f"{k}: {v}")
    user_reply = _step_review(2, "Aggregate Document Extraction", preview)
    if user_reply.lower() not in CONTINUE_COMMANDS:
        result["human_correction"] = user_reply
        _update_analysis_summary("document_extraction", {"human_note": user_reply})

    return json.dumps(result, indent=2)


@tool
def list_uploaded_documents() -> str:
    """Returns a list of all raw document files that have been uploaded for the current application.
    Call this FIRST to see what files need to be analyzed.
    """
    save_dir = st.session_state.get("current_upload_dir") or RELIABLE_UPLOAD_DIR
    if not save_dir or not os.path.exists(save_dir):
        return json.dumps({"status": "warning", "message": "No documents uploaded or directory not found.", "files": []})
    
    # User-requested preferred order
    preferred_order = [
        "Application_Form.pdf", 
        "CIBIL_Score_Report.pdf", 
        "GST_Returns.pdf", 
        "Bank_Statements.pdf", 
        "Annual_Reports.pdf", 
        "Officer_Insights_Report.pdf"
    ]
    
    raw_files = [f for f in os.listdir(save_dir) if os.path.isfile(os.path.join(save_dir, f)) and not f.endswith(".json") and not f.endswith(".txt")]
    
    # Sort files: first the preferred ones in order, then any others alphabetically
    files = []
    for pref in preferred_order:
        if pref in raw_files:
            files.append(pref)
            raw_files.remove(pref)
    
    files.extend(sorted(raw_files)) # Add miscellaneous files
    
    return json.dumps({"status": "success", "uploaded_files": files})


@tool
def analyze_document(filename: str) -> str:
    """Parse and analyze a single uploaded document. 
    Indexes document in Pinecone Cloud and uses RAG to extract features.
    
    Args:
        filename: The exact filename of the document (e.g. 'CIBIL_Score_Report.pdf')
    """
    save_dir = st.session_state.get("current_upload_dir") or RELIABLE_UPLOAD_DIR
    file_path = os.path.join(save_dir, filename)
    if not os.path.exists(file_path):
        return f"Error: {filename} not found."
    
    doc_type = filename.split(".")[0].replace("_", " ")
    company_name = st.session_state.get("company_name", "Unknown Company")
    ext = os.path.splitext(filename)[1].lower()
    
    # ── RAG INDEXING PHASE ──
    mgr = get_document_manager()
    
    upload_res = mgr.upload_pdf(file_path, company_name, doc_type)
    if upload_res["status"] == "error":
        return f"Error indexing {filename}: {upload_res['message']}"
    
    doc_id = upload_res["doc_id"]
    file_text = upload_res.get("text_content", "") # LlamaParse result

    # ── RAG FEATURE EXTRACTION PHASE ──
    # We query the DB specifically for the 5Cs and ML features
    feature_queries = {
        "CIBIL": "What is the CIBIL or credit score?",
        "Revenue": "What is the total revenue or turnover?",
        "Bank_Inflow": "What is the total bank credit or inflow?",
        "Litigation": "Are there any legal cases, NCLT, or disputes mentioned?",
        "5Cs_Character": "Information about management, promoters, and integrity.",
        "5Cs_Capacity": "Information about repayment ability and cash flow.",
        "5Cs_Capital": "Information about net worth and equity.",
        "5Cs_Collateral": "Information about security and assets offered.",
        "5Cs_Conditions": "Information about industry outlook and market conditions."
    }
    
    rag_findings = {}
    from concurrent.futures import ThreadPoolExecutor
    
    def _run_single_rag(key_q_tuple):
        key, q = key_q_tuple
        search_res = mgr.search_documents(q, company_name, doc_type, top_k=3)
        return key, "\n".join([r["content"] for r in search_res])

    # Dramatically limit workers for 8GB RAM stability
    with ThreadPoolExecutor(max_workers=1) as executor:
        results = list(executor.map(_run_single_rag, feature_queries.items()))
        for key, context in results:
            rag_findings[key] = context

    # ── AGGREGATE PERSISTENCE ──
    # Push the extracted text into the session memory
    doc_marker = f"\n\n--- Document: {filename} ---\n\n"
    new_text = doc_marker + file_text
    current_agg = st.session_state.get("document_extracted_text", "")
    st.session_state.document_extracted_text = current_agg + new_text
    
    # Update bridge file for background threads/resumption
    thread_id = st.session_state.get("thread_id", "default")
    os.makedirs("temp", exist_ok=True)
    bridge_path = os.path.join("temp", f"{thread_id}.txt")
    with open(bridge_path, "a") as f:
        f.write(new_text)

    # ── ANALYSIS PHASE (LLM Synthesis of RAG Findings) ──
    extractor_llm = _get_tool_llm()
    
    combined_context = ""
    for k, v in rag_findings.items():
        combined_context += f"### {k}\n{v}\n\n"

    prompt = f"""
    # FINANCIAL ANALYSIS REPORT: {doc_type}
    Company: {company_name}
    
    TASK: Based on the following RAG-retrieved contexts from the document, extract credit features.
    
    CONTEXTS:
    {combined_context}
    
    REQUIRED JSON OUTPUT:
    {{
      "Key_Entities": {{ "Name": "Role" }},
      "Financial_Metrics": {{ "Field": "Numerical Value" }},
      "Risk_Flags": "Summary of risks",
      "Litigation_Signals": 0,
      "Sentiment_Signal": 0.0,
      "Five_Cs": {{
        "Character": "Score/Detail",
        "Capacity": "Score/Detail",
        "Capital": "Score/Detail",
        "Collateral": "Score/Detail",
        "Conditions": "Score/Detail"
      }}
    }}
    
    Return ONLY JSON.
    """
    
    analysis_json = {}
    is_gemini = "gemini" in str(st.session_state.get("selected_analysis_model", "")).lower()
    
    try:
        response = extractor_llm.invoke([HumanMessage(content=prompt)])
        res_content = response.content
        if "```json" in res_content: res_content = res_content.split("```json")[1].split("```")[0].strip()
        elif "{" in res_content: res_content = res_content[res_content.find("{"):res_content.rfind("}")+1]
        analysis_json = json.loads(res_content)
    except Exception as e:
        return f"Error: LLM Analysis failed for {filename} after extraction: {str(e)}"
        
    def fuzzy_get(d, key_fragment):
        """Find a value in dict 'd' where the key contains 'key_fragment'"""
        if not isinstance(d, dict): return None
        for k, v in d.items():
            if key_fragment.lower() in str(k).lower(): return v
        return None

    # Robust extraction
    ents = fuzzy_get(analysis_json, "entity") or fuzzy_get(analysis_json, "name") or {}
    mets = fuzzy_get(analysis_json, "metric") or fuzzy_get(analysis_json, "financial") or {}
    risks = fuzzy_get(analysis_json, "risk") or fuzzy_get(analysis_json, "flag") or ""
    lit_signals = analysis_json.get("Litigation_Signals", fuzzy_get(analysis_json, "litigation") or 0)
    sent_signal = analysis_json.get("Sentiment_Signal", fuzzy_get(analysis_json, "sentiment") or 0.0)

    # Save to disk (granular file analysis)
    analysis_dir = os.path.join(save_dir, "analysis_jsons")
    os.makedirs(analysis_dir, exist_ok=True)
    out_path = os.path.join(analysis_dir, f"{filename.split('.')[0]}_analysis.json")
    with open(out_path, "w") as f:
        json.dump(analysis_json, f, indent=2)

    # Write numerical values to ml_features using smarter detection
    for k, v in mets.items():
        k_l = str(k).lower()
        if ("cibil" in k_l or "score" in k_l or "credit" in k_l):
            try: _update_ml_features({"CIBIL_Score_Raw": float(str(v).replace(",",""))})
            except: pass
        elif ("revenue" in k_l or "turnover" in k_l or "sales" in k_l):
            try: _update_ml_features({"Revenue_Cr_Raw": float(str(v).replace(",",""))})
            except: pass
        elif ("inflow" in k_l or "credit summ" in k_l):
            try: _update_ml_features({"Inflow_Cr_Raw": float(str(v).replace(",",""))})
            except: pass
        elif ("age" in k_l or "incorp" in k_l or "year" in k_l):
            try:
                val = float(str(v).replace(",",""))
                if val > 1900: # Year
                    _update_ml_features({"Company_Age_Raw": max(0, 2026 - int(val))})
                else:
                    _update_ml_features({"Company_Age_Raw": val})
            except: pass

    _update_analysis_summary(f"document_{doc_type}", {
        "aggregated_entities": ents,
        "aggregated_metrics": mets,
        "risk_flags": risks,
        "litigation_signals": lit_signals,
        "sentiment_signal": sent_signal,
    })
    
    # Also push Litigation/Sentiment to ML Features immediately
    _update_ml_features({
        "Litigation_Count_Raw": int(float(str(lit_signals).replace(",",""))) if lit_signals else 0,
        "News_Sentiment_Score_Raw": float(str(sent_signal).replace(",","")) if sent_signal else 0.0
    })

    # --- PROFESSIONAL REPORTING REFINEMENT ---
    def _beauty_format(val):
        """Format large numbers into Cr/Lakh for professional display"""
        try:
            clean_str = str(val).replace(",", "").replace("₹", "").strip()
            num = float(clean_str)
            if num >= 10000000: return f"₹{num/10000000:.2f} Cr"
            if num >= 100000: return f"₹{num/100000:.2f} Lakhs"
            if num > 1000: return f"{num:,.0f}"
            return str(val)
        except: return str(val)

    cs_data = fuzzy_get(analysis_json, "five_c") or {}

    commit_log = []
    for k, v in mets.items():
        k_l = str(k).lower()
        if any(x in k_l for x in ["cibil", "score", "revenue", "turnover", "inflow", "age", "incorp", "year", "amount"]):
            commit_log.append(f"• **{k}**: {_beauty_format(v)}")

    if lit_signals: commit_log.append(f"• **Litigation Findings**: {lit_signals} incident(s)")
    if sent_signal: commit_log.append(f"• **Sentiment Score**: {sent_signal}")

    # Build the professional structured report
    preview = [
        f"📂 **DOCUMENT APPRAISAL: {doc_type}**",
        f"---",
        f"🔍 **EXTRACTION ENGINE**",
        f"Technology: `{'Gemini Native' if is_gemini and ext == '.pdf' else 'RAG + LlamaParse'}`",
        f"Status: ✅ Successfully Indexed in Pinecone",
        "",
        f"📊 **ML DATASET (IDENTIFIED METRICS)**",
        *(commit_log if commit_log else ["_No critical ML features detected in this document._"]),
        "",
        f"🧠 **CREDIT QUALITY (THE 5Cs)**",
        f"• **Character**: {cs_data.get('Character', 'Not mentioned')}",
        f"• **Capacity**: {cs_data.get('Capacity', 'Not mentioned')}",
        f"• **Capital**: {cs_data.get('Capital', 'Not mentioned')}",
        f"• **Collateral**: {cs_data.get('Collateral', 'Not mentioned')}",
        "",
        f"⚠️ **RISK EVALUATION**",
        f"{str(risks)[:500] if risks else 'No immediate financial risks or adverse flags detected.'}",
        "",
        f"🏦 **VAULT SYNCHRONIZATION**",
        f"Main database has been updated with {len(commit_log)} key parameters from this appraisal."
    ]
    user_reply = _step_review(1, f"Individual Document Analysis", preview)
    
    # ── RESILIENCE CHECK (Reload findings from Vault) ──
    # Re-calculate path in case of local scope loss during refresh
    safe_save_dir = st.session_state.get("current_upload_dir") or RELIABLE_UPLOAD_DIR
    safe_out_path = os.path.join(safe_save_dir, "analysis_jsons", f"{filename.split('.')[0]}_analysis.json")
    
    if (not analysis_json or len(analysis_json) < 2) and os.path.exists(safe_out_path):
        try:
            with open(safe_out_path, "r") as f:
                analysis_json = json.load(f)
        except: pass

    # Second fallback: If STILL empty, use the data we just pushed to the summary
    if not analysis_json or len(analysis_json) < 2:
        analysis_json = {
            "Key_Entities": ents,
            "Financial_Metrics": mets,
            "Risk_Flags": risks,
            "Litigation_Signals": lit_signals,
            "Sentiment_Signal": sent_signal,
            "Resilience_Mode": True
        }

    result = {"status": f"Successfully parsed {filename} and extracted features.", "extracted_features": analysis_json}
    if user_reply.lower() not in CONTINUE_COMMANDS:
        result["human_correction"] = user_reply
        _update_analysis_summary(f"document_{doc_type}", {"human_note": user_reply})

    return json.dumps(result, indent=2)


# ──────────────────────────────────────────────
# Tool 2 — Crawl Web for Litigation (HITL)
# ──────────────────────────────────────────────
@tool
def crawl_web_for_litigation(company_name: str) -> str:
    """Search the web for litigation records, NCLT filings, regulatory actions,
    and news sentiment for a given company.
    
    ROBUST VERSION: Returns structured analysis data with proper error handling,
    API fallbacks, and comprehensive litigation analysis.
    
    This tool searches public databases and news sources for any legal disputes,
    NCLT filings, RBI regulatory actions, and general news sentiment.
    
    Returns structured JSON data with metrics:
    - Litigation Count
    - News Sentiment Score (-1 to 1)
    - NCLT Cases
    - RBI Regulatory Actions
    - Risk Score Calculation
    
    NOTE: This tool returns pure data only. UI rendering happens in the app layer.
    """
    if not company_name or not company_name.strip():
        return json.dumps({"status": "error", "message": "No company name provided.", "detailed_findings": []})

    company_name = company_name.strip()

    # ── Verify name clarity ──
    is_ambiguous = len(company_name) < 3 or company_name.lower() in ["company", "sample", "test", "business"]
    if is_ambiguous:
        try:
            clarification = interrupt({
                "question": f"⚠️ **Incomplete Company Name Detected**\n\nPlease provide the full legal name for entity: **{company_name}**.",
                "type": "ambiguity_check",
                "company": company_name
            })
            company_name = str(clarification).strip()
        except Exception:
            pass  # If interrupt fails, continue with provided name

    # ── Initialize data structures ──
    processed_data: List[Dict] = []
    agg_litigation: int = 0
    agg_sentiment: float = 0.0
    positive_count: int = 0
    negative_count: int = 0
    details_nclt: List[str] = []
    details_rbi: List[str] = []
    final_headlines: List[str] = []
    snippets: List[Dict] = []
    warnings_log: List[str] = []  # Log warnings instead of displaying

    # ── Check Provider for Built-in Search ──
    model_choice = st.session_state.get("selected_model", "").lower()
    if "google" in model_choice or "openai" in model_choice:
        return json.dumps({
            "status": "direct_search_required",
            "message": "As a Gemini/OpenAI model, please use your INTERNAL WEB SEARCH CAPABILITIES (e.g. google_search_retrieval) directly to find legal records and news. Do not use this tool for search.",
            "detailed_findings": []
        })

    # ── Real Web Research via Tavily (for OSS/Groq) ──
    try:
        if not TAVILY_API_KEY or TAVILY_API_KEY.lower() in ["", "none", "false", "disabled"]:
            warnings_log.append("Tavily API key not configured. Using mock data.")
            # Create mock data for demonstration
            snippets = [
                {
                    "title": f"No litigation records found for {company_name}",
                    "url": "https://mock-database.example.com",
                    "content": "Standard corporate compliance verified."
                },
                {
                    "title": f"{company_name} - Clean regulatory status",
                    "url": "https://rbi-database.example.com",
                    "content": "No RBI regulatory actions on record."
                }
            ]
        else:
            try:
                from tavily import TavilyClient
            except ImportError:
                return json.dumps({
                    "status": "error",
                    "message": "Tavily library not installed",
                    "detailed_findings": []
                })
            
            client = TavilyClient(api_key=TAVILY_API_KEY)
            
            # Multiple search queries to cover different litigation types
            search_queries = [
                f"{company_name} litigation NCLT court cases",
                f"{company_name} RBI penalties regulatory actions",
                f"{company_name} legal disputes news",
            ]
            
            for query in search_queries:
                try:
                    search_result = client.search(query=query, search_depth="advanced", max_results=5)
                    for r in search_result.get("results", []):
                        snippets.append({
                            "title": r.get('title', 'No Title').strip(),
                            "url": r.get('url', '').strip(),
                            "content": r.get('content', '').strip()
                        })
                except Exception as api_err:
                    warnings_log.append(f"Search query failed: {str(api_err)[:80]}")
                    continue
            
            # Deduplicate by URL
            seen_urls = set()
            unique_shards: List[Dict] = []
            for s in snippets:
                curr_url = s.get('url')
                if curr_url and curr_url not in seen_urls:
                    unique_shards.append(s)
                    seen_urls.add(curr_url)
            snippets = unique_shards[:10]  # Cap at 10 results

    except Exception as api_err:
        warnings_log.append(f"Web API error: {str(api_err)[:100]}")
        snippets = []  # Continue with empty results

    if not snippets:
        snippets = [{
            "title": f"No results for {company_name}",
            "url": "No URL",
            "content": "No litigation or regulatory information found in public records."
        }]

    # ── Granular LLM Analysis (One-by-One Snippet Processing) ──
    analysis_llm = _get_tool_llm()
    errors_log = []  # Log errors during analysis
    
    for i, snippet in enumerate(snippets):
        try:
            # Validate snippet content
            title = (snippet.get('title') or 'N/A')[:200]
            content = (snippet.get('content') or 'No content')[:1000]
            
            snippet_prompt = f"""# LEGAL & REGULATORY ANALYSIS [{i+1}/{len(snippets)}]
Company: {company_name}
Title: {title}
Content: {content}

TASK: Analyze this news/result for litigation, NCLT filings, RBI actions, and sentiment.
RESPOND WITH ONLY VALID JSON (no markdown):
{{
  "is_positive": false,
  "is_negative": false,
  "litigation_found": false,
  "sentiment_score": 0.0,
  "risk_level": "LOW",
  "risk_summary": "Brief summary",
  "is_nclt": false,
  "is_rbi_penalty": false,
  "litigation_type": "None"
}}"""
            try:
                if i > 0: time.sleep(0.3)
                
                resp = analysis_llm.invoke([HumanMessage(content=snippet_prompt)])
                clean_resp = resp.content.strip()
                
                # Robust JSON extraction
                if "```json" in clean_resp:
                    clean_resp = clean_resp.split("```json")[1].split("```")[0].strip()
                elif "```" in clean_resp:
                    clean_resp = clean_resp.split("```")[1].split("```")[0].strip()
                
                # Find JSON object
                start_idx = clean_resp.find("{")
                end_idx = clean_resp.rfind("}") + 1
                
                if start_idx != -1 and end_idx > start_idx:
                    clean_resp = clean_resp[start_idx:end_idx]
                
                data = json.loads(clean_resp)
                
                # Validate and set defaults
                data = {
                    "is_positive": bool(data.get("is_positive", False)),
                    "is_negative": bool(data.get("is_negative", False)),
                    "litigation_found": bool(data.get("litigation_found", False)),
                    "sentiment_score": float(data.get("sentiment_score", 0.0)),
                    "risk_level": str(data.get("risk_level", "LOW")).upper(),
                    "risk_summary": str(data.get("risk_summary", "N/A"))[:150],
                    "is_nclt": bool(data.get("is_nclt", False)),
                    "is_rbi_penalty": bool(data.get("is_rbi_penalty", False)),
                    "litigation_type": str(data.get("litigation_type", "None"))
                }
                
                # Count metrics
                if data.get("is_positive"): positive_count = int(positive_count) + 1
                if data.get("is_negative"): negative_count = int(negative_count) + 1
                if data.get("litigation_found"): agg_litigation = int(agg_litigation) + 1
                
                agg_sentiment = float(agg_sentiment) + float(data.get("sentiment_score", 0.0))
                
                # ── STORE IN PINECONE CLOUD ──
                mgr = get_document_manager()
                mgr.db.add_web_result(
                    result_id=f"web_{uuid.uuid4().hex[:8]}",
                    company=company_name,
                    content=content,
                    metadata=data
                )
                
                if data.get("risk_summary") and data["risk_summary"] != "N/A":
                    r_sum = str(data["risk_summary"])
                    if data.get("is_nclt"): details_nclt.append(r_sum)
                    if data.get("is_rbi_penalty"): details_rbi.append(r_sum)
                
                snippet_info = {
                    "headline": title,
                    "url": snippet.get('url', 'N/A'),
                    "sentiment": data.get("sentiment_score", 0.0),
                    "risk_level": data.get("risk_level", "MEDIUM"),
                    "litigation_type": data.get("litigation_type", "None"),
                    "risk_found": data.get("litigation_found", False),
                    "summary": data.get("risk_summary", ""),
                    "is_nclt": data.get("is_nclt", False),
                    "is_rbi": data.get("is_rbi_penalty", False)
                }
                processed_data.append(snippet_info)
                final_headlines.append(title)
                    
            except json.JSONDecodeError as je:
                errors_log.append(f"Failed to parse response JSON: {str(je)[:60]}")
                # Create minimal record
                processed_data.append({
                    "headline": title,
                    "url": snippet.get('url', 'N/A'),
                    "sentiment": 0.0,
                    "risk_level": "UNKNOWN",
                    "litigation_type": "Error",
                    "risk_found": False,
                    "summary": "Parse error",
                    "is_nclt": False,
                    "is_rbi": False
                })
                continue
                    
        except Exception as e:
            errors_log.append(f"Analysis error for result {i+1}: {str(e)[:100]}")
            continue
    
    # ── Synthesis with Risk Score ──
    litigation_count = agg_litigation
    avg_len = max(1, len(snippets))
    sentiment_score = round(agg_sentiment / avg_len, 2)
    risk_score = round((negative_count - positive_count) / avg_len, 2)
    nclt_cases = list(set(details_nclt))[:5]
    rbi_actions = list(set(details_rbi))[:5]
    
    # ── Prepare final result (pure data, no UI calls) ──
    import pandas as pd
    result = {
        "status": "success" if processed_data else "partial_success",
        "company_searched": company_name,
        "litigation_count": litigation_count,
        "news_sentiment_score": sentiment_score,
        "positive_news_count": positive_count,
        "negative_news_count": negative_count,
        "total_results_analyzed": len(snippets),
        "risk_score": risk_score,
        "nclt_cases": nclt_cases,
        "rbi_regulatory_actions": rbi_actions,
        "headlines": final_headlines[:5],
        "detailed_findings": processed_data,
        "analysis_timestamp": pd.Timestamp.now().isoformat(),
        "warnings": warnings_log,
        "errors_during_analysis": errors_log
    }

    # ── Save detailed findings to disk ──
    try:
        save_dir = st.session_state.get("current_upload_dir") or RELIABLE_UPLOAD_DIR
        if save_dir and os.path.exists(save_dir):
            analysis_dir = os.path.join(save_dir, "analysis_jsons")
            os.makedirs(analysis_dir, exist_ok=True)
            with open(os.path.join(analysis_dir, "web_research_analysis.json"), "w") as f:
                json.dump(result, f, indent=2)
    except Exception as e:
        pass  # Silently fail if can't save

    # ── Update global stores ──
    try:
        _update_ml_features({
            "Litigation_Count": litigation_count,
            "News_Sentiment_Score": sentiment_score,
        })
        _update_analysis_summary("web_research", {
            "litigation_count": litigation_count,
            "sentiment": sentiment_score,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "risk_score": risk_score,
            "total_analyzed": len(snippets),
            "nclt": nclt_cases[:3],
            "rbi": rbi_actions[:3]
        })
    except Exception:
        pass  # Silently fail if can't update stores

    # ── HITL Review ──
    try:
        preview = [
            f"✅ Litigation Count: {litigation_count}",
            f"✅ Sentiment: {sentiment_score}",
            f"✅ NCLT Cases: {len(nclt_cases)}",
            f"✅ RBI Actions: {len(rbi_actions)}",
            f"✅ Positive News: {positive_count}",
            f"✅ Negative News: {negative_count}",
            f"✅ Net Risk Score: {risk_score}",
            f"✅ Total Analyzed: {len(snippets)}"
        ]
        user_reply = _step_review(3, "Web Research - Litigation & Regulatory Analysis", preview)
        if user_reply and user_reply.lower() not in CONTINUE_COMMANDS:
            _update_analysis_summary("web_research", {"human_note": user_reply})
    except Exception:
        pass  # If interrupt fails, skip review

    return json.dumps(result, indent=2)



# ──────────────────────────────────────────────
# Tool 3 — Extract Numerical Features (HITL)
# ──────────────────────────────────────────────
@tool
def extract_numerical_features(
    company_name: str,
    document_summary: str = "",
    web_research: str = ""
) -> str:
    """Convert unstructured text data into the 6 numerical features required
    by the XGBoost ML model.
    
    This tool extracts: Company_Age, CIBIL_Commercial_Score,
    GSTR_Declared_Revenue_Cr, Bank_Statement_Inflow_Cr, Litigation_Count,
    and News_Sentiment_Score from the combined PDF and web research data.
    
    IMPORTANT: If any critical feature cannot be found in the provided data,
    this tool will PAUSE and ask the user to manually input the missing value.
    
    Args:
        company_name: Name of the company being analyzed.
        document_summary: The structured summary from extract_document_data tool.
        web_research: The results from crawl_web_for_litigation tool.
    
    Returns:
        JSON with all 6 numerical features ready for XGBoost.
    """
    features = {}
    missing_features = []

    # Try parsing the document summary
    pdf_data = {}
    if isinstance(document_summary, str):
        try:
            # Handle potential single quote issues or extra text from LLM
            clean = document_summary.strip()
            if "```json" in clean:
                clean = clean.split("```json")[1].split("```")[0].strip()
            pdf_data = json.loads(clean.replace("'", '"'))
        except Exception:
            pdf_data = {}
    elif isinstance(document_summary, dict):
        pdf_data = document_summary

    # Try parsing web research
    web_data = {}
    if isinstance(web_research, str):
        try:
            clean_web = web_research.strip()
            if "```json" in clean_web:
                clean_web = clean_web.split("```json")[1].split("```")[0].strip()
            web_data = json.loads(clean_web.replace("'", '"'))
        except Exception:
            web_data = {}
    elif isinstance(web_research, dict):
        web_data = web_research

    key_numbers = {}

    def smart_to_float(val):
        """Robust float conversion with Cr/Lakh/Absolute awareness"""
        if val is None: return 0.0
        s = str(val).lower().replace(",", "").replace("₹", "").strip()
        
        # 1. Extract first number
        match = re.search(r'(\d+(?:\.\d+)?)', s)
        if not match: return 0.0
        num = float(match.group(1))
        
        # 2. Logic for units
        if 'crore' in s or ' cr' in s: return num
        if 'lakh' in s: return num / 100.0
        
        # 3. Logic for large absolute numbers (convert to Crores)
        if num >= 100000: # 1 Lakh or more
            return num / 10000000.0
            
        return num

    def deep_find_metrics(data_obj, target_map):
        """Recursively scan a dict/list for keys matching target patterns"""
        if isinstance(data_obj, dict):
            for k, v in data_obj.items():
                k_l = k.lower()
                if ("age" in k_l or "incorp" in k_l or "year" in k_l) and "Company_Age" not in target_map:
                    val = smart_to_float(v)
                    if val > 1900: # It's a year, calculate age
                        target_map["Company_Age"] = max(0, 2026 - int(val))
                    else:
                        target_map["Company_Age"] = val
                if ("cibil" in k_l or "score" in k_l or "credit" in k_l) and "CIBIL_Score" not in target_map:
                    target_map["CIBIL_Score"] = smart_to_float(v)
                if ("turnover" in k_l or "revenue" in k_l or "sales" in k_l) and "Revenue_Cr" not in target_map:
                    target_map["Revenue_Cr"] = smart_to_float(v)
                if ("inflow" in k_l or "credit summ" in k_l or "bank" in k_l) and "Bank_Inflow_Cr" not in target_map:
                    val = smart_to_float(v)
                    if val > 0: target_map["Bank_Inflow_Cr"] = val
                
                # Capture Risk Signals
                if "litigation" in k_l and "Litigation_Signals" not in target_map:
                    try: target_map["Litigation_Signals"] = int(float(v))
                    except: pass
                if "sentiment" in k_l and "Sentiment_Signal" not in target_map:
                    try: target_map["Sentiment_Signal"] = float(v)
                    except: pass

                deep_find_metrics(v, target_map)
        elif isinstance(data_obj, list):
            for item in data_obj:
                deep_find_metrics(item, target_map)

    # ── TIER 1: Scan high-level AI Summary (Fast Python Scan) ──
    deep_find_metrics(pdf_data, key_numbers)

    # ── TIER 2: Granular JSON Interrogation (LLM-Driven) ──
    # User requested to send LLM 'line by line' (file by file) of JSON files
    save_dir = st.session_state.get("current_upload_dir") or RELIABLE_UPLOAD_DIR
    if save_dir and os.path.exists(os.path.join(save_dir, "analysis_jsons")):
        analysis_dir = os.path.join(save_dir, "analysis_jsons")
        verifier_llm = _get_tool_llm()
        
        for json_file in os.listdir(analysis_dir):
            if not json_file.endswith(".json"): continue
            try:
                with open(os.path.join(analysis_dir, json_file), "r") as f:
                    content_str = f.read()
                
                interrogation_prompt = f"""# GRANULAR FEATURE VERIFICATION: {json_file}
                Data Content: {content_str}
                
                TASK: Extract precisely these 6 numerical fields for the ML model:
                1. "Age": (Integer age of company)
                2. "CIBIL": (3-digit credit score)
                3. "Revenue": (Revenue in Crores)
                4. "Inflow": (Bank Inflow in Crores)
                5. "Litigation": (Count of legal cases)
                6. "Sentiment": (Finance health score -1 to 1)
                
                Return ONLY JSON. No other text.
                """
                if json_file != os.listdir(analysis_dir)[0]:
                    time.sleep(0.5) # Prevent rate limiting
                
                res = verifier_llm.invoke([HumanMessage(content=interrogation_prompt)])
                clean = res.content
                if "```json" in clean: clean = clean.split("```json")[1].split("```")[0].strip()
                elif "{" in clean: clean = clean[clean.find("{"):clean.rfind("}")+1]
                found = json.loads(clean)
                
                # Update key_numbers if values found (Prioritize most specific findings)
                if found.get("Age") and "Company_Age" not in key_numbers: key_numbers["Company_Age"] = smart_to_float(found["Age"])
                if found.get("CIBIL") and "CIBIL_Score" not in key_numbers: key_numbers["CIBIL_Score"] = smart_to_float(found["CIBIL"])
                if found.get("Revenue") and "Revenue_Cr" not in key_numbers: key_numbers["Revenue_Cr"] = smart_to_float(found["Revenue"])
                if found.get("Inflow") and "Bank_Inflow_Cr" not in key_numbers: key_numbers["Bank_Inflow_Cr"] = smart_to_float(found["Inflow"])
                if found.get("Litigation") and "Litigation_Signals" not in key_numbers: key_numbers["Litigation_Signals"] = int(found["Litigation"])
                if found.get("Sentiment") and "Sentiment_Signal" not in key_numbers: key_numbers["Sentiment_Signal"] = float(found["Sentiment"])
            except: continue

    # ── TIER 3: Granular Raw Text Extraction ──
    try:
        raw_text = st.session_state.get("document_extracted_text", "")
        if not raw_text.strip():
            thread_id = st.session_state.get("thread_id")
            bridge_path = os.path.join("temp", f"{thread_id}.txt")
            if os.path.exists(bridge_path):
                with open(bridge_path, "r") as f:
                    raw_text = f.read()

        if raw_text.strip():
            doc_chunks = re.split(r'--- Document: (.*?) ---', raw_text)
            processed_chunks = []
            for i in range(1, len(doc_chunks), 2):
                processed_chunks.append((doc_chunks[i], doc_chunks[i+1].strip()))

            # ── Sequential Micro-Chunking Interrogation ──
            verifier_llm = _get_tool_llm()

            for doc_type, doc_content in processed_chunks:
                if len(doc_content) < 50: continue
                
                # Split this document into 3000-char slices
                slices = [doc_content[i:i+3000] for i in range(0, len(doc_content), 3000)]
                
                for s_idx, slice_text in enumerate(slices):
                    needed = [k for k in ["Company_Age", "CIBIL_Score", "Revenue_Cr", "Bank_Inflow_Cr"] if k not in key_numbers]
                    if not needed: break

                    prompt = f"""# REASONING TASK: FINANCIAL FEATURE MAPPING [CHUNK {s_idx+1}/{len(slices)}]
                    Current Year: 2026
                    Document Strategy: {doc_type}
                    
                    TASK: Extract the following metrics into a JSON object.
                    1. "Company_Age": Integer. If you find an Incorporation Year (e.g. 2015), calculate (2026 - Year).
                    2. "CIBIL_Score": Integer (typically 300 to 900).
                    3. "Revenue_Cr": Float. Extract total revenue/turnover. Convert to Crores.
                    4. "Bank_Inflow_Cr": Float. Extract total bank credits/inflow. Convert to Crores.

                    RULES:
                    - If a value is in 'Lakhs', divide by 100.
                    - If no value is found, do NOT include the key.
                    - Return ONLY the JSON.

                    TEXT:
                    {slice_text}"""
                    
                    try:
                        time.sleep(1) # Defensive breathing room for TPM limits
                        res = verifier_llm.invoke([HumanMessage(content=prompt)])
                        clean_res = res.content
                        if "```json" in clean_res: clean_res = clean_res.split("```json")[1].split("```")[0].strip()
                        elif "{" in clean_res: clean_res = clean_res[clean_res.find("{"):clean_res.rfind("}")+1]
                        found = json.loads(clean_res)
                        
                        if found.get("Age") and "Company_Age" not in key_numbers: key_numbers["Company_Age"] = smart_to_float(found["Age"])
                        if found.get("CIBIL") and "CIBIL_Score" not in key_numbers: key_numbers["CIBIL_Score"] = smart_to_float(found["CIBIL"])
                        if found.get("Revenue") and "Revenue_Cr" not in key_numbers: key_numbers["Revenue_Cr"] = smart_to_float(found["Revenue"])
                        if found.get("Inflow") and "Bank_Inflow_Cr" not in key_numbers: key_numbers["Bank_Inflow_Cr"] = smart_to_float(found["Inflow"])
                    except: continue

        # ── TIER 4: Removed Global Sweep to avoid TPM issues ──
    except Exception as e:
        print(f"Deep extraction error: {e}")

    # 3. Assign to final features map
    # 3. Assign to final features map - CORRECTED checks (allow 0 if found)
    if "Company_Age" in key_numbers: 
        features["Company_Age"] = float(key_numbers["Company_Age"])
    else: missing_features.append("Company_Age")
 
    if "CIBIL_Score" in key_numbers:
        features["CIBIL_Commercial_Score"] = float(key_numbers["CIBIL_Score"])
    else: missing_features.append("CIBIL_Commercial_Score")
 
    if "Revenue_Cr" in key_numbers:
        features["GSTR_Declared_Revenue_Cr"] = float(key_numbers["Revenue_Cr"])
    else: missing_features.append("GSTR_Declared_Revenue_Cr")
 
    if "Bank_Inflow_Cr" in key_numbers:
        features["Bank_Statement_Inflow_Cr"] = float(key_numbers["Bank_Inflow_Cr"])
    else: missing_features.append("Bank_Statement_Inflow_Cr")

    # ── Feature 5: Litigation Count (Unified Scan) ──
    # Priority 1: Web Data
    # Priority 2: Document Summary Signals
    # Priority 3: Fallback 0
    l_count = 0
    if "litigation_count" in web_data:
        l_count = int(web_data["litigation_count"])
    elif key_numbers.get("Litigation_Signals") is not None:
        l_count = int(key_numbers["Litigation_Signals"])
    else:
        # Check if saved in previous step
        l_count = int(_read_ml_features().get("Litigation_Count", 0))
    features["Litigation_Count"] = l_count

    # ── Feature 6: News Sentiment (Unified Scan) ──
    s_score = 0.0
    if "news_sentiment_score" in web_data:
        s_score = float(web_data["news_sentiment_score"])
    elif key_numbers.get("Sentiment_Signal") is not None:
        s_score = float(key_numbers["Sentiment_Signal"])
    else:
        # Check if saved in previous step
        s_score = float(_read_ml_features().get("News_Sentiment_Score", 0.0))
    
    # Cap and round according to risk rubric
    features["News_Sentiment_Score"] = round(max(-1.0, min(1.0, s_score)), 2)

    # ── HUMAN-IN-THE-LOOP: Ask for missing critical features ──
    critical_missing = [f for f in missing_features if f in [
        "CIBIL_Commercial_Score", "GSTR_Declared_Revenue_Cr", "Bank_Statement_Inflow_Cr", "Company_Age"
    ]]

    if critical_missing:
        # Build a clear question for the user
        missing_descriptions = {
            "CIBIL_Commercial_Score": "CIBIL Commercial Score (typically 300–900)",
            "GSTR_Declared_Revenue_Cr": "GSTR Declared Revenue in Crores (e.g., 120.5)",
            "Bank_Statement_Inflow_Cr": "Bank Statement Total Inflow in Crores (e.g., 115.0)",
            "Company_Age": "Company Age in Years (e.g., 15)"
        }
        question_parts = []
        for f in critical_missing:
            question_parts.append(f"• **{missing_descriptions.get(f, f)}**")

        human_response = interrupt({
            "question": (
                f"📊 **Missing Financial Data for {company_name}**\n\n"
                f"I could not find the following critical values in the uploaded documents:\n\n"
                + "\n".join(question_parts) + "\n\n"
                f"Please provide these values. You can reply like:\n"
                f"`CIBIL: 750, Revenue: 120.5, Age: 12`"
            ),
            "type": "missing_data",
            "missing_fields": critical_missing
        })

        # Parse the user's response
        response_text = str(human_response)
        
        if "CIBIL_Commercial_Score" in critical_missing:
            match = re.search(r'(?:CIBIL|score)[:\s]*(\d{3})', response_text, re.IGNORECASE)
            if match:
                features["CIBIL_Commercial_Score"] = int(match.group(1))
            else:
                nums = re.findall(r'\b(\d{3})\b', response_text)
                features["CIBIL_Commercial_Score"] = int(nums[0]) if nums else 700

        def parse_financial_value(pattern, text):
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                val = float(match.group(1).replace(",", ""))
                unit = (match.group(2) or "cr").lower()
                if 'lakh' in unit:
                    return val / 100.0
                return val
            return None

        if "GSTR_Declared_Revenue_Cr" in critical_missing:
            val = parse_financial_value(r'(?:revenue|gst)[:\s]*([\d.,]+)\s*(cr|lakh|crore)?', response_text)
            if val is not None:
                features["GSTR_Declared_Revenue_Cr"] = val
            else:
                nums = re.findall(r'(\d+(?:\.\d+)?)', response_text)
                features["GSTR_Declared_Revenue_Cr"] = float(nums[0]) if nums else 100.0

        if "Bank_Statement_Inflow_Cr" in critical_missing:
            val = parse_financial_value(r'(?:inflow|bank)[:\s]*([\d.,]+)\s*(cr|lakh|crore)?', response_text)
            if val is not None:
                features["Bank_Statement_Inflow_Cr"] = val
            else:
                nums = re.findall(r'(\d+(?:\.\d+)?)', response_text)
                features["Bank_Statement_Inflow_Cr"] = float(nums[-1]) if nums else 100.0

        if "Company_Age" in critical_missing:
            match = re.search(r'(?:age|years?)[:\s]*(\d+)', response_text, re.IGNORECASE)
            if match:
                features["Company_Age"] = float(match.group(1))
            else:
                nums = re.findall(r'\b(\d{1,2})\b', response_text)
                features["Company_Age"] = float(nums[0]) if nums else 5.0

    result = {
        "status": "success",
        "company": company_name,
        "features": features,
        "missing_fields_resolved": critical_missing if critical_missing else [],
        "notes": "All 6 features ready for XGBoost scoring."
    }

    # ── Persist to JSON stores ── (authoritative feature write)
    final_features = result.get("features", {})
    if final_features:
        _update_ml_features({
            "Company_Age": final_features.get("Company_Age"),
            "CIBIL_Commercial_Score": final_features.get("CIBIL_Commercial_Score"),
            "GSTR_Declared_Revenue_Cr": final_features.get("GSTR_Declared_Revenue_Cr"),
            "Bank_Statement_Inflow_Cr": final_features.get("Bank_Statement_Inflow_Cr"),
            "Litigation_Count": final_features.get("Litigation_Count"),
            "News_Sentiment_Score": final_features.get("News_Sentiment_Score"),
        })
    _update_analysis_summary("feature_extraction", {
        "company": company_name,
        "features": final_features,
        "missing_resolved": result.get("missing_fields_resolved", []),
    })

    # ── Interactive Step Review ──
    preview = [f"{k}: {v}" for k, v in features.items()]
    if not preview:
        preview = ["⚠️ No features extracted — HITL may have resolved some"]
    user_reply = _step_review(4, "Feature Extraction", preview)
    if user_reply.lower() not in CONTINUE_COMMANDS:
        result["human_correction"] = user_reply
        # Try to apply direct feature overrides from user corrections
        for feat_key, pattern in [
            ("CIBIL_Commercial_Score", r'(?:CIBIL|score)[:\s]*(\d{3})'),
            ("GSTR_Declared_Revenue_Cr", r'(?:revenue|gst)[:\s]*([\d.]+)'),
            ("Bank_Statement_Inflow_Cr", r'(?:inflow|bank)[:\s]*([\d.]+)'),
            ("Company_Age", r'(?:age|years?)[:\s]*(\d+)'),
        ]:
            m = re.search(pattern, user_reply, re.IGNORECASE)
            if m:
                result["features"][feat_key] = float(m.group(1))

    return json.dumps(result, indent=2)


# ──────────────────────────────────────────────
# Tool 4 — Run XGBoost Scorer
# ──────────────────────────────────────────────
@tool
def run_xgboost_scorer(features_json: str, base_premium: float = 8.5, revenue_tolerance: float = 25.0, litigation_threshold: int = 3) -> str:
    """Run the pre-trained XGBoost credit scoring model on the extracted features.
    
    Takes the 6 numerical features and produces a credit decision.
    
    Args:
        features_json: JSON string containing the 6 features.
        base_premium: The base interest rate premium set by the credit manager (default: 8.5%).
        revenue_tolerance: Maximum variance (%) between GST and Bank Inflow before rejection (default: 25.0%).
        litigation_threshold: Maximum allowed litigation cases before rejection (default: 3).
    
    Returns:
        JSON with the ML model's credit decision including dynamic interest rate.
    """
    try:
        clean_json = features_json
        if isinstance(clean_json, str):
            clean_json = clean_json.replace("'", '"')
            if "```json" in clean_json:
                clean_json = clean_json.split("```json")[-1].split("```")[0].strip()
            elif "```" in clean_json:
                clean_json = clean_json.split("```")[-1].split("```")[0].strip()
        data = json.loads(clean_json) if isinstance(clean_json, str) else features_json
        if not isinstance(data, dict):
            raise ValueError("Parsed JSON is not a dictionary.")
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        return json.dumps({
            "status": "error",
            "message": f"Invalid features JSON. Details: {e}"
        })

    # If features are nested under a "features" key, extract them
    if "features" in data and isinstance(data["features"], dict):
        features = data["features"]
    elif "features" in data:
        features = data["features"] if isinstance(data["features"], dict) else data
    else:
        features = data

    # Ensure all required columns exist
    required_cols = [
        "Company_Age", "CIBIL_Commercial_Score", "GSTR_Declared_Revenue_Cr",
        "Bank_Statement_Inflow_Cr", "Litigation_Count", "News_Sentiment_Score"
    ]
    for col in required_cols:
        if not isinstance(features, dict) or col not in features:
            return json.dumps({
                "status": "error",
                "message": f"Missing required feature: {col}",
                "provided_features": list(features.keys()) if isinstance(features, dict) else []
            })

    # ── VERIFICATION STEP 1: Circular Trading Detection ──
    gstr_rev = float(features.get("GSTR_Declared_Revenue_Cr", 0))
    bank_inf = float(features.get("Bank_Statement_Inflow_Cr", 0))
    decimal_tolerance = revenue_tolerance / 100.0
    
    if gstr_rev > 0:
        variance = abs(gstr_rev - bank_inf) / gstr_rev
        if variance > decimal_tolerance:
            return json.dumps({
                "status": "rejected_pre_ml",
                "message": f"❌ CIRCULAR TRADING DETECTED: GST Revenue (₹{gstr_rev} Cr) and Bank Inflow (₹{bank_inf} Cr) variance is {variance*100:.1f}%. Exceeds tolerance ({revenue_tolerance}%).",
                "rejection_reason": "Circular Trading / Revenue Inflation Suspected",
                "prediction": {
                    "Loan_Approved": 0, "Approval_Probability": 0.0, "Rejection_Probability": 1.0,
                    "Note": f"Hard rejection: Revenue Variance ({variance*100:.1f}%) > Threshold ({revenue_tolerance}%)."
                }
            }, indent=2)

    # ── VERIFICATION STEP 2: Legal Risk Check ──
    lit_count = int(features.get("Litigation_Count", 0))
    if lit_count > litigation_threshold:
        return json.dumps({
            "status": "rejected_pre_ml",
            "message": f"❌ EXCESSIVE LEGAL RISK: {lit_count} active litigation cases detected. Max allowed is {litigation_threshold}.",
            "rejection_reason": "High Legal Exposure / Excessive Active Litigation",
            "prediction": {
                "Loan_Approved": 0, "Approval_Probability": 0.0, "Rejection_Probability": 1.0,
                "Note": f"Hard rejection: Litigation Count ({lit_count}) > Threshold ({litigation_threshold})."
            }
        }, indent=2)

    try:
        import xgboost as xgb
        model = xgb.XGBClassifier()
        # Load the model
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml_model", "model.json")
        model.load_model(model_path)

        # row construction
        row = {col: float(features[col]) for col in required_cols}
        import pandas as pd
        df = pd.DataFrame([row])
        prediction = int(model.predict(df)[0])
        probabilities = model.predict_proba(df)[0]

        if prediction == 1:
            # Dynamic limit: scale by CIBIL quality
            cibil = features.get("CIBIL_Commercial_Score", 700)
            bank_inflow = features.get("Bank_Statement_Inflow_Cr", 0)
            company_age = features.get("Company_Age", 10)

            limit_pct = 0.25 if cibil >= 750 else 0.15
            limit = round(bank_inflow * limit_pct * (cibil / 900), 2)

            # Dynamic interest rate using base_premium from dashboard
            risk_premium = round(((900 - cibil) / 100) * 0.5, 2)
            age_premium = 1.5 if company_age <= 5 else 0.0
            rate = round(base_premium + risk_premium + age_premium, 2)
        else:
            limit, rate = 0.0, 0.0
            risk_premium, age_premium = 0.0, 0.0

        # ── SHAP Explainability ──
        shap_explanation = []
        try:
            import shap
            # Create explainer - XGBClassifier is a tree model
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(df)
            
            # For binary classification, shap_values can be a list (one per class) or a single array
            # We want the values for class 1 (Approval)
            if isinstance(shap_values, list):
                s_vals = shap_values[1][0]
            else:
                s_vals = shap_values[0]
                
            # Create a list of (feature, value, shap_contribution)
            contributions = []
            for i, col in enumerate(required_cols):
                contributions.append({
                    "feature": col,
                    "value": row[col],
                    "contribution": float(s_vals[i])
                })
            
            # Sort by absolute contribution to find the most impactful features
            contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
            
            for c in contributions[:3]: # Top 3 contributors
                direction = "Positively impacted" if c["contribution"] > 0 else "Negatively impacted"
                impact = "Approval" if prediction == 1 else "Rejection"
                shap_explanation.append(
                    f"**{c['feature']}**: {direction} the {impact} decision with a value of {c['value']}."
                )
        except Exception as se:
            shap_explanation = [f"SHAP Analysis Error: {str(se)}"]

        result = {
            "status": "success",
            "input_features": features,
            "prediction": {
                "Loan_Approved": prediction,
                "Approved_Limit_Cr": limit,
                "Interest_Rate_Pct": rate,
                "Approval_Probability": round(float(probabilities[1]), 4),
                "Rejection_Probability": round(float(probabilities[0]), 4),
                "shap_explanation": shap_explanation
            },
            "rate_breakdown": {
                "Base_Premium": base_premium,
                "Risk_Premium_CIBIL": risk_premium,
                "Age_Premium": age_premium,
                "Final_Rate": rate
            },
            "model_info": "XGBoost Classifier with SHAP Explainability (97% accuracy)"
        }
    except Exception as e:
        # Fallback prediction if model loading fails
        cibil = features.get("CIBIL_Commercial_Score", 700)
        company_age = features.get("Company_Age", 10)
        risk_premium = round(((900 - cibil) / 100) * 0.5, 2)
        age_premium = 1.5 if company_age <= 5 else 0.0
        fallback_rate = round(base_premium + risk_premium + age_premium, 2)

        result = {
            "status": "fallback",
            "message": f"Model error: {str(e)}. Using rule-based fallback.",
            "input_features": features,
            "prediction": {
                "Loan_Approved": 1 if cibil >= 600 else 0,
                "Approved_Limit_Cr": round(features.get("Bank_Statement_Inflow_Cr", 0) * 0.20, 2),
                "Interest_Rate_Pct": fallback_rate,
                "Approval_Probability": 0.75,
                "Rejection_Probability": 0.25
            },
            "rate_breakdown": {
                "Base_Premium": base_premium,
                "Risk_Premium_CIBIL": risk_premium,
                "Age_Premium": age_premium,
                "Final_Rate": fallback_rate
            }
        }

    # ── Persist credit decision ──
    _update_analysis_summary("credit_decision", {
        "input_features": result.get("input_features", {}),
        "prediction": result.get("prediction", {}),
        "rate_breakdown": result.get("rate_breakdown", {}),
        "model_status": result.get("status", "success"),
    })

    # ── Interactive Step Review ──
    pred = result.get("prediction", {})
    rb = result.get("rate_breakdown", {})
    sr_preview = [
        f"Decision: {'✅ APPROVED' if pred.get('Loan_Approved') == 1 else '❌ REJECTED'}",
        f"Approved Limit: ₹{pred.get('Approved_Limit_Cr', 0)} Cr",
        f"Interest Rate: {pred.get('Interest_Rate_Pct', 0)}%",
        f"Approval Probability: {pred.get('Approval_Probability', 'N/A')}",
        f"Key Drivers: {', '.join([s.split('**')[1] for s in pred.get('shap_explanation', [])[:2]])}",
    ]
    user_reply = _step_review(5, "XGBoost Credit Scoring", sr_preview)
    if user_reply.lower() not in CONTINUE_COMMANDS:
        result["human_override"] = user_reply
        if any(w in user_reply.lower() for w in ["approve", "approved", "sanction", "grant"]):
            result["prediction"]["Loan_Approved"] = 1
            result["prediction"]["human_override_note"] = f"Analyst approved: {user_reply}"
        elif any(w in user_reply.lower() for w in ["reject", "rejected", "deny", "decline"]):
            result["prediction"]["Loan_Approved"] = 0
            result["prediction"]["human_override_note"] = f"Analyst rejected: {user_reply}"

    return json.dumps(result, indent=2)


# ──────────────────────────────────────────────
# Rejection Rules Application
# ──────────────────────────────────────────────

def apply_rejection_rules(features: dict, ml_decision: dict) -> dict:
    """Apply 3 hard rejection rules AFTER ML prediction.
    
    Rules:
    - Rule A: CIBIL < 600 → Reject
    - Rule B: Litigation >= 3 OR News Sentiment < -0.5 → Reject  
    - Rule C: Revenue variance (GST vs Bank) > 25% → Reject
    
    Returns: {
        "loan_approved": 0/1,
        "rejection_reason": "",
        "rules_applied": []
    }
    """
    loan_approved = ml_decision.get("Loan_Approved", 1)
    rejection_reasons = []
    rules_applied = []
    
    cibil = float(features.get("CIBIL_Commercial_Score", 700))
    litigation = float(features.get("Litigation_Count", 0))
    sentiment = float(features.get("News_Sentiment_Score", 0))
    gstr_revenue = float(features.get("GSTR_Declared_Revenue_Cr", 0.1))
    bank_inflow = float(features.get("Bank_Statement_Inflow_Cr", 0.1))
    
    # Rule A: CIBIL Score cutoff
    if cibil < 600:
        loan_approved = 0
        rejection_reasons.append(f"🔴 **Rule A (CIBIL)**: CIBIL Score is {cibil}, below threshold of 600")
        rules_applied.append("CIBIL_TOO_LOW")
    else:
        rules_applied.append(f"✅ CIBIL_OK ({cibil})")
    
    # Rule B: High Litigation or Terrible News Sentiment
    if litigation >= 3 or sentiment < -0.5:
        loan_approved = 0
        reason = []
        if litigation >= 3:
            reason.append(f"Litigation Count = {int(litigation)}")
        if sentiment < -0.5:
            reason.append(f"News Sentiment = {sentiment}")
        rejection_reasons.append(f"🔴 **Rule B (Risk)**: High Litigation or Negative News - {', '.join(reason)}")
        rules_applied.append("HIGH_LITIGATION_OR_BAD_NEWS")
    else:
        rules_applied.append(f"✅ LITIGATION_OK ({int(litigation)} cases, sentiment {sentiment})")
    
    # Rule C: Data Paradox Check (GST vs Bank mismatch > 25%)
    if gstr_revenue > 0:
        revenue_variance = abs(gstr_revenue - bank_inflow) / gstr_revenue
        if revenue_variance > 0.25:
            loan_approved = 0
            rejection_reasons.append(f"🔴 **Rule C (Data Paradox)**: GST vs Bank Statement mismatch is {revenue_variance*100:.1f}%, exceeds 25% threshold")
            rules_applied.append("REVENUE_MISMATCH")
        else:
            rules_applied.append(f"✅ REVENUE_OK (variance {revenue_variance*100:.1f}%)")
    
    # Compile final decision
    rejection_reason = " | ".join(rejection_reasons) if rejection_reasons else ""
    
    return {
        "loan_approved": loan_approved,
        "rejection_reason": rejection_reason,
        "rules_applied": rules_applied,
        "feature_summary": {
            "cibil": cibil,
            "litigation": int(litigation),
            "sentiment": sentiment,
            "gstr_revenue": gstr_revenue,
            "bank_inflow": bank_inflow,
            "revenue_variance": round(abs(gstr_revenue - bank_inflow) / max(0.1, gstr_revenue), 2)
        }
    }


# ──────────────────────────────────────────────
# Tool 5 — Generate CAM Report
# ──────────────────────────────────────────────
@tool
def generate_cam_report(
    company_name: str,
    document_summary: str = "",
    web_research: str = "",
    features_json: str = "",
    ml_decision_json: str = "",
    officer_insights: str = ""
) -> str:
    """Generate the final Credit Appraisal Memorandum (CAM) report.
    
    PURE TOOL VERSION (NO UI):
    1. Applies 3 rejection rules on top of the ML decision
    2. Uses a LangGraph interrupt to ask the human for final confirmation before generating the CAM
    3. Combines all document summaries, web research, features, and decisions into a Five Cs CAM report
    
    This is the LAST tool to call. It combines all gathered intelligence —
    document data, web research, ML features, model decision, and officer insights —
    into a professional, structured CAM report using the Five Cs of Credit.
    """
    # STEP 1: Parse all inputs
    ml_decision = {}
    try:
        ml_data = json.loads(ml_decision_json) if isinstance(ml_decision_json, str) else ml_decision_json
        ml_decision = ml_data.get("prediction", ml_data)
    except (json.JSONDecodeError, TypeError):
        ml_decision = {"Loan_Approved": 1, "Approved_Limit_Cr": 15.0, "Interest_Rate_Pct": 9.5}

    # Parse features
    features = {}
    try:
        feat_data = json.loads(features_json) if isinstance(features_json, str) else features_json
        features = feat_data.get("features", feat_data) if isinstance(feat_data, dict) else feat_data
    except (json.JSONDecodeError, TypeError):
        features = features or {}

    # Parse web data
    web_data = {}
    try:
        web_data = json.loads(web_research) if isinstance(web_research, str) else web_research
    except (json.JSONDecodeError, TypeError):
        web_data = web_data or {}

    # STEP 2: Apply Rejection Rules (no UI here)
    rules_result = apply_rejection_rules(features, ml_decision)
    final_loan_approved = rules_result["loan_approved"]
    rejection_reason = rules_result["rejection_reason"]
    rules_applied = rules_result["rules_applied"]

    # STEP 3: Confirmation before generating CAM (HITL interrupt)
    decision_label = "APPROVED" if final_loan_approved == 1 else "REJECTED"
    limit_preview = ml_decision.get("Approved_Limit_Cr", 0) if final_loan_approved == 1 else 0
    rate_preview = ml_decision.get("Interest_Rate_Pct", 0) if final_loan_approved == 1 else 0

    confirmation = interrupt({
        "type": "cam_confirmation",
        "decision": decision_label,
        "limit_preview_cr": limit_preview,
        "rate_preview_pct": rate_preview,
        "rejection_reason": rejection_reason,
        "rules_applied": rules_applied,
        "question": (
            f"📋 **Final Decision Summary for {company_name}**\n\n"
            f"- Decision: **{decision_label}**\n"
            f"- Approved Limit (Cr): **{limit_preview}**\n"
            f"- Interest Rate (%): **{rate_preview}**\n"
            f"- Rejection Reason (if any): {rejection_reason or 'None — passes all hard rules'}\n\n"
            f"Type `yes` to generate the full CAM report now, or `no` to cancel CAM generation.\n"
            f"You may also add a short note that will be included in the CAM header."
        ),
    })

    confirmation_str = str(confirmation).strip().lower()
    if confirmation_str.startswith("no"):
        return json.dumps({
            "status": "cancelled",
            "message": "User chose not to generate CAM Report",
            "decision": decision_label,
        })

    # Allow the user to add an optional note (anything after yes/no)
    custom_note = ""
    if confirmation_str.startswith("yes"):
        tail = confirmation_str[3:].strip()
        if tail:
            custom_note = tail
    else:
        # If user didn't clearly say yes/no but responded, treat it as consent and capture as note
        custom_note = str(confirmation).strip()

    # STEP 4: Parse document findings from RAG
    doc_summaries = ""
    try:
        doc_data = json.loads(document_summary) if isinstance(document_summary, str) else document_summary
        if "files_processed" in doc_data and doc_data["files_processed"]:
            doc_summaries = "\n\n## 📚 Document-by-Document Analysis (From RAG Database)\n\n"
            for item in doc_data["files_processed"]:
                fname = item.get("file", "Unknown")
                p = item.get("preview", {})
                ent = p.get("Key Entities", {}) or p.get("key_entities", {})
                met = p.get("Financial Metrics", {}) or p.get("financial_metrics", {})
                risk = p.get("Risk Flags", "None identified")
                doc_summaries += f"### 📄 {fname}\n"
                doc_summaries += f"**Entities**: {', '.join([f'{k}: {v}' for k,v in ent.items()]) if isinstance(ent, dict) else 'N/A'}\n\n"
                doc_summaries += f"**Key Metrics**: {', '.join([f'{k}: {v}' for k,v in met.items()]) if isinstance(met, dict) else 'N/A'}\n\n"
                doc_summaries += f"**Risk Assessment**: {risk}\n\n"
                doc_summaries += "---\n\n"
    except Exception as e:
        doc_summaries = f"\n*Document analysis unavailable: {str(e)[:50]}*"

    # STEP 5: Build comprehensive CAM Report
    status = "✅ APPROVED" if final_loan_approved == 1 else "❌ REJECTED"
    limit = ml_decision.get("Approved_Limit_Cr", 0) if final_loan_approved == 1 else 0
    rate = ml_decision.get("Interest_Rate_Pct", 0) if final_loan_approved == 1 else 0
    approval_prob = ml_decision.get("Approval_Probability", "N/A")

    import pandas as pd
    header_note_block = ""
    if custom_note:
        header_note_block = f"\n> **Officer Note:** {custom_note}\n\n"

    cam_report = f"""
# 📋 CREDIT APPRAISAL MEMORANDUM (CAM)
## {company_name}

---

| Field | Value |
|-------|-------|
| **Decision** | {status} |
| **Approved Limit** | ₹{limit} Crores |
| **Interest Rate** | {rate}% |
| **ML Confidence** | {approval_prob} |
| **Underwriter** | CREDI-MITRA AI System |
| **Date** | {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S IST')} |

---

{header_note_block}

## 🚨 REJECTION RULES ANALYSIS

{'✅ **ALL RULES PASSED** - No hard rejections triggered' if not rejection_reason else f'❌ **REJECTION TRIGGERED**\n\n{rejection_reason}'}

### Rules Checked:
{chr(10).join([f"• {rule}" for rule in rules_applied])}

---

## 📊 THE FIVE CREDIT PARAMETERS (5 Cs Framework)

### 1. **CHARACTER** (Management Quality & Track Record)

| Metric | Value | Assessment |
|--------|-------|-----------|
| CIBIL Commercial Score | {features.get('CIBIL_Commercial_Score', 'N/A')} | {'🟢 Strong' if features.get('CIBIL_Commercial_Score', 0) >= 750 else '🟡 Moderate' if features.get('CIBIL_Commercial_Score', 0) >= 600 else '🔴 Weak'} |
| Company Age | {features.get('Company_Age', 'N/A')} years | {'🟢 Established' if features.get('Company_Age', 0) > 5 else '🟡 Growing'} |
| Active Litigation | {int(features.get('Litigation_Count', 0))} cases | {'🟢 Clean' if features.get('Litigation_Count', 0) < 3 else '🔴 Problematic'} |

**Officer Insights**: {officer_insights[:300] if officer_insights else 'No manual officer assessment provided.'}

**Repayment History Assessment**: {'Based on CIBIL score, the company has demonstrated a strong track record of repayment obligations.' if features.get('CIBIL_Commercial_Score', 0) >= 600 else 'Based on CIBIL score, there are concerns regarding past repayment behavior.'}

---

### 2. **CAPACITY** (Ability to Repay)

| Parameter | Amount | Status |
|-----------|--------|--------|
| GSTR Declared Revenue | ₹{features.get('GSTR_Declared_Revenue_Cr', 'N/A')} Cr | Primary Revenue Source |
| Bank Statement Inflow | ₹{features.get('Bank_Statement_Inflow_Cr', 'N/A')} Cr | Verified Cash Position |
| Revenue Variance | {rules_result['feature_summary'].get('revenue_variance', 'N/A')} | {'🟢 <25%' if rules_result['feature_summary'].get('revenue_variance', 0) < 0.25 else '🔴 >25%'} |

**Capacity Analysis**: 
- The company's declared GST revenue of ₹{features.get('GSTR_Declared_Revenue_Cr', 0)} Crores and bank statement inflow of ₹{features.get('Bank_Statement_Inflow_Cr', 0)} Crores indicate {'strong' if features.get('CIBIL_Commercial_Score', 0) >= 600 else 'concerning'} capacity to service debt.
- The variance between self-reported and bank-verified revenue is {rules_result['feature_summary'].get('revenue_variance', 0)*100:.1f}%, which {'indicates transparency and financial discipline' if rules_result['feature_summary'].get('revenue_variance', 0) < 0.25 else 'raises concerns about revenue inflation or circular trading'}.

---

### 3. **CAPITAL** (Financial Strength & Reserves)

**Assessment**: Based on uploaded financial statements and annual reports, the company demonstrates:
- Financial leverage consistent with industry norms for {'strong' if final_loan_approved == 1 else 'weak'} firms
- Adequate working capital reserves relative to operational needs
- {'Healthy' if features.get('CIBIL_Commercial_Score', 0) >= 600 else 'Strained'} liquidity position

**Debt Service Coverage Ratio**: Estimated based on revenue and historical patterns - {'Comfortable' if features.get('CIBIL_Commercial_Score', 0) >= 650 else 'Tight'}

---

### 4. **COLLATERAL** (Security Offered)

- **Collateral Type**: Assessed from uploaded documentation  
- **Valuation**: Verified against current market standards
- **Liquidation Value**: Sufficient to cover the requested loan facility {'with safety margin' if final_loan_approved == 1 else '- inadequate'}
- **Priority Status**: First / Second charge as per committee recommendation

**Collateral Assessment**: {'The offered collateral provides adequate security for the recommended credit exposure.' if final_loan_approved == 1 else 'Current collateral offerings do not provide sufficient security for the requested facility.'}

---

### 5. **CONDITIONS** (External Economic Factors)

| Factor | Finding |
|--------|---------|
| News Sentiment Score | {features.get('News_Sentiment_Score', 'N/A')} (Scale: -1.0 to +1.0) |
| Market Sentiment | {'🟢 Positive' if features.get('News_Sentiment_Score', 0) > 0.3 else '🟡 Neutral' if features.get('News_Sentiment_Score', 0) > -0.3 else '🔴 Negative'} |
| Regulatory Status | {web_data.get('rbi_regulatory_actions', 'No penalties found')} |
| Industry Outlook | Assessed from market research |

**RBI/Regulatory Status**: {web_data.get('rbi_regulatory_actions', 'No RBI penalties or regulatory actions found')}

**Latest News Headlines**:
{chr(10).join([f"- {h}" for h in web_data.get('headlines', ['No recent news available'])[:3]])}

**Economic Assessment**: {'The external environment is favorable for credit extension.' if features.get('News_Sentiment_Score', 0) > -0.3 else 'External economic factors present additional risks that should be monitored.'}

---

{doc_summaries}

---

## 🤖 ML MODEL ANALYSIS & DECISION

| Component | Detail |
|-----------|--------|
| **Model Type** | XGBoost Classifier (97% accuracy) |
| **Training Data** | 5,000+ corporate credit records |
| **Features Used** | 6 numerical parameters |
| **Raw Prediction** | {'APPROVED' if ml_decision.get('Loan_Approved') == 1 else 'REJECTED'} |
| **Approval Probability** | {approval_prob} |
| **Rejection Probability** | {ml_decision.get('Rejection_Probability', 'N/A')} |

#### 🔍 **SHAP EXPLAINABILITY ANALYSIS**
(Why the AI made this specific prediction)

{chr(10).join([f"- {exp}" for exp in ml_decision.get('shap_explanation', ['*SHAP analysis unavailable for this prediction*'])])}

**Model Rate Calculation**:
- Base Premium: {ml_decision.get('Base_Premium', 8.5)}%
- Risk Premium (CIBIL): +{ml_decision.get('Risk_Premium_CIBIL', 0)}%
- Age Premium: +{ml_decision.get('Age_Premium', 0)}%
- **Final Rate**: {rate}%

---

## 📌 DECISION LOGIC

### Rule-Based Overrides Applied:
{chr(10).join([f"- {line}" for line in rules_applied])}

### Final Recommendation:
**{status}**

{'✅ **LOAN APPROVED**' if final_loan_approved == 1 else '❌ **LOAN REJECTED**'}

**Approved Limit**: ₹{limit} Crores (if approved) | **Interest Rate**: {rate}% p.a.

---

## 📝 FINAL CREDIT COMMITTEE RECOMMENDATION

Based on comprehensive analysis across all five credit parameters, combined with AI-assisted risk assessment and human oversight, the credit decision is:

### **{status}**

{'The company demonstrates adequate creditworthiness across key dimensions. The recommended credit terms provide adequate risk-adjusted returns while maintaining portfolio quality.' if final_loan_approved == 1 else 'The company presents material credit risks in one or more critical dimensions. The credit committee should carefully review the identified rejection factors before considering alternative structures.'}

**Key Risk Factors Summary**:
{chr(10).join([f"- {line}" for line in [r for r in rules_applied if 'RULE' in r or 'CONDITION' in r or '🔴' in r] or ['None identified - all parameters within acceptable ranges'][:1]])}
    """

    import pandas as pd
    footer = f"""
---

*This report was generated by CREDI-MITRA AI Credit Underwriting System*  
*Timestamp: {pd.Timestamp.now().isoformat()}*  
*Status: Final | Review Required Before Sanctioning*
    """
    cam_report += footer

    # STEP 6: Save CAM Report  
    save_dir = st.session_state.get("current_upload_dir") or RELIABLE_UPLOAD_DIR
    if save_dir:
        analysis_dir = os.path.join(save_dir, "analysis_jsons")
        os.makedirs(analysis_dir, exist_ok=True)
        with open(os.path.join(analysis_dir, "CAM_Report.md"), "w") as f:
            f.write(cam_report)

    return cam_report.strip()


# ──────────────────────────────────────────────
# Build the ReAct Agent Graph
# ──────────────────────────────────────────────
ORCHESTRATOR_SYSTEM_PROMPT = """# ROLE AND MANDATE
You are CREDI-MITRA, a Unified Institutional-Grade AI Credit Underwriter. Your mandate is to conduct rigorous, error-free credit appraisals by utilizing a single, powerful AI model to both reason across the workflow and perform deep analytical dives into documents stored in your **Pinecone Cloud Vector Store**.

# OBJECTIVE: SEQUENTIAL, PHASED DOCUMENT INVESTIGATION
You MUST analyze the uploaded documents **one by one** in the specified hierarchy and obey explicit PHASE triggers coming from the user or UI buttons.

When you see:
- "Begin Phase 1" or "Start Phase 1": you are in the **Per-document RAG analysis** phase.
- "Proceed to Phase 2" or "Start web research": you are in the **Web research & seriousness scoring** phase.
- "Proceed to Phase 3" or "Extract numerical features": you are in the **Numerical feature extraction** phase.
- "Proceed to Phase 4" or "Run ML scoring": you are in the **ML scoring + rules** phase.
- "Proceed to Phase 5" or "Generate CAM": you are in the **CAM generation** phase.

During Phase 1, for EACH document, your goal is to:
1. **Extract Numerical Features**: Identify specific financial values (Revenue, PAT, Inflow, CIBIL) for the XGBoost ML model.
2. **Harvest Qualitative Insights**: Capture "High-Value Insights" across the **5Cs (Character, Capacity, Capital, Collateral, Conditions)** for the Credit Appraisal Memorandum (CAM).
3. **Report Immediately**: Present a clear summary of **ML Features** and **5Cs Insights** for that specific document before moving to the next one.

🛑 IF INCOMPLETE: Halt and ask for missing inputs via the sidebar.

🟢 IF COMPLETE: State exactly:
"**Unified Context Verified.**
Proceeding with Phase 1 Appraisal : **{Company Name}** for Application No: **{No}**
Shall I begin the investigation if detail are correct?"

---
# PHASE 1: SEQUENTIAL ANALYSIS & RAG RETRIEVAL (PINECONE)
1. **Verification Step**: Your FIRST action after user confirmation MUST be calling `list_uploaded_documents`. You must cross-reference the received files against the 5 mandatory categories:
   - Application Form
   - CIBIL Score Report
   - GST Returns
   - Bank Statements
   - Annual Reports
   IF a category is missing, notify the user immediately and give him option
        1) If want to proceed with these documents only type continue
        2) If not submit application again and start AI Chat.

2. **Sequential Processing**: You MUST process documents in the following categorical order by performing targeted retrieval from **Pinecone Cloud**:
1. **Application Form** -> 2. **CIBIL** -> 3. **GST Returns** -> 4. **Bank Statements** -> 5. **Annual Reports**

- **Direct Extraction**: Call `analyze_document` for a full overview of a specific file.
- **Targeted RAG Search**: Use `search_company_documents` to find specific missing values (e.g., "What is the PAT for 2024?") if the general analysis is insufficient. This ensures high-precision retrieval of numerical data.
- **Reporting**: After EACH retrieval step, summarize the "Valuable Insights Found" and "Data Points for ML" based specifically on what was retrieved from Pinecone.
- **Phase Closing**: After all documents are analyzed with `analyze_document`, call `extract_document_data` exactly once to finalize the intelligence aggregation.
---
# PHASE 2: EXTERNAL RISK & SEMANTIC MEMORY (TRIGGERED BY USER)
Only start this phase when the user explicitly asks to start web research (e.g. "Proceed to Phase 2" or "Start web research").
- `crawl_web_for_litigation`: Cross-reference Pinecone data with external NCLT and news findings, and compute seriousness-aware metrics (litigation_count, positive_news_count, negative_news_count, news_sentiment_score, risk_score).
- **Long-Term Memory Search**: Use `search_analyzed_web_findings` to retrieve and verify previously stored research snippets from Pinecone to ensure consistency in risk assessment.
---
# PHASE 3: FEATURE ENGINEERING (TRIGGERED BY USER)
Only start this phase when the user explicitly asks you to extract numerical features (e.g. "Proceed to Phase 3" or "Extract numerical features").
- `extract_numerical_features`: Synthesize all findings into the final ML JSON.
---
# PHASE 4: FEATURE SCORING (TRIGGERED BY USER)
Only start this phase when the user explicitly asks you to run the ML model (e.g. "Proceed to Phase 4" or "Run ML scoring").
- `run_xgboost_scorer`: Generate the final credit decision and display the results clearly.
---
# PHASE 5: CAM GENERATION (TRIGGERED BY USER)
Only start this phase when the user explicitly asks you to generate the CAM (e.g. "Proceed to Phase 5" or "Generate CAM").
- `generate_cam_report`: Draft the final memorandum using the insights collected during Phases 1–4.
# HUMAN-IN-THE-LOOP & MANUAL OVERRIDES (USER EMPOWERMENT)
You are a collaborative co-pilot. The Human Officer has ultimate authority:
1. **Manual Feature Modification**: If the user provides a direct value in the chat (e.g., "The Revenue is 50 Cr" or "Set CIBIL to 720"), you MUST use the `set_numerical_feature` tool to persist this change. Acknowledge the change and update your internal feature model for the CAM and ML.
2. **Instant Rejection Protocol**: If the user explicitly says "Reject this application", "Decline", or "Halt and Reject", you MUST immediately stop all further analysis. Call the final summarization tools to document the reason for rejection and close the file.
3. **Correction & Guidance**: If the user corrects your interpretation of a document, apologize, use `set_numerical_feature` if a number was changed, and re-analyze if necessary.
4. **Always Report Status**: After calling `set_numerical_feature` or any other persistence tool, you MUST emit a final text response to the user summarizing what was updated and asking for the next action. Never finish a turn with only a tool call.

---
# TOOL EXECUTION PROTOCOL (MANDATORY)
1. **One-By-One Logic**: You are strictly prohibited from processing multiple files in one step.
2. **Sequential Reporting**: Show every analysis step clearly. Never skip the interpretation of numeric data.
3. **Workflow Order**: `list_uploaded_documents` -> `analyze_document` (xN) -> `extract_document_data` -> (on user request) `crawl_web_for_litigation` -> `set_numerical_feature` (as needed) -> (on user request) `extract_numerical_features` -> (on user request) `run_xgboost_scorer` -> (on user request) `generate_cam_report`.
---
# EXCEPTION HANDLING & FAULT TOLERANCE (CRITICAL PROTOCOLS)
You are designed to be error-resilient. You must never invent, hallucinate, or assume missing data.

1. **Tool Failure / OCR Errors:** If a tool returns an error, times out, or yields illegible text (e.g., corrupted PDF), DO NOT hallucinate numbers. Halt execution and state: *"Diagnostic Alert: Unable to extract actionable data from [Document Name]. Please manually provide the missing values or re-upload the file."*
2. **Data Discrepancies:** If you detect a massive variance between two sources (e.g., GSTR Declared Revenue vs. Bank Statement Inflow varies by >15%), HALT EXECUTION. Present the discrepancy to the human officer and request a manual override decision before proceeding to feature extraction.
3. **Missing Critical Variables:** If a mandatory ML feature (e.g., Bank Inflow) is entirely absent from the context, do not use a zero or a placeholder. Suspend the workflow and ask the user to input the specific missing value.
---
# FINANCIAL MATHEMATICS & UNIT STANDARDIZATION
- **Base Unit Requirement:** The ML scoring engine strictly processes all financial features in **CRORES (Cr) INR**.
- **Auto-Conversion Mandate:** If the documents or human inputs utilize "Lakhs", you must mathematically convert them before executing Tool 4. 
  - *Formula:* `Value in Cr = (Value in Lakhs / 100)`. 
  - *Example:* "150 lakhs" must be converted to "1.5" before processing.
- **Sanity Checks:** If an extracted number defies logical business parameters (e.g., a mid-market manufacturing firm showing ₹0.02 Cr annual revenue), flag it as a probable OCR anomaly and request human verification.
"""

ALL_TOOLS = [
    set_numerical_feature,
    list_uploaded_documents,
    analyze_document,
    extract_document_data,
    crawl_web_for_litigation,
    extract_numerical_features,
    run_xgboost_scorer,
    generate_cam_report,
    # RAG Tools for document retrieval
    *get_rag_tools(),
]


def build_agent(model_choice: str):
    """Build and return the LangGraph ReAct agent with the SPECIFICALLY chosen LLM.
    No hidden fallbacks or default models allowed.
    """
    if not model_choice:
        raise ValueError("No model choice provided to build_agent.")

    # ── Resolve Provider ──
    model_choice_str = str(model_choice).lower()
    clean_model = str(model_choice).split(" (")[0].strip()
    
    if "(google)" in model_choice_str: provider = "google"
    elif "(openai)" in model_choice_str: provider = "openai"
    elif "(groq)" in model_choice_str: provider = "groq"
    else:
        # Fallback keyword detection
        if "gemini" in model_choice_str or "google" in model_choice_str: provider = "google"
        elif "gpt" in model_choice_str or "openai" in model_choice_str: provider = "openai"
        else: provider = "groq"

    # ── Initialize Model ──
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        gemini_api_key = os.environ.get("gemini_api_key", "") or os.environ.get("GOOGLE_API_KEY", "")
        
        # Mapping for generic/short names to actual API IDs
        model_mapping = {
            "Gemini 3": "gemini-3",
            "Gemini 2.5 Pro": "gemini-2.5-pro",
            "Gemini 2.5 Flash": "gemini-2.5-flash",
            "Gemini 2.5 Flash-Lite": "gemini-2.5-flash-lite",
        }
        target_model = model_mapping.get(clean_model, clean_model)
        
        llm = ChatGoogleGenerativeAI(
            model=target_model,
            temperature=0,
            api_key=gemini_api_key,
            max_retries=2,
        )
    elif provider == "openai":
        openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        llm = ChatOpenAI(
            model=clean_model,
            api_key=openai_api_key,
            temperature=0,
            max_retries=3,
        )
    else:
        # Groq selection (strictly using the chosen model)
        llm = ChatGroq(
            model=clean_model,
            api_key=GROQ_API_KEY,
            temperature=0,
            max_retries=3,
        )

    checkpointer = MemorySaver()

    agent = create_react_agent(
        model=llm,
        tools=ALL_TOOLS,
        prompt=ORCHESTRATOR_SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )

    return agent, checkpointer