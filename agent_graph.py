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
import xgboost as xgb
import pandas as pd
from typing import Annotated, Any, Dict
from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

load_dotenv()

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

# ──────────────────────────────────────────────
# Tool 1 — Extract PDF Data
# ──────────────────────────────────────────────
@tool
def extract_pdf_data(pdf_text: str) -> str:
    """Extract and summarize key financial data from uploaded PDF documents.
    
    Call this tool FIRST to parse the raw PDF text provided by the user.
    It returns a structured summary of the financial information found
    in the uploaded documents (Application Form, CIBIL Report, GST Returns,
    Bank Statements, Annual Reports, and Officer Insights).
    
    Args:
        pdf_text: The raw text content extracted from the uploaded PDF documents.
    
    Returns:
        A structured summary of all financial data found in the documents.
    """
    if not pdf_text or not pdf_text.strip():
        return json.dumps({
            "status": "warning",
            "message": "No PDF text was provided. Ask the user to upload documents.",
            "data": {}
        })

    # Identify document sections
    sections_found = []
    section_markers = [
        "Application_Form", "CIBIL_Score_Report", "GST_Returns",
        "Bank_Statements", "Annual_Reports", "Officer_Insights_Report"
    ]
    for marker in section_markers:
        if marker in pdf_text:
            sections_found.append(marker.replace("_", " "))

    # Extract key numbers via pattern matching
    import re
    numbers = {}

    # Try to find CIBIL score
    cibil_match = re.search(r'(?:CIBIL|cibil|credit\s*score)[:\s]*(\d{3})', pdf_text, re.IGNORECASE)
    if cibil_match:
        numbers["CIBIL_Score"] = int(cibil_match.group(1))

    # Try to find revenue figures
    revenue_match = re.search(r'(?:revenue|turnover|sales)[:\s]*(?:Rs\.?|INR|₹)?\s*([\d,.]+)\s*(?:Cr|Crore|cr)', pdf_text, re.IGNORECASE)
    if revenue_match:
        numbers["Revenue_Cr"] = float(revenue_match.group(1).replace(",", ""))

    # Try to find bank inflow
    inflow_match = re.search(r'(?:inflow|bank\s*inflow|total\s*inflow)[:\s]*(?:Rs\.?|INR|₹)?\s*([\d,.]+)\s*(?:Cr|Crore|cr)?', pdf_text, re.IGNORECASE)
    if inflow_match:
        numbers["Bank_Inflow_Cr"] = float(inflow_match.group(1).replace(",", ""))

    result = {
        "status": "success",
        "sections_found": sections_found,
        "document_length_chars": len(pdf_text),
        "key_numbers_extracted": numbers,
        "raw_text_preview": pdf_text[:2000] + ("..." if len(pdf_text) > 2000 else "")
    }
    return json.dumps(result, indent=2)


# ──────────────────────────────────────────────
# Tool 2 — Crawl Web for Litigation (HITL)
# ──────────────────────────────────────────────
@tool
def crawl_web_for_litigation(company_name: str) -> str:
    """Search the web for litigation records, NCLT filings, regulatory actions,
    and news sentiment for a given company.
    
    This tool searches public databases and news sources for any legal disputes,
    NCLT (National Company Law Tribunal) filings, RBI regulatory actions, and
    general news sentiment about the company.
    
    IMPORTANT: If the company name is ambiguous or multiple matches are found,
    this tool will PAUSE and ask the user for clarification before proceeding.
    
    Args:
        company_name: The name of the company to search for.
    
    Returns:
        JSON with litigation data, news sentiment, and any regulatory findings.
    """
    if not company_name or not company_name.strip():
        return json.dumps({
            "status": "error",
            "message": "No company name provided. Please provide the company name."
        })

    company_name = company_name.strip()

    # ── Simulate ambiguity detection ──
    # In production, this would come from actual web search results
    ambiguous_names = ["infosys", "tata", "reliance", "bajaj", "mahindra", "adani", "birla", "godrej"]
    is_ambiguous = any(keyword in company_name.lower() for keyword in ambiguous_names)

    if is_ambiguous:
        # HUMAN-IN-THE-LOOP: Interrupt and ask the user for clarification
        clarification = interrupt({
            "question": (
                f"⚠️ **Ambiguous Company Name Detected**\n\n"
                f"While searching for **'{company_name}'**, I found multiple entities "
                f"with similar names in NCLT filings and public databases:\n\n"
                f"1. **{company_name} Technologies Ltd.** — IT Services (Mumbai)\n"
                f"2. **{company_name} Industrial Solutions Pvt. Ltd.** — Manufacturing (Pune)\n"
                f"3. **{company_name} Finance & Leasing Co.** — NBFC (Delhi)\n\n"
                f"Which company are you analyzing? Please reply with the number (1, 2, or 3) "
                f"or provide the full legal entity name."
            ),
            "type": "clarification_needed"
        })
        # When resumed, `clarification` contains the user's answer
        company_name = f"{company_name} (User clarified: {clarification})"

    # ── Simulate web research results ──
    import random
    random.seed(hash(company_name) % 1000)
    litigation_count = random.choice([0, 0, 0, 1, 2, 3])
    sentiment_score = round(random.uniform(-0.3, 0.8), 2)

    nclt_cases = []
    if litigation_count > 0:
        case_types = ["IBC Insolvency Petition", "Winding Up Petition", "Oppression & Mismanagement", "Scheme of Arrangement"]
        for i in range(litigation_count):
            nclt_cases.append({
                "case_id": f"NCLT/{random.randint(2020,2025)}/{random.randint(100,999)}",
                "type": random.choice(case_types),
                "status": random.choice(["Pending", "Disposed", "Under Hearing"]),
                "bench": random.choice(["Mumbai Bench", "Delhi Bench", "Chennai Bench"])
            })

    news_headlines = [
        f"{company_name} reports steady growth in Q3 FY25",
        f"Sector outlook: RBI policies favorable for {company_name}'s industry",
        f"{company_name} announces expansion plans in Tier-2 cities"
    ]
    if sentiment_score < 0:
        news_headlines.append(f"Concerns raised over {company_name}'s debt-to-equity ratio")

    result = {
        "status": "success",
        "company_searched": company_name,
        "litigation_count": litigation_count,
        "nclt_cases": nclt_cases,
        "news_sentiment_score": sentiment_score,
        "news_headlines": news_headlines,
        "rbi_regulatory_actions": "None found",
        "source": "Simulated (Tavily/NCLT Public Database)"
    }
    return json.dumps(result, indent=2)


# ──────────────────────────────────────────────
# Tool 3 — Extract Numerical Features (HITL)
# ──────────────────────────────────────────────
@tool
def extract_numerical_features(
    pdf_summary: str,
    web_research: str,
    company_name: str
) -> str:
    """Convert unstructured text data into the 6 numerical features required
    by the XGBoost ML model.
    
    This tool extracts: Company_Age, CIBIL_Commercial_Score,
    GSTR_Declared_Revenue_Cr, Bank_Statement_Inflow_Cr, Litigation_Count,
    and News_Sentiment_Score from the combined PDF and web research data.
    
    IMPORTANT: If any critical feature cannot be found in the provided data,
    this tool will PAUSE and ask the user to manually input the missing value.
    
    Args:
        pdf_summary: The structured summary from extract_pdf_data tool.
        web_research: The results from crawl_web_for_litigation tool.
        company_name: Name of the company being analyzed.
    
    Returns:
        JSON with all 6 numerical features ready for XGBoost.
    """
    features = {}
    missing_features = []

    # Try parsing the PDF summary
    pdf_data = {}
    try:
        pdf_data = json.loads(pdf_summary) if isinstance(pdf_summary, str) else pdf_summary
    except (json.JSONDecodeError, TypeError):
        pass

    # Try parsing web research
    web_data = {}
    try:
        web_data = json.loads(web_research) if isinstance(web_research, str) else web_research
    except (json.JSONDecodeError, TypeError):
        pass

    key_numbers = pdf_data.get("key_numbers_extracted", {})

    # ── Feature 1: Company Age ──
    # Typically from Application Form or Annual Report
    if "Company_Age" in key_numbers:
        features["Company_Age"] = key_numbers["Company_Age"]
    else:
        features["Company_Age"] = 10  # Reasonable default
        missing_features.append("Company_Age")

    # ── Feature 2: CIBIL Score ──
    if "CIBIL_Score" in key_numbers:
        features["CIBIL_Commercial_Score"] = key_numbers["CIBIL_Score"]
    else:
        missing_features.append("CIBIL_Commercial_Score")

    # ── Feature 3: GSTR Revenue ──
    if "Revenue_Cr" in key_numbers:
        features["GSTR_Declared_Revenue_Cr"] = key_numbers["Revenue_Cr"]
    else:
        missing_features.append("GSTR_Declared_Revenue_Cr")

    # ── Feature 4: Bank Statement Inflow ──
    if "Bank_Inflow_Cr" in key_numbers:
        features["Bank_Statement_Inflow_Cr"] = key_numbers["Bank_Inflow_Cr"]
    else:
        missing_features.append("Bank_Statement_Inflow_Cr")

    # ── Feature 5: Litigation Count (from web research) ──
    if "litigation_count" in web_data:
        features["Litigation_Count"] = web_data["litigation_count"]
    else:
        features["Litigation_Count"] = 0

    # ── Feature 6: News Sentiment (from web research) ──
    if "news_sentiment_score" in web_data:
        features["News_Sentiment_Score"] = web_data["news_sentiment_score"]
    else:
        features["News_Sentiment_Score"] = 0.0

    # ── HUMAN-IN-THE-LOOP: Ask for missing critical features ──
    critical_missing = [f for f in missing_features if f in [
        "CIBIL_Commercial_Score", "GSTR_Declared_Revenue_Cr", "Bank_Statement_Inflow_Cr"
    ]]

    if critical_missing:
        # Build a clear question for the user
        missing_descriptions = {
            "CIBIL_Commercial_Score": "CIBIL Commercial Score (typically 300–900)",
            "GSTR_Declared_Revenue_Cr": "GSTR Declared Revenue in Crores (e.g., 120.5)",
            "Bank_Statement_Inflow_Cr": "Bank Statement Total Inflow in Crores (e.g., 115.0)"
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
                f"`CIBIL: 750, Revenue: 120.5, Inflow: 115.0`"
            ),
            "type": "missing_data",
            "missing_fields": critical_missing
        })

        # Parse the user's response
        import re
        response_text = str(human_response)
        
        if "CIBIL_Commercial_Score" in critical_missing:
            match = re.search(r'(?:CIBIL|score)[:\s]*(\d{3,4})', response_text, re.IGNORECASE)
            if match:
                features["CIBIL_Commercial_Score"] = int(match.group(1))
            else:
                # Try to find any 3-digit number
                nums = re.findall(r'\b(\d{3})\b', response_text)
                features["CIBIL_Commercial_Score"] = int(nums[0]) if nums else 700

        if "GSTR_Declared_Revenue_Cr" in critical_missing:
            match = re.search(r'(?:revenue|gstr|gst)[:\s]*([\d.]+)', response_text, re.IGNORECASE)
            if match:
                features["GSTR_Declared_Revenue_Cr"] = float(match.group(1))
            else:
                nums = re.findall(r'(\d+\.?\d*)', response_text)
                features["GSTR_Declared_Revenue_Cr"] = float(nums[1]) if len(nums) > 1 else 100.0

        if "Bank_Statement_Inflow_Cr" in critical_missing:
            match = re.search(r'(?:inflow|bank)[:\s]*([\d.]+)', response_text, re.IGNORECASE)
            if match:
                features["Bank_Statement_Inflow_Cr"] = float(match.group(1))
            else:
                nums = re.findall(r'(\d+\.?\d*)', response_text)
                features["Bank_Statement_Inflow_Cr"] = float(nums[-1]) if nums else 100.0

    result = {
        "status": "success",
        "company": company_name,
        "features": features,
        "missing_fields_resolved": critical_missing if critical_missing else [],
        "notes": "All 6 features ready for XGBoost scoring."
    }
    return json.dumps(result, indent=2)


# ──────────────────────────────────────────────
# Tool 4 — Run XGBoost Scorer
# ──────────────────────────────────────────────
@tool
def run_xgboost_scorer(features_json: str) -> str:
    """Run the pre-trained XGBoost credit scoring model on the extracted features.
    
    Takes the 6 numerical features and produces a credit decision:
    Loan_Approved (0/1), Approved_Limit_Cr, and Interest_Rate_Pct.
    
    Args:
        features_json: JSON string containing the 6 features:
            Company_Age, CIBIL_Commercial_Score, GSTR_Declared_Revenue_Cr,
            Bank_Statement_Inflow_Cr, Litigation_Count, News_Sentiment_Score
    
    Returns:
        JSON with the ML model's credit decision.
    """
    try:
        data = json.loads(features_json) if isinstance(features_json, str) else features_json
    except (json.JSONDecodeError, TypeError):
        return json.dumps({
            "status": "error",
            "message": "Invalid features JSON. Please provide valid feature data."
        })

    # If features are nested under a "features" key, extract them
    if "features" in data:
        features = data["features"]
    else:
        features = data

    # Ensure all required columns exist
    required_cols = [
        "Company_Age", "CIBIL_Commercial_Score", "GSTR_Declared_Revenue_Cr",
        "Bank_Statement_Inflow_Cr", "Litigation_Count", "News_Sentiment_Score"
    ]
    for col in required_cols:
        if col not in features:
            return json.dumps({
                "status": "error",
                "message": f"Missing required feature: {col}",
                "provided_features": list(features.keys())
            })

    try:
        model = xgb.XGBClassifier()
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "model.json")
        model.load_model(model_path)

        df = pd.DataFrame([{col: features[col] for col in required_cols}])
        prediction = int(model.predict(df)[0])
        probabilities = model.predict_proba(df)[0].tolist()

        if prediction == 1:
            limit = round(features.get("Bank_Statement_Inflow_Cr", 0) * 0.20, 2)
            rate = 9.5
        else:
            limit, rate = 0.0, 0.0

        result = {
            "status": "success",
            "input_features": features,
            "prediction": {
                "Loan_Approved": prediction,
                "Approved_Limit_Cr": limit,
                "Interest_Rate_Pct": rate,
                "Approval_Probability": round(probabilities[1], 4),
                "Rejection_Probability": round(probabilities[0], 4)
            },
            "model_info": "XGBoost Classifier (pre-trained on 5000 synthetic corporate credit records)"
        }
    except Exception as e:
        # Fallback prediction if model loading fails
        result = {
            "status": "fallback",
            "message": f"Model error: {str(e)}. Using rule-based fallback.",
            "input_features": features,
            "prediction": {
                "Loan_Approved": 1 if features.get("CIBIL_Commercial_Score", 0) >= 600 else 0,
                "Approved_Limit_Cr": round(features.get("Bank_Statement_Inflow_Cr", 0) * 0.20, 2),
                "Interest_Rate_Pct": 9.5,
                "Approval_Probability": 0.75,
                "Rejection_Probability": 0.25
            }
        }

    return json.dumps(result, indent=2)


# ──────────────────────────────────────────────
# Tool 5 — Generate CAM Report
# ──────────────────────────────────────────────
@tool
def generate_cam_report(
    company_name: str,
    pdf_summary: str,
    web_research: str,
    features_json: str,
    ml_decision_json: str,
    officer_insights: str
) -> str:
    """Generate the final Credit Appraisal Memorandum (CAM) report.
    
    This is the LAST tool to call. It combines all gathered intelligence —
    PDF data, web research, ML features, model decision, and officer insights —
    into a professional, structured CAM report using the Five Cs of Credit.
    
    Args:
        company_name: Name of the company.
        pdf_summary: Output from extract_pdf_data tool.
        web_research: Output from crawl_web_for_litigation tool.
        features_json: Output from extract_numerical_features tool.
        ml_decision_json: Output from run_xgboost_scorer tool.
        officer_insights: Manual officer notes or uploaded report text.
    
    Returns:
        A complete, professional CAM report in Markdown format.
    """
    # Parse the ML decision
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
        features = feat_data.get("features", feat_data)
    except (json.JSONDecodeError, TypeError):
        pass

    # Parse web data
    web_data = {}
    try:
        web_data = json.loads(web_research) if isinstance(web_research, str) else web_research
    except (json.JSONDecodeError, TypeError):
        pass

    status = "✅ APPROVED" if ml_decision.get("Loan_Approved") == 1 else "❌ REJECTED"
    limit = ml_decision.get("Approved_Limit_Cr", 0)
    rate = ml_decision.get("Interest_Rate_Pct", 0)
    approval_prob = ml_decision.get("Approval_Probability", "N/A")

    cam_report = f"""
# 📋 Credit Appraisal Memorandum (CAM)

---

### Company: **{company_name}**
### Decision: **{status}**
### Approved Limit: **₹{limit} Crores** | Interest Rate: **{rate}%**
### ML Confidence: **{approval_prob}**

---

## 1. Character (Management Quality & Track Record)

- **Company Age**: {features.get('Company_Age', 'N/A')} years
- **CIBIL Commercial Score**: {features.get('CIBIL_Commercial_Score', 'N/A')}
- **Litigation History**: {features.get('Litigation_Count', 0)} active cases found
- **Officer Assessment**: {officer_insights[:500] if officer_insights else 'No manual insights provided.'}

---

## 2. Capacity (Ability to Repay)

- **GSTR Declared Revenue**: ₹{features.get('GSTR_Declared_Revenue_Cr', 'N/A')} Crores
- **Bank Statement Inflow**: ₹{features.get('Bank_Statement_Inflow_Cr', 'N/A')} Crores
- **Revenue-Inflow Variance**: The difference between declared GST revenue and actual bank inflow is a critical indicator of circular trading or revenue inflation.

---

## 3. Capital (Financial Strength)

- **Analysis**: Based on the uploaded financial documents and annual reports, the company demonstrates {'adequate' if ml_decision.get('Loan_Approved') == 1 else 'insufficient'} capital reserves relative to the requested credit facility.

---

## 4. Collateral (Security Offered)

- **Assessment**: Collateral evaluation is based on the uploaded documentation and officer's field visit report. Further details should be verified by the credit committee.

---

## 5. Conditions (Economic & Industry Environment)

- **News Sentiment Score**: {features.get('News_Sentiment_Score', 'N/A')} (scale: -1.0 to 1.0)
- **NCLT / Regulatory Findings**: {web_data.get('rbi_regulatory_actions', 'None found')}
- **Key Headlines**: {', '.join(web_data.get('news_headlines', ['No news data available']))}

---

## 🤖 ML Model Recommendation

| Parameter | Value |
|-----------|-------|
| Model | XGBoost Classifier |
| Decision | {status} |
| Approved Limit | ₹{limit} Cr |
| Interest Rate | {rate}% |
| Approval Probability | {approval_prob} |

---

## 📝 Final Recommendation

Based on the comprehensive analysis of all five credit parameters, the AI-assisted
credit appraisal system recommends **{status}** for **{company_name}**.

{'The company demonstrates strong fundamentals across all parameters. The recommended credit limit and interest rate are calibrated based on the ML model and verified financial data.' if ml_decision.get('Loan_Approved') == 1 else 'The company does not meet the minimum thresholds required for credit approval. Key risk factors have been identified in the analysis above.'}

---
*Report generated by CREDI-MITRA AI Agent System*
"""

    return cam_report.strip()


# ──────────────────────────────────────────────
# Build the ReAct Agent Graph
# ──────────────────────────────────────────────
ORCHESTRATOR_SYSTEM_PROMPT = """You are CREDI-MITRA, an expert AI Credit Analyst Agent. Your job is to perform
a comprehensive credit appraisal for a company by gathering data, analyzing it,
and producing a final Credit Appraisal Memorandum (CAM).

You have access to 5 specialized tools. You MUST use them in a logical order:

1. **extract_pdf_data** — FIRST, extract and parse the uploaded PDF documents.
2. **crawl_web_for_litigation** — THEN, search the web for litigation, NCLT filings, and news.
3. **extract_numerical_features** — NEXT, convert all gathered data into 6 numerical ML features.
4. **run_xgboost_scorer** — THEN, run the XGBoost model to get a credit decision.
5. **generate_cam_report** — FINALLY, produce the full CAM report.

IMPORTANT RULES:
- Always call tools one at a time and analyze each result before deciding the next step.
- If a tool asks for clarification (ambiguous data or missing values), relay the question
  EXACTLY to the user and wait for their response before continuing.
- After each tool result, briefly explain to the user what was found before proceeding.
- You MUST pass the outputs from earlier tools as inputs to later tools.
- Be thorough and transparent — show your reasoning at every step.
- When calling extract_numerical_features, pass the pdf_summary (output of extract_pdf_data),
  the web_research (output of crawl_web_for_litigation), and the company_name.
- When calling run_xgboost_scorer, pass the features_json (output of extract_numerical_features).
- When calling generate_cam_report, pass ALL previous outputs as arguments.

The user has already uploaded documents and provided company details. Begin analysis
when the user asks you to start.
"""

ALL_TOOLS = [
    extract_pdf_data,
    crawl_web_for_litigation,
    extract_numerical_features,
    run_xgboost_scorer,
    generate_cam_report,
]


def build_agent():
    """Build and return the LangGraph ReAct agent with Groq LLM and memory."""
    llm = ChatGroq(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0,
    )

    checkpointer = MemorySaver()

    agent = create_react_agent(
        model=llm,
        tools=ALL_TOOLS,
        prompt=ORCHESTRATOR_SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )

    return agent, checkpointer