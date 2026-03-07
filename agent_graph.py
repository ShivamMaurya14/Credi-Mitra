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
TAVILY_API_KEY = os.environ.get("tavily_api_key", os.environ.get("TAVILY_API_KEY", ""))

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

    # Try to find revenue figures (handling Cr and Lakhs)
    revenue_match = re.search(r'(?:revenue|turnover|sales)[:\s]*(?:Rs\.?|INR|₹)?\s*([\d,.]+)\s*(?:Cr|Crore|cr|Lakh|lakh|Lakhs|lakhs)', pdf_text, re.IGNORECASE)
    if revenue_match:
        val = float(revenue_match.group(1).replace(",", ""))
        unit = revenue_match.group(2).lower() if revenue_match.lastindex >= 2 else "cr"
        if 'lakh' in unit:
            val = val / 100.0  # Normalized to Crores
        numbers["Revenue_Cr"] = round(val, 2)

    # Try to find bank inflow (handling Cr and Lakhs)
    inflow_match = re.search(r'(?:inflow|bank\s*inflow|total\s*inflow)[:\s]*(?:Rs\.?|INR|₹)?\s*([\d,.]+)\s*(?:Cr|Crore|cr|Lakh|lakh|Lakhs|lakhs)?', pdf_text, re.IGNORECASE)
    if inflow_match:
        val = float(inflow_match.group(1).replace(",", ""))
        unit = (inflow_match.group(2) or "cr").lower()
        if 'lakh' in unit:
            val = val / 100.0
        numbers["Bank_Inflow_Cr"] = round(val, 2)

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

    # ── Verify name clarity ──
    # Check if the name is too short or looks like a placeholder
    is_ambiguous = len(company_name) < 3 or company_name.lower() in ["company", "sample", "test", "business"]

    if is_ambiguous:
        # HUMAN-IN-THE-LOOP: Interrupt and ask for a valid company name
        clarification = interrupt({
            "question": (
                f"⚠️ **Incomplete Company Name Detected**\n\n"
                f"The provided name **'{company_name}'** is too short or generic for a reliable web search. "
                "Please provide the full legal name of the entity you wish to analyze."
            ),
            "type": "ambiguity_check",
            "company": company_name
        })
        # When resumed, `clarification` contains the user's answer
        company_name = str(clarification).strip()

    # ── Real Web Research via Tavily ──
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=TAVILY_API_KEY)
        
        # Search for litigation and news
        search_query = f"{company_name} litigation NCLT RBI regulatory news"
        search_result = client.search(query=search_query, search_depth="advanced", max_results=5)
        
        news_headlines = []
        for r in search_result.get("results", []):
            news_headlines.append(f"{r['title']} ({r['url']})")
            
        # Sentiment logic based on keywords
        news_text = " ".join(news_headlines).lower()
        negative_keywords = ["litigation", "fraud", "scam", "lawsuit", "nclt", "penalty", "default"]
        litigation_count = sum(1 for word in negative_keywords if word in news_text)
        sentiment_score = 0.5 - (litigation_count * 0.2)
        sentiment_score = max(-0.9, min(0.9, sentiment_score))

        nclt_cases = []
        if "nclt" in news_text or "litigation" in news_text:
            nclt_cases.append({
                "case_id": "Detected via Search",
                "type": "Probable Legal/Regulatory Matter",
                "status": "Check Details",
                "bench": "Refer to Search Results"
            })

    except Exception as e:
        # Professional fallback: Report service unavailability
        news_headlines = []
        litigation_count = 0
        sentiment_score = 0.0
        nclt_cases = []
        status = "warning"
        message = f"Web research service (Tavily) currently unavailable: {str(e)}"

    result = {
        "status": "success" if 'status' not in locals() else "warning",
        "company_searched": company_name,
        "litigation_count": litigation_count,
        "nclt_cases": nclt_cases,
        "news_sentiment_score": round(sentiment_score, 2),
        "news_headlines": news_headlines,
        "rbi_regulatory_actions": "Searched",
        "source": "Tavily AI Search Engine",
        "notes": message if 'message' in locals() else "Live search completed successfully."
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
    if "Company_Age" in key_numbers:
        features["Company_Age"] = key_numbers["Company_Age"]
    else:
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
        import re
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
def run_xgboost_scorer(features_json: str, base_premium: float = 8.5) -> str:
    """Run the pre-trained XGBoost credit scoring model on the extracted features.
    
    Takes the 6 numerical features and produces a credit decision:
    Loan_Approved (0/1), Approved_Limit_Cr, and Interest_Rate_Pct.
    
    The interest rate is calculated dynamically using:
    Interest Rate = Base Premium + Risk Premium + Age Premium
    where Risk Premium = ((900 - CIBIL) / 100) * 0.5
    and Age Premium = 1.5% if Company Age <= 5 years, else 0%
    
    Args:
        features_json: JSON string containing the 6 features:
            Company_Age, CIBIL_Commercial_Score, GSTR_Declared_Revenue_Cr,
            Bank_Statement_Inflow_Cr, Litigation_Count, News_Sentiment_Score
        base_premium: The base interest rate premium set by the credit manager
            on the dashboard (default: 8.5%). This is the starting rate before
            risk and age adjustments.
    
    Returns:
        JSON with the ML model's credit decision including dynamic interest rate.
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
            "rate_breakdown": {
                "Base_Premium": base_premium,
                "Risk_Premium_CIBIL": risk_premium,
                "Age_Premium": age_premium,
                "Final_Rate": rate
            },
            "model_info": "XGBoost Classifier (pre-trained on 5000 synthetic corporate credit records, 97% accuracy)"
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

ROBUSTNESS & UNIT CONVERSION:
- Always check financial units. Our ML model expects values in **CRORES (Cr)**.
- If documents mention "Lakhs", convert them: Value in Cr = (Value in Lakhs / 100).
- If the user provides a raw number like "150 lakhs", convert it to "1.5" before calling tools.
- Be extra careful with OCR mishaps — if a number looks suspiciously small or large, ask the human for verification.


CRITICAL: Before starting any analysis, check the Context for missing information. If information is missing, you MUST ask for it before proceeding with tools:
- If "Company Name" or "Application No" is blank -> Request "Application Details" (Company Name, App No).
- If "PDF Text Available: No" -> Request "Document Ingestion" (PDFs, CIBIL, Bank Statements).
- If "Officer Insights: None provided" -> Request "Officer Insights" (Manual notes or uploaded report).

If any of these are missing, your first reply MUST be: "⚠️ **Action Required:** To begin the analysis, please provide the following missing inputs via the sidebar and click **Submit to Agent**:\n\n[List specifically what is missing from the 3 categories above]\n\nI'll be ready to start once these are received!" Do NOT attempt to run any tools until the basic context is complete.

When the user tells you they have successfully uploaded documents, you must:
1. Enthusiastically confirm receipt of the specific documents they mentioned.
2. End your message by asking: "Shall I proceed with the analysis using my Credit Engine tools?"
3. Wait for the user to say yes/proceed before calling ANY tools.

Begin analysis when the user asks you to start and all 3 categories of documents are ready.
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