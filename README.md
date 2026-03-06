<div align="center">

# 🏦 CREDI-MITRA

### AI-Powered Corporate Credit Appraisal System

*Bridging the Intelligence Gap through Agentic Underwriting*

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://credibmitra-ai.streamlit.app/)
[![Python](https://img.shields.io/badge/Python_3.13+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-ReAct_Agent-7C3AED?style=for-the-badge)](https://langchain-ai.github.io/langgraph/)
[![XGBoost](https://img.shields.io/badge/XGBoost-97%25_Accuracy-006600?style=for-the-badge)](https://xgboost.readthedocs.io/)

<br/>

<img src="https://img.shields.io/badge/Status-Active_Development-brightgreen?style=flat-square" />
<img src="https://img.shields.io/badge/License-MIT-blue?style=flat-square" />
<img src="https://img.shields.io/badge/Architecture-LLM_Orchestrator-purple?style=flat-square" />

</div>

<br/>

---

<br/>

## 📖 About

**CREDI-MITRA** is an enterprise-grade, AI-orchestrated credit underwriting platform. It replaces traditional rigid pipelines with a **ReAct LLM Orchestrator** — a Llama 3.1 model that dynamically reasons, plans, and calls specialized tools to perform end-to-end credit appraisal.

The system ingests financial documents, crawls the web for litigation & news, runs a pre-trained ML model for credit scoring, and produces a professional **Credit Appraisal Memorandum (CAM)** — all while keeping a human analyst in the loop for ambiguous decisions.

<br/>

> **💡 Core Innovation:** Instead of a fixed pipeline (A→B→C→D→E), an LLM Agent dynamically decides what to do next, asks the analyst for help when data is ambiguous, and shows every intermediate step transparently in a chat interface.

<br/>

---

<br/>

## 🏗️ System Architecture

<br/>

<div align="center">

```
                    ┌──────────────────────────────────┐
                    │        🧠 LLM ORCHESTRATOR       │
                    │      Llama 3.1 · 8B · Groq       │
                    │                                  │
                    │     Thinks → Plans → Executes    │
                    └───────────────┬──────────────────┘
                                    │
              ┌─────────────────────┼──────────────────────┐
              │                     │                      │
        TOOL CALLS            REASONING             HUMAN-IN-THE-LOOP
              │                     │                      │
   ┌──────────┴──────────┐         │         ┌────────────┴────────────┐
   │                     │         │         │                         │
┌──┴───┐  ┌──────────┐   │         │         │  ⏸  Ambiguous company?  │
│ 📄   │  │ 🔍       │   │         │         │  ⏸  Missing CIBIL?      │
│ PDF  │  │ Web      │   │         │         │  ⏸  Missing Revenue?    │
│ Data │  │ Research │   │         │         │                         │
└──┬───┘  └──────┬───┘   │         │         │  → Pauses execution     │
   │             │       │         │         │  → Asks analyst in chat │
┌──┴───┐  ┌──────┴───┐   │         │         │  → Resumes with answer  │
│ 📊   │  │ 🤖       │   │         │         └─────────────────────────┘
│ Feat │  │ XGBoost  │   │         │
│ Eng  │  │ Scorer   │   │         │
└──┬───┘  └──────┬───┘   │         │
   │             │       │         │
   └──────┬──────┘       │         │
    ┌─────┴─────┐        │         │
    │ 📋 CAM    │        │         │
    │ Report    │◄───────┘─────────┘
    └───────────┘
```

</div>

<br/>

### Old vs. New Architecture

| | **Before** (Fixed Pipeline) | **Now** (LLM Orchestrator) |
|:---|:---|:---|
| **Execution** | Hardcoded sequential steps | LLM dynamically decides order |
| **Error Handling** | Crash or silent skip | Agent reasons about missing data, asks user |
| **Interactivity** | None — runs to completion | Pauses mid-analysis for clarification |
| **Extensibility** | New step = rewrite pipeline | New tool = agent discovers it automatically |
| **Transparency** | Final output only | Every tool call visible in real-time chat |
| **Model** | Gemini (Google) | Llama 3.1 via Groq — open-source, blazing fast |

<br/>

---

<br/>

## 🔧 Tool Suite

The orchestrator has access to **5 specialized tools**, each responsible for a critical phase of credit analysis:

<br/>

<table>
<thead>
<tr>
<th align="center">#</th>
<th>Tool</th>
<th>Function</th>
<th align="center">HITL</th>
</tr>
</thead>
<tbody>
<tr>
<td align="center">1</td>
<td><code>extract_pdf_data</code></td>
<td>Parses uploaded PDFs using regex to extract CIBIL scores, GST revenue, bank inflows, and document structure</td>
<td align="center">—</td>
</tr>
<tr>
<td align="center">2</td>
<td><code>crawl_web_for_litigation</code></td>
<td>Searches NCLT filings, litigation records, RBI regulatory actions, and aggregates news sentiment</td>
<td align="center">⏸️</td>
</tr>
<tr>
<td align="center">3</td>
<td><code>extract_numerical_features</code></td>
<td>Merges PDF + Web data into 6 numerical features required by the ML model</td>
<td align="center">⏸️</td>
</tr>
<tr>
<td align="center">4</td>
<td><code>run_xgboost_scorer</code></td>
<td>Runs the pre-trained XGBoost classifier → Approved/Rejected, Limit (₹Cr), Interest Rate (%)</td>
<td align="center">—</td>
</tr>
<tr>
<td align="center">5</td>
<td><code>generate_cam_report</code></td>
<td>Produces the final Credit Appraisal Memorandum using the Five Cs of Credit framework</td>
<td align="center">—</td>
</tr>
</tbody>
</table>

<br/>

> **⏸️ HITL** = Human-in-the-Loop. These tools can pause execution via LangGraph's `interrupt()` to ask the analyst a question, then resume seamlessly with `Command(resume=...)`.

<br/>

---

<br/>

## 📊 Data Pipeline

How unstructured documents become a credit decision:

```
  UPLOADED PDFs                          WEB SEARCH
  ────────────                           ──────────
  ┌─────────────┐                    ┌──────────────────┐
  │ App Form    │                    │ NCLT Filings     │
  │ CIBIL Report│──► extract_pdf    │ News Headlines   │──► crawl_web
  │ GST Returns │      _data        │ RBI Actions      │     _for_litigation
  │ Bank Stmt   │        │          │ Court Records    │        │
  │ Annual Rpt  │        │          └──────────────────┘        │
  └─────────────┘        │                                      │
                         │          ┌───────────────────┐       │
                         └─────────►│ extract_numerical │◄──────┘
                                    │    _features      │
                                    └────────┬──────────┘
                                             │
                               ┌─────────────┴─────────────┐
                               │     6 ML FEATURES          │
                               │  ┌───────────────────────┐ │
                               │  │ Company_Age           │ │
                               │  │ CIBIL_Commercial_Score│ │
                               │  │ GSTR_Declared_Rev_Cr  │ │
                               │  │ Bank_Stmt_Inflow_Cr   │ │
                               │  │ Litigation_Count      │ │
                               │  │ News_Sentiment_Score  │ │
                               │  └───────────────────────┘ │
                               └─────────────┬──────────────┘
                                             │
                                    ┌────────┴────────┐
                                    │  🤖 XGBoost     │
                                    │  97% Accuracy   │
                                    ├─────────────────┤
                                    │ ✅ Approved / ❌│
                                    │ ₹ Limit (Cr)   │
                                    │ % Interest Rate │
                                    │ Probability     │
                                    └────────┬────────┘
                                             │
                                    ┌────────┴────────┐
                                    │  📋 CAM REPORT  │
                                    │  Five Cs Format │
                                    │  PDF Download   │
                                    └─────────────────┘
```

<br/>

---

<br/>

## 🤝 Human-in-the-Loop

CREDI-MITRA is designed for **collaboration**, not automation-only. The agent knows when to ask for help:

<br/>

### Scenario 1 — Ambiguous Company Name

When web research finds multiple matching entities:

```
┌──────────────────────────────────────────────────────────────┐
│  ⏸️  Agent Message                                           │
│                                                              │
│  ⚠️ Ambiguous Company Name Detected                         │
│                                                              │
│  While searching for "Tata", I found multiple entities:      │
│                                                              │
│  1. Tata Technologies Ltd. — IT Services (Mumbai)            │
│  2. Tata Industrial Solutions Pvt. Ltd. — Manufacturing      │
│  3. Tata Finance & Leasing Co. — NBFC (Delhi)                │
│                                                              │
│  Which company are you analyzing?                            │
└──────────────────────────────────────────────────────────────┘
```

### Scenario 2 — Missing Financial Data

When critical numbers can't be extracted from uploaded documents:

```
┌──────────────────────────────────────────────────────────────┐
│  ⏸️  Agent Message                                           │
│                                                              │
│  📊 Missing Financial Data for XYZ Corp                      │
│                                                              │
│  I could not find the following in the uploaded documents:    │
│                                                              │
│  • CIBIL Commercial Score (typically 300–900)                 │
│  • Bank Statement Total Inflow in Crores                     │
│                                                              │
│  Please provide: CIBIL: 750, Inflow: 115.0                   │
└──────────────────────────────────────────────────────────────┘
```

<br/>

### Technical Implementation

```
interrupt()  →  Graph pauses  →  Streamlit shows question  →  User replies
     →  Command(resume=answer)  →  Graph resumes from exact pause point
```

State is persisted via **LangGraph MemorySaver** — no page refresh, no data loss.

<br/>

---

<br/>

## � ML Model Performance

<br/>

<div align="center">

| Metric | Value |
|:-------|:-----:|
| **Algorithm** | XGBoost Classifier |
| **Accuracy** | **97%** |
| **Training Data** | 5,000 synthetic corporate credit records |
| **Features** | 6 numerical features |
| **Explainability** | SHAP-based feature importance |
| **Outputs** | Approval Decision, Credit Limit, Interest Rate, Probability |

</div>

<br/>

### Credit Decision Rules

The model learns patterns from these underwriting business rules:

| Rule | Condition | Result |
|:-----|:----------|:-------|
| **CIBIL Cutoff** | Score < 600 | ❌ Rejected |
| **Litigation Risk** | Count ≥ 3 or Sentiment < -0.5 | ❌ Rejected |
| **Data Paradox** | GST vs Bank variance > 25% | ❌ Rejected (circular trading flag) |
| **Limit Calculation** | Bank Inflow × (15-25%) × (CIBIL/900) | ₹ Cr limit |
| **Interest Rate** | Base 8.5% + risk premium + age premium | Dynamic % |

<br/>

---

<br/>

## 🛠️ Tech Stack

<br/>

<div align="center">

| Layer | Technology | Role |
|:------|:-----------|:-----|
| 🧠 **LLM** | Llama 3.1 8B via [Groq](https://groq.com) | Central reasoning & orchestration |
| 🔗 **Agent Framework** | [LangGraph](https://langchain-ai.github.io/langgraph/) | ReAct agent, tool calling, interrupt/resume |
| 🖥️ **Frontend** | [Streamlit](https://streamlit.io/) | Chat interface, document upload, PDF export |
| 🤖 **ML Engine** | [XGBoost](https://xgboost.readthedocs.io/) | Credit risk classification (97% accuracy) |
| 📄 **PDF Processing** | [pypdf](https://pypdf.readthedocs.io/) + [fpdf2](https://py-pdf.github.io/fpdf2/) | Document parsing & report generation |
| 🐍 **Language** | Python 3.10+ | Core application |

</div>

<br/>

---

<br/>

## � Quick Start

### Prerequisites

- Python 3.10+
- [Groq API Key](https://console.groq.com) (free tier available)

### Setup

```bash
# 1. Clone
git clone https://github.com/ShivamMaurya14/CREDI-MITRA.git
cd CREDI-MITRA

# 2. Install
pip install -r requirements.txt

# 3. Configure — create .env file
echo 'GROQ_API_KEY=gsk_your_key_here' > .env
echo 'GROQ_MODEL=llama-3.1-8b-instant' >> .env

# 4. Launch
streamlit run app.py
```

<br/>

---

<br/>

## 🎮 Usage Guide

<br/>

| Step | Action | Details |
|:----:|:-------|:-------|
| **1** | 🔐 Login | Authenticate with `admin` / `password` |
| **2** | 📊 Dashboard | View portfolio metrics, click "New Application Analysis" |
| **3** | 📁 Upload | Upload required documents in the sidebar |
| **4** | 🧑‍💼 Officer Notes | Add field visit notes or upload officer report |
| **5** | 🚨 Submit | Click "Submit to Agent" to initialize |
| **6** | 💬 Chat | Type `"start analysis"` to trigger the LLM orchestrator |
| **7** | ⏸️ Respond | Answer any clarification questions from the agent |
| **8** | 📋 Review | Inspect each tool output in expandable panels |
| **9** | 📄 Download | Export the final CAM report as PDF |

<br/>

---

<br/>

## 📁 Project Structure

```
CREDI-MITRA/
│
├── app.py                          # Streamlit UI — Chat, HITL, tool visibility
├── agent_graph.py                  # LangGraph ReAct Agent — LLM + 5 tools
├── requirements.txt                # Python dependencies
├── .env                            # API keys (GROQ_API_KEY, GROQ_MODEL)
│
├── ml/                             # Machine Learning
│   ├── model.json                  # Pre-trained XGBoost (97% accuracy)
│   ├── data.csv                    # 5,000 synthetic records
│   ├── data_maker.py               # Data generation script
│   ├── credit_prediction.ipynb     # Model training notebook
│   ├── Reason_for_rejection.py     # Business rejection rules
│   └── shap_summary_global.png     # Feature importance visualization
│
├── uploads/                        # Saved documents (CompanyName_AppNo/)
└── README.md
```

<br/>

---

<br/>

## 🔮 Roadmap/Future Scope

- [ ] Live web search via Tavily API integration
- [ ] Structured PDF parsing with LlamaParse
- [ ] Enhanced ML — dynamic interest rates & rejection reasons
- [ ] Multi-model support — Groq / Ollama (local) / Gemini
- [ ] Compliance audit trail — persist agent logs
- [ ] Voice-driven appraisal via LiveKit

<br/>

---

<br/>

<div align="center">

**Built with ❤️ by Shivam Maurya**

*CREDI-MITRA — Bridging the Intelligence Gap through Agentic Underwriting*

</div>
