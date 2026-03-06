<div align="center">

# 🏦 CREDI-MITRA

### AI-Powered Corporate Credit Appraisal System

*Bridging the Intelligence Gap through Agentic Underwriting*

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://credibmitra-ai.streamlit.app/)
[![Python](https://img.shields.io/badge/Python_3.13+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-ReAct_Agent-7C3AED?style=for-the-badge)](https://langchain-ai.github.io/langgraph/)
[![XGBoost](https://img.shields.io/badge/XGBoost-97%25_Accuracy-006600?style=for-the-badge)](https://xgboost.readthedocs.io/)

</div>

<br/>

---

<br/>

## 📖 About

**CREDI-MITRA** is an AI-powered Credit Decisioning Engine that automates the end-to-end preparation of a Comprehensive Credit Appraisal Memo (CAM). It ingests multi-format data (**PDF, CSV, XLSX**), conducts real-time secondary research via the **Tavily AI Search Engine**, and leverages a pre-trained **XGBoost** model to recommend credit limits and risk-adjusted interest rates.

> **💡 Core Innovation:** Instead of a fixed pipeline, an LLM Agent dynamically decides what to do next, asks the analyst for help when data is ambiguous, and shows every intermediate step transparently in a chat environment.

<br/>

---

<br/>

## 🏗️ System Architecture

<div align="center">

```mermaid
flowchart TD
    %% Global Style
    classDef default font-size:16px,padding:15px;
    
    %% Input Layer
    IN[["📁 <b>INPUT DATA</b><br/>PDF • CSV • XLSX<br/>Officer Notes"]]
    
    %% Orchestration Layer
    A["🧠 <b>LLM ORCHESTRATOR</b><br/>Llama 3.1 • 8B • Groq<br/><i>ReAct Agentic Workspace</i>"]
    
    subgraph ToolSuite ["<b>INTELLIGENCE TOOLS</b>"]
        direction TB
        B(["📄 <b>Data Ingestion</b><br/>Extract Multi-format Text"])
        C(["🔍 <b>Tavily Search</b><br/>Litigation & News Sentiment"])
        D(["📊 <b>Feature Engine</b><br/>HITL Missing Data Collector"])
        E(["🤖 <b>ML Scorer</b><br/>XGBoost Classification"])
        G(["📋 <b>CAM Engine</b><br/>Report Synthesis"])
    end
    
    subgraph UI ["<b>HUMAN-IN-THE-LOOP</b>"]
        H{{"👤 <b>CREDIT ANALYST</b><br/>• Clarify Ambiguities<br/>• Input Missing Metrics"}}
    end

    F{{"📄 <b>CREDIT MEMO (PDF)</b><br/>The Final CAM Report"}}

    %% Flow
    IN ==> A
    A ==>|"Decides Plan"| ToolSuite
    
    C -.->|"Ambiguity Pause"| H
    D -.->|"Data Gap Pause"| H
    H -.->|"Resume Analysis"| A
    
    ToolSuite ===>|"Final Synthesis"| F

    %% Styling
    style A fill:#667eea,stroke:#fff,stroke-width:3px,color:#fff,font-weight:bold
    style ToolSuite fill:#1e1e35,stroke:#4a4a75,stroke-width:2px,color:#fff
    style UI fill:#2a1b2e,stroke:#ff6b6b,stroke-width:2px,stroke-dasharray: 5 5,color:#fff
    style H fill:#4a1e35,stroke:#ff6b6b,stroke-width:2px,color:#fff
    style F fill:#141423,stroke:#667eea,stroke-width:3px,color:#fff,font-weight:bold
    style IN fill:#34d399,stroke:#fff,stroke-width:2px,color:#000,font-weight:bold
```

</div>

## ✨ Key Features

*   **🧠 ReAct LLM Orchestrator:** Powered by Llama 3.1 via Groq. The agent dynamically decides which tools to call, rather than following a rigid pipeline.
*   **🤝 Human-in-the-Loop (HITL):** The system pauses execution to ask the human analyst for clarification when data is missing or ambiguous (e.g., multiple companies found, missing CIBIL score) before resuming the analysis.
*   **🌐 Multi-Source Data Ingestion:** Extracts data from uploaded files (**PDF, CSV, Excel**) and performs web-scale secondary research (NCLT filings, news sentiment).
*   **🤖 XGBoost Credit Scoring:** A custom ML model computes the probability of default, recommends an approved limit, and sets a dynamic interest rate based on risk premiums.
*   **📄 Automated CAM Generation:** Synthesizes all gathered data, financial metrics, and ML decisions into a final, downloadable PDF Credit Appraisal Memorandum.

## 🤖 Machine Learning Engine

CREDI-MITRA uses a pre-trained **XGBoost Classifier** to evaluate credit risk, trained on 5,000 synthetic corporate credit records.

*   **Accuracy:** 97%
*   **Features Analyzed:** Company Age, CIBIL Score, GST Revenue, Bank Inflow, Litigation Count, and News Sentiment.
*   **Decision Outputs:**
    1.  **Approval Decision** (Approved / Rejected)
    2.  **Recommended Limit (₹)** (Scaled by CIBIL and inflows)
    3.  **Dynamic Interest Rate (%)** (Base Premium + Risk Premium)

## 🛠️ Tech Stack

<div align="center">

| Layer | Technology | Role |
|:------|:-----------|:-----|
| 🧠 **LLM** | Llama 3.1 8B via [Groq](https://groq.com) | Central reasoning & orchestration |
| 🔗 **Agent Framework** | [LangGraph](https://langchain-ai.github.io/langgraph/) | ReAct agent, tool calling, interrupt/resume |
| 🖥️ **Frontend** | [Streamlit](https://streamlit.io/) | Chat interface, document upload, PDF export |
| 🤖 **ML Engine** | [XGBoost](https://xgboost.readthedocs.io/) | Credit risk classification (97% accuracy) |
| 📄 **Data Parsing** | [pypdf](https://pypdf.readthedocs.io/) + [Pandas](https://pandas.pydata.org/) | PDF extraction & Tabular data (CSV/XLSX) processing |
| 🐍 **Language** | Python 3.10+ | Core application |

</div>

<br/>

---

<br/>

## 🚀 Quick Start

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

## 🎮 Usage Guide

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

## 📁 Project Structure

```
CREDI-MITRA/
│
├── app.py                          # Streamlit UI — Chat, HITL, tool visibility
├── agent_graph.py                  # LangGraph ReAct Agent — LLM + 5 tools
├── requirements.txt                # Python dependencies
├── .env                            # API keys (GROQ_API_KEY, GROQ_MODEL)
│
├── model/                          # Machine Learning
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

## 🔮 Future Roadmap

- [ ] Multi-model support — Llama-3-70B / Gemini Pro / Ollama
- [ ] Structured OCR with **LlamaParse** or **Unstructured.io**
- [ ] Compliance audit trail — persistent agent trace logs
- [ ] Voice-driven appraisal via **LiveKit** integration
- [ ] Direct bank API integrations

<br/>

---

<br/>

<div align="center">

**Built with ❤️ by Shivam Maurya**

*CREDI-MITRA — Bridging the Intelligence Gap through Agentic Underwriting*

</div>
