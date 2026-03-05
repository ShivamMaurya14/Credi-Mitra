# 🏦 CREDI-MITRA AI-AGENT
### *Next-Gen Corporate Credit Appraisal : Bridging the Intelligence Gap*

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_svg)](https://credibmitra-ai.streamlit.app/)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)

---

## 📖 Overview

**CREDI-MITRA** is an enterprise-grade AI-orchestrated platform designed to revolutionize the corporate credit underwriting lifecycle. By deploying a sophisticated **Multi-Agent Simulation**, it automates the transition from raw document ingestion to deep risk synthesis, empowering credit managers with high-fidelity, data-driven insights.

> **The Vision:** Eliminate the "Intelligence Gap" in credit appraisal by utilizing specialized AI agents that simulate an entire credit department in seconds.

---

## � The Multi-Agent Ecosystem

The core of Credi-Mitra is its step-controlled agent workflow. Each agent is responsible for a critical pillar of credit analysis:

| Agent | Responsibility | Key Output |
| :--- | :--- | :--- |
| **🤖 Main Agent** | Orchestration & Routing | Validated inputs, Task delegation |
| **📄 Extraction Agent** | Vision-based Data Parsing | Structured P&L & Balance Sheet data |
| **📊 Data Engineer** | Financial Reconciliation | GST vs. Bank Statement delta report |
| **🕵️ Research Agent** | Risk Sweep & Compliance | MCA filings & e-Court risk profiling |
| **🧠 Underwriting Agent**| Credit Decision Synthesis | ML-based scoring & 5-Cs logic |

---

## ✨ Key Capabilities

### �️ Intelligent Document Validation
Strict verification logic ensures that an application only proceeds once the **Application Form**, **GST Returns**, **Bank Statements**, and **Annual Reports** are successfully uploaded and parsed.

### ✍️ Human-Centric Interactive Analysis
Unlike "black-box" systems, Credi-Mitra maintains a **Human-in-the-Loop** approach:
- **Interactive Probes:** Users can guide agents during any analysis step.
- **Workflow Control:** Simple keyword triggers (`next`, `generate`) ensure the human remains the final arbiter.

### � Professional CAM Generation
Synthesizes a full **Credit Appraisal Memo (CAM)** covering:
- **Character**: Promoter background & legal checks.
- **Capacity**: DSCR and liquidity reconciliations.
- **Capital**: Leverage & quasi-equity analysis.
- **Collateral**: Valuation & charge verification.
- **Conditions**: Macro-economic & industry outlook.

---

## 🛠️ Infrastructure & Stack

*   **Frontend/App Layer:** [Streamlit](https://streamlit.io/) — Professional, responsive UI.
*   **Engine:** Python 3.x — Asynchronous agent logic simulation.
*   **Reporting:** [fpdf2](https://py-pdf.github.io/fpdf2/) — Dynamic PDF generation with dynamic naming headers.
*   **ML Integration:** SHAP values for model transparency and feature importance visualization.

---

## 🚦 Getting Started

### Prerequisites
- Python 3.10 or higher
- `pip` package manager

### Installation
1.  **Clone the Repository**
    ```bash
    git clone https://github.com/your-repo/credi-mitra.git
    cd credi-mitra
    ```
2.  **Environment Setup**
    ```bash
    pip install streamlit fpdf2 pandas matplotlib
    ```
3.  **Run Application**
    ```bash
    streamlit run app.py
    ```

---

## 🎮 Operational Flow

1.  **Log In**: Standard security gate (`admin`/`password`).
2.  **Configure Application**: Set Company Name, App ID, and upload the 4 required document sets.
3.  **Initiate Agent Flow**: Click the **🚨 Submit to Agent** button.
4.  **Guided Analysis**:
    *   Observe agent progress in real-time.
    *   Type `next` to advance to specialized checks.
    *   Provide custom prompts to the agent if you need deeper scrutiny.
5.  **Finalize**: Type `generate` to finalize the CAM and download the dynamic PDF report.

---

## � Architecture

```text
├── app.py                # Main Application & Agent Orchestrator
├── ml/                   # Machine Learning models (JSON) & SHAP visualizations
│   ├── model.json        # Pre-trained XGBoost/LightGBM logic
│   └── data_maker.py     # Synthetic data generation logic
├── uploads/              # Structured storage: /CompanyName_AppNo/
└── README.md             # Project Documentation
```

---

---
**Credi-Mitra: Bridging the Intelligence Gap through Agentic Underwriting.**
