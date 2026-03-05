import streamlit as st
import time
import pandas as pd
import numpy as np
import os
import io
import re
from fpdf import FPDF

# Set page configuration with a wide layout and a dark, professional theme
st.set_page_config(
    page_title="CREDI-MITRA - Next-Gen Corporate Credit Appraisal : Bridging the Intelligence Gap",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UI enhancements (Dark mode / Financial theme)
def local_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@800&display=swap');
        
        /* Sleek background and text color adjustments */
        .reportview-container {
            background-color: #0E1117;
            color: #FAFAFA;
        }
        /* Style for primary title */
        .main-title {
            font-family: 'Outfit', sans-serif;
            background: linear-gradient(90deg, #1E90FF 0%, #00FF7F 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            font-size: 5.5rem;
            font-weight: 800;
            margin-bottom: 0rem;
            line-height: 1.1;
            padding-top: 1rem;
            text-transform: uppercase;
        }
        .sub-title {
            color: #A0AEC0;
            text-align: center;
            font-size: 2.2rem;
            font-weight: 400;
            margin-bottom: 3rem;
            margin-top: 0.5rem;
        }
        /* Enhance metric containers */
        [data-testid="stMetricValue"] {
            font-size: 2.5rem;
            color: #00FF7F;
        }
        [data-testid="stMetricDelta"] {
            font-size: 1.2rem;
        }
        /* Buttons styling */
        .stButton>button {
            width: 100%;
            border-radius: 8px;
            font-weight: bold;
            transition: all 0.3s;
        }
        .stButton>button:hover {
            border-color: #1E90FF;
            color: #1E90FF;
        }
        /* Override primary button color to a professional blue */
        button[kind="primary"] {
            background-color: #1E90FF !important;
            color: white !important;
            border: none !important;
        }
        button[kind="primary"]:hover {
            background-color: #0073e6 !important;
            color: white !important;
        }
        /* Increase chat text size */
        [data-testid="stChatMessageContent"] {
            font-size: 1.25rem !important;
        }
        [data-testid="stChatMessageContent"] p {
            font-size: 1.25rem !important;
        }
    </style>
    """, unsafe_allow_html=True)

# Call CSS
local_css()

# --- INITIALIZE SESSION STATE ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'login'
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'cam_generated' not in st.session_state:
    st.session_state.cam_generated = False
if 'cam_content' not in st.session_state:
    st.session_state.cam_content = ""
if 'docs_verified' not in st.session_state:
    st.session_state.docs_verified = False
if 'company_name' not in st.session_state:
    st.session_state.company_name = ""
if 'app_no' not in st.session_state:
    st.session_state.app_no = ""
if 'analyzed_company' not in st.session_state:
    st.session_state.analyzed_company = ""
if 'analysis_step' not in st.session_state:
    st.session_state.analysis_step = 0

# --- PAGE ROUTING FUNCTIONS ---
def switch_page(page_name):
    st.session_state.current_page = page_name
    st.rerun()

# --- VIEW 1: THE LOGIN VIEW ---
def render_login():
    st.markdown("<h1 class='main-title'>CREDI-MITRA</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>{ Next-Gen Corporate Credit Appraisal }</p>", unsafe_allow_html=True)
    
    # Center the login form
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.container(border=True):
            st.subheader("Credit Manager Login")
            username = st.text_input("Username", value="admin", placeholder="Enter your corporate ID (hint: admin)")
            password = st.text_input("Password", value="password", type="password", placeholder="Enter your password (hint: password)")
            
            if st.button("Authenticate", type="primary"):
                if username == "admin" and password == "password":
                    st.session_state.logged_in = True
                    st.success("Authentication successful!!! ")
                    time.sleep(0.5)
                    switch_page('dashboard')
                else:
                    st.error("Invalid credentials. Please try again.")

# --- VIEW 2: THE ADMIN DASHBOARD ---
def render_dashboard():
    # Header area with logout button
    col_head, col_logout = st.columns([4, 1])
    with col_head:
        st.header("Dashboard")
    with col_logout:
        if st.button("Log Out"):
            st.session_state.logged_in = False
            switch_page('login')
    
    st.markdown("---")
    
    # Portfolio Metrics using st.columns and st.metric
    col1, col2, col3 = st.columns(3)
    with col1:
        with st.container(border=True):
            st.metric(label="Active Applications", value="12", delta="+2 this week")
    with col2:
        with st.container(border=True):
            st.metric(label="Accepted (MTD)", value="45", delta="15% vs Last Mo.")
    with col3:
        with st.container(border=True):
            st.metric(label="Rejected (MTD)", value="8", delta="-2% vs Last Mo.", delta_color="inverse")
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Call to action
    with st.container(border=True):
        st.subheader("⚙️ Application Analysis Center")
        st.caption("Initiate a multi-agent AI analysis for new appliction")
        st.markdown("<br>", unsafe_allow_html=True)
        col_btn_1, col_btn_2, col_btn_3 = st.columns([1, 2, 1])
        with col_btn_2:
            if st.button(" 🚀 New Application Analysis 🚀 ", type="primary", use_container_width=True):
                switch_page('analysis')
        st.markdown("<br>", unsafe_allow_html=True)

# --- VIEW 3: THE ANALYSIS & AGENT CHAT VIEW ---
def render_analysis():
    # Sidebar for inputs
    with st.sidebar:
        with st.expander("Application Details", expanded=True):
            company_name = st.text_input("Enter Company Name")
            app_no = st.text_input("Enter Application No.")
            app_date = st.date_input("Enter Application Date")
            
            # Dummy button for future integration
            if st.button("🔄 Auto-Fetch Documents "):
                st.info("Integration Pending : This will be integrated in later stages of development.")
        
        st.markdown("---")
        with st.expander("Document Ingestion", expanded=True):
            st.info("Upload required documents for AI processing.")
            
            # File uploaders
            app_form = st.file_uploader("Upload Application Form", type=["pdf", "docx"])
            gst = st.file_uploader("Upload GST Returns (GSTR-2A/3B)", type=["pdf", "csv", "xlsx"])
            bank = st.file_uploader("Upload Bank Statements", type=["pdf", "csv", "xlsx"])
            annual = st.file_uploader("Upload Annual Reports (PDF)", type=["pdf"])
            
        
        st.markdown("---")
        with st.expander("Officer Insights Documents", expanded=True):
            officer_report = st.file_uploader("  Upload Officer Insights Report", type=["pdf", "docx", "txt"])
            manual_entry = st.text_area("  Manual Entry ", 
                         placeholder="e.g., Factory visit notes, management interview summary .", 
                         height=150)
            st.caption("💡 *Note: You only need to provide ONE of the above (either upload the report OR type a manual entry) to proceed.*")
        
        st.markdown("---")
        
        # Verify Button & Logic
        if st.button("🚨 Submit to Agent", type="primary", use_container_width=True):
            st.session_state.docs_verified = True
            st.session_state.company_name = company_name
            st.session_state.app_no = app_no
            st.session_state.analyzed_company = company_name
            st.session_state.analysis_step = 1
            st.session_state.chat_history = []
            st.session_state.chat_history.append({"role": "assistant", "content": f"✅ Received Details of **{company_name}**! Starting multi-agent analysis. Type **next** to proceed or tell the agent something in analysis."})

            # --- Save uploaded files into a folder named CompanyName_AppNo ---
            c_name = company_name.strip().replace(' ', '_') or 'Company'
            a_no = app_no.strip().replace(' ', '_') or 'Application'
            folder_name = f"{c_name}_{a_no}"
            save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads", folder_name)
            os.makedirs(save_dir, exist_ok=True)

            # Map each uploaded file to its section name
            file_map = {
                "Application_Form": app_form,
                "GST_Returns": gst,
                "Bank_Statements": bank,
                "Annual_Reports": annual,
                "Officer_Insights_Report": officer_report,
            }

            saved_count = 0
            for section_name, uploaded_file in file_map.items():
                if uploaded_file is not None:
                    # Keep the original extension
                    ext = os.path.splitext(uploaded_file.name)[1]
                    save_path = os.path.join(save_dir, f"{section_name}{ext}")
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    saved_count += 1

            # Save manual officer entry as a text file if provided
            if manual_entry and manual_entry.strip():
                save_path = os.path.join(save_dir, "Officer_Manual_Notes.txt")
                with open(save_path, "w") as f:
                    f.write(manual_entry.strip())
                saved_count += 1

            if saved_count > 0:
                st.success(f"✅ {saved_count} document(s) saved to `uploads/{folder_name}/`")
            st.success("All details and documents submitted! You may talk to Agent for analysis.")
            
            # --- Original Strict Validation Logic (commented out for testing) ---
            # if not company_name or not app_no or not app_date:
            #     st.session_state.docs_verified = False
            #     st.error("Missing Application Details. Please enter Company Name, Application No, and Application Date first.")
            # elif app_form is None or gst is None or bank is None or annual is None:
            #     st.session_state.docs_verified = False
            #     st.error("Missing required documents. Please upload the Application Form, GST, Bank Statements, and Annual Reports.")
            # elif officer_report is None and not manual_entry.strip():
            #     st.session_state.docs_verified = False
            #     st.error("Missing Officer Insights. Please upload an Insights Report or enter notes manually.")
            # else:
            #     st.session_state.docs_verified = True
            #     st.success("All details and documents submitted! You may talk to Agent for analysis.")

        st.markdown("---")
        if st.button("← Back to Dashboard"):
            switch_page('dashboard')

    # Main column config
    st.markdown("<h1 class='main-title' style='font-size: 4rem;'>CREDI-MITRA-AI-AGENT</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title' style='margin-bottom: 1rem;'>{ Next-Gen Corporate Credit Appraisal }</p>", unsafe_allow_html=True)
    
    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # Re-render CAM and Chart if it's the final output
            if msg.get("is_cam", False) and st.session_state.cam_generated:
                render_cam_output()
    
    # Chat Input Configuration
    docs_ready = st.session_state.get('docs_verified', False)
    step = st.session_state.get('analysis_step', 0)
    
    # Define agent steps
    AGENT_STEPS = [
        {"agent": "🤖 Main Agent", "action": "Validating inputs & routing tasks...", "result": "✔️ Inputs validated. Routing tasks to specialized agents.", "type": "success"},
        {"agent": "📄 Extraction Agent", "action": "Parsing complex PDF tables using Vision models...", "result": "📌 Extracted: Balance Sheet data, P&L statements, and 4 director IDs from Annual Report.", "type": "info"},
        {"agent": "📊 Data Engineer Agent", "action": "Reconciling GSTR-3B vs Bank Statements...", "result": "📌 Reconciled: 98.4% match found between reported GST revenue and Bank inflows.", "type": "info"},
        {"agent": "🕵️ Research Agent", "action": "Crawling MCA filings and e-Courts for litigation risk...", "result": "📌 Research: 0 active litigations found. MCA status is 'Active'.", "type": "info"},
        {"agent": "🧠 Underwriting Agent", "action": "Calculating ML risk score and generating CAM...", "result": "✔️ ML Score Calculated: 840 (Low Risk). Generated CAM based on 5 Cs of Credit.", "type": "success"},
    ]
    
    if not docs_ready:
        with st.chat_message("assistant"):
            st.warning("⚠️ **Action Required:** Please upload all required documents and click 'Submit to Agent' in the sidebar to unlock the AI Analysis engine.")
        st.chat_input("🔒 Chat locked — Submit documents first", disabled=True)
    else:
        # Determine chat input placeholder based on current step
        if step <= len(AGENT_STEPS):
            placeholder = f"Type 'next' to proceed → Step {step}/{len(AGENT_STEPS)}"
        elif step == len(AGENT_STEPS) + 1:
            placeholder = "Type 'generate' to generate CAM Report"
        else:
            placeholder = "Analysis complete. Type company name to start a new analysis."
        
        if prompt := st.chat_input(placeholder):
            if step <= len(AGENT_STEPS):
                # Only proceed if user types "next"
                if prompt.strip().lower() != "next":
                    st.session_state.chat_history.append({"role": "user", "content": prompt})
                    st.session_state.chat_history.append({"role": "assistant", "content": "❌ Please type **next** to proceed to the next step."})
                    st.rerun()
                
                # Run current agent step
                agent = AGENT_STEPS[step - 1]
                st.session_state.chat_history.append({"role": "user", "content": f"next"})
                result_msg = f"**{agent['agent']}:** {agent['action']}\n\n{agent['result']}"
                st.session_state.chat_history.append({"role": "assistant", "content": result_msg})
                
                # If this was the last agent step, prompt for generate
                if step == len(AGENT_STEPS):
                    st.session_state.chat_history.append({"role": "assistant", "content": "✅ All agents have completed their analysis. Type **generate** to generate the CAM Report."})
                else:
                    st.session_state.chat_history.append({"role": "assistant", "content": f"✅ Step {step}/{len(AGENT_STEPS)} complete. Would you like to proceed to the next agent (type **next**) or tell the agent want something more in analysis/??"})
                
                st.session_state.analysis_step = step + 1
                st.rerun()
            
            elif step == len(AGENT_STEPS) + 1:
                # User must type "generate" to create CAM
                if prompt.strip().lower() != "generate":
                    st.session_state.chat_history.append({"role": "user", "content": prompt})
                    st.session_state.chat_history.append({"role": "assistant", "content": "❌ Please type **generate** to generate the CAM Report."})
                    st.rerun()
                
                st.session_state.chat_history.append({"role": "user", "content": "generate"})
                company = st.session_state.analyzed_company
                cam_markdown = f"""
### Comprehensive Credit Appraisal Memo (CAM)
**Entity Analyzed:** {company}
**Date:** {time.strftime('%Y-%m-%d')}
**Industry Sector:** Advanced Manufacturing & Technology
**Requested Facility:** $2.5M Term Loan, $500k Working Capital

---

#### 1. Business & Management Overview (Character)
**Assessment: Excellent**
- **Promoter Background:** The management team possesses over 15 years of industry experience. KYC and background checks on key directors returned clear reports. 
- **Litigation & Compliance:** An exhaustive AI-driven sweep of e-Courts and MCA databases revealed zero active litigations, no default histories, and a pristine track record regarding statutory filings (GST, Tax, ROC).

#### 2. Financial Performance & Cash Flows (Capacity)
**Assessment: Strong**
- **DSCR:** Calculated at a healthy 1.8x, indicating comfortable debt servicing ability.
- **Reconciliation:** Machine learning reconciliation between GSTR-3B filings and Bank Statements yields a 98.4% match, confirming robust, verifiable top-line revenue.
- **Liquidity:** Current Ratio stands at 1.45. Operating margins have remained stable at 18% despite recent sector-wide supply chain disruptions.

#### 3. Capital Structure (Capital)
**Assessment: Adequate to Strong**
- **Leverage:** Debt-to-Equity ratio is a conservative 1.2x. 
- **Skin in the Game:** The promoters have infused $300k in additional quasi-equity over the past 6 months to fund recent capital expenditure.

#### 4. Security & Collateral (Collateral)
**Assessment: Good**
- **Primary Security:** First charge on all current assets and newly acquired machinery.
- **Collateral Coverage:** Security value completely covers 120% of the proposed total exposure. The valuation reports were parsed and verified against internal real-estate benchmark models.

#### 5. Macro-Economic Context (Conditions)
**Assessment: Favorable**
- **Industry Outlook:** The sector is currently enjoying significant government tailwinds and subsidies. No immediate macro-economic or regulatory red flags appear on the horizon.

---

#### AI Final Recommendation:
Based on the synthesized multi-agent analysis, the entity demonstrates robust financial health, excellent promoter standing, and sufficient collateral coverage.

- **Proposed Limit:** $2.5 Million (Term Loan) + $500k (Working Capital)
- **Suggested Interest Rate:** 8.5% p.a. (Priced for Low Risk tier)
- **Risk Category:** Low-to-Moderate (Risk Score: 840/1000)
- **Covenants Recommended:** Maintain DSCR > 1.3x and provide quarterly automated bank statement ingestion for continuous monitoring.
                """
                st.session_state.cam_content = cam_markdown
                st.session_state.cam_generated = True
                st.session_state.chat_history.append({"role": "assistant", "content": "Analysis finalized. See the detailed report below.", "is_cam": True})
                st.session_state.analysis_step = step + 1
                st.rerun()
            
            else:
                # Analysis already complete — restart for new company
                st.session_state.analyzed_company = prompt
                st.session_state.chat_history = []
                st.session_state.cam_generated = False
                st.session_state.cam_content = ""
                st.session_state.analysis_step = 1
                st.session_state.chat_history.append({"role": "assistant", "content": f"✅ Received Details of **{prompt}**! Starting multi-agent analysis. Type **next** to proceed."})
                st.rerun()


def generate_cam_pdf(cam_text):
    """Convert CAM markdown content into a professionally formatted PDF."""
    # Strip emojis and other non-Latin-1 characters
    cam_text = re.sub(r'[^\x00-\xff]', '', cam_text)
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # -- Title --
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(30, 144, 255)  # Dodger Blue
    pdf.cell(0, 14, "CREDI-MITRA", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, "Next-Gen Credit Intelligence", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(4)
    # Divider line
    pdf.set_draw_color(30, 144, 255)
    pdf.set_line_width(0.6)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(6)

    # -- Parse and render each line --
    for raw_line in cam_text.strip().splitlines():
        line = raw_line.strip()
        if not line:
            pdf.ln(3)
            continue

        # Section heading (#### or ###)
        if line.startswith("####") or line.startswith("###"):
            heading = line.lstrip("#").strip()
            # Strip bold markdown markers
            heading = heading.replace("**", "")
            pdf.set_font("Helvetica", "B", 13)
            pdf.set_text_color(30, 60, 110)
            pdf.multi_cell(0, 8, heading)
            pdf.ln(2)

        # Horizontal rule
        elif line == "---":
            pdf.ln(2)
            pdf.set_draw_color(180, 180, 180)
            pdf.set_line_width(0.3)
            pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
            pdf.ln(4)

        # Bullet point
        elif line.startswith("- "):
            text = line[2:].replace("**", "")
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(50, 50, 50)
            bullet_x = pdf.l_margin + 4
            pdf.set_x(bullet_x)
            pdf.multi_cell(pdf.w - pdf.l_margin - pdf.r_margin - 4, 6, "-  " + text)
            pdf.ln(1)

        # Regular text (bold labels like **Key:** Value)
        else:
            text = line.replace("**", "")
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(50, 50, 50)
            pdf.multi_cell(0, 6, text)
            pdf.ln(1)

    # Output to bytes
    return bytes(pdf.output())


def render_cam_output():
    # Render the text
    st.markdown(st.session_state.cam_content)
    
    st.divider()
    
    # Explainability: Mock SHAP Visual
    st.subheader("Model Explainability (Mock SHAP Values)")
    st.caption("Feature impacts on the final risk score & interest rate determination:")
    
    # Mock data for explainability bar chart
    features = ['DSCR > 1.5', 'Clean Credit History', 'High Collateral Value', 'Sector Tailwinds', 'High Debt/Eq Ratio', 'Recent Modest Revenue Dip']
    impact = [0.8, 0.6, 0.4, 0.2, -0.3, -0.1]
    
    shap_df = pd.DataFrame({
        'Feature': features,
        'Impact on Favorable Score': impact
    }).set_index('Feature')
    
    # Wrap in columns to reduce width, and increase height parameter explicitly
    col_chart_left, col_chart_center, col_chart_right = st.columns([1, 3, 1])
    with col_chart_center:
        st.bar_chart(shap_df, color="#1E90FF", height=500)
    
    # Generate PDF and offer download
    pdf_bytes = generate_cam_pdf(st.session_state.cam_content)
    # Build filename from analyzed company name & application no
    c_name = st.session_state.get('analyzed_company', '').strip().replace(' ', '_') or 'Company'
    a_no = st.session_state.get('app_no', '').strip().replace(' ', '_') or 'Application'
    pdf_filename = f"{c_name}_{a_no}.pdf"
    st.download_button(
        label="📄 Download CAM Report (PDF)",
        data=pdf_bytes,
        file_name=pdf_filename,
        mime="application/pdf",
        type="primary"
    )


# --- MAIN APP CONTROLLER ---
def main():
    if not st.session_state.logged_in:
        # Force to login page if not authenticated
        st.session_state.current_page = 'login'
        render_login()
    else:
        # Route based on state
        if st.session_state.current_page == 'dashboard':
            render_dashboard()
        elif st.session_state.current_page == 'analysis':
            render_analysis()
        else:
            # Fallback
            render_dashboard()

if __name__ == "__main__":
    main()
