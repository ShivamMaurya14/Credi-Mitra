"""
Streamlit UI Components for RAG Document Management
Handles document upload, display, search, and updates
"""

import streamlit as st
import json
import os
from datetime import datetime
import pandas as pd

from rag_manager import get_document_manager

# ──────────────────────────────────────────────
# UI Theme Configuration
# ──────────────────────────────────────────────
def apply_rag_css():
    """Apply custom CSS for RAG components"""
    st.markdown("""
    <style>
    /* RAG Document Container */
    .rag-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        color: white;
    }
    
    .doc-card {
        background: white;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .metric-badge {
        display: inline-block;
        background: #667eea;
        color: white;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.85em;
        margin: 5px 5px 5px 0;
    }
    
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 12px;
        border-radius: 5px;
        margin: 10px 0;
    }
    
    .error-box {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 12px;
        border-radius: 5px;
        margin: 10px 0;
    }
    
    .search-result {
        background: #f8f9fa;
        padding: 12px;
        border-radius: 5px;
        margin: 10px 0;
        border-left: 3px solid #764ba2;
    }
    </style>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Document Upload Component
# ──────────────────────────────────────────────
def render_document_upload():
    """Render document upload interface"""
    st.markdown("### 📤 Upload Documents to RAG Database")
    
    with st.container():
        col1, col2, col3 = st.columns([2, 2, 1.5])
        
        with col1:
            company_name = st.text_input(
                "Company Name",
                value=st.session_state.get("company_name", ""),
                placeholder="e.g., Acme Corp Ltd",
                key="upload_company_name"
            )
        
        with col2:
            doc_type = st.selectbox(
                "Document Type",
                [
                    "Annual Report",
                    "CIBIL Report",
                    "GST Return",
                    "Bank Statement",
                    "Balance Sheet",
                    "Audit Report",
                    "Income Statement",
                    "Cash Flow Statement",
                    "Board Resolution",
                    "Financial Highlights",
                    "Other"
                ],
                key="upload_doc_type"
            )
        
        with col3:
            extract_data = st.checkbox(
                "Extract Metrics",
                value=True,
                help="Automatically extract financial metrics"
            )
    
    # File upload
    st.markdown("**Upload PDF Files:**")
    uploaded_files = st.file_uploader(
        "Choose PDF files to upload to RAG database",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload credit-related documents (Annual Reports, CIBIL, Bank Statements, etc.)"
    )
    
    if uploaded_files and company_name:
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("🚀 Upload & Index", use_container_width=True):
                doc_manager = get_document_manager()
                
                progress_bar = st.progress(0)
                status_container = st.container()
                
                results_summary = {
                    "successful": 0,
                    "failed": 0,
                    "results": []
                }
                
                for idx, uploaded_file in enumerate(uploaded_files):
                    with status_container:
                        st.info(f"Processing: {uploaded_file.name} ({idx+1}/{len(uploaded_files)})")
                    
                    result = doc_manager.upload_pdf(
                        pdf_file=uploaded_file,
                        company_name=company_name,
                        document_type=doc_type,
                        extract_structured_data=extract_data
                    )
                    
                    if result["status"] == "success":
                        results_summary["successful"] += 1
                        results_summary["results"].append({
                            "file": result["file_name"],
                            "status": "✅ Success",
                            "doc_id": result["doc_id"],
                            "metrics": result.get("structured_data")
                        })
                    else:
                        results_summary["failed"] += 1
                        results_summary["results"].append({
                            "file": uploaded_file.name,
                            "status": "❌ Failed",
                            "error": result.get("error")
                        })
                    
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                # Show summary
                st.markdown("---")
                st.markdown("### Upload Summary")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Successful", results_summary["successful"])
                with col2:
                    st.metric("Failed", results_summary["failed"])
                with col3:
                    st.metric("Total", len(uploaded_files))
                
                # Show detailed results
                for result in results_summary["results"]:
                    if "doc_id" in result:
                        st.markdown(f"<div class='success-box'>{result['status']} — {result['file']}</div>", unsafe_allow_html=True)
                        if result["metrics"]:
                            with st.expander("📊 Extracted Metrics"):
                                for key, value in result["metrics"].items():
                                    st.write(f"• **{key}:** {value}")
                    else:
                        st.markdown(f"<div class='error-box'>{result['status']} — {result['file']}<br/>{result.get('error')}</div>", unsafe_allow_html=True)
                
                st.session_state["company_name"] = company_name
                st.success("✅ Upload complete! Documents are now searchable in your analysis.")
        
        with col2:
            if st.button("📋 View Company Docs", use_container_width=True):
                st.session_state["show_company_docs"] = True
    
    elif not company_name:
        st.warning("⚠️ Please enter a company name to upload documents")


# ──────────────────────────────────────────────
# Document Management Component
# ──────────────────────────────────────────────
def render_document_management():
    """Render document management interface"""
    st.markdown("### 📚 Document Management")
    
    doc_manager = get_document_manager()
    company_name = st.session_state.get("company_name", "")
    
    if not company_name:
        st.info("Enter a company name to view and manage its documents")
        return
    
    # Get documents
    documents = doc_manager.list_company_documents(company_name)
    
    if not documents:
        st.info(f"No documents found for '{company_name}'. Upload documents above to get started.")
        return
    
    st.info(f"📂 Found {len(documents)} documents for {company_name}")
    
    # Document list
    for idx, doc in enumerate(documents, 1):
        with st.expander(f"📄 {doc['file_name']} ({doc['type']})"):
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.write(f"**Type:** {doc['type']}")
                st.write(f"**ID:** `{doc['id']}`")
                st.write(f"**Upload Date:** {doc['upload_date']}")
                st.write(f"**Size:** {doc['content_length']:,} chars")
                st.write(f"**Chunks:** {doc['chunks']}")
            
            with col2:
                if st.button("🔍 View", key=f"view_{doc['id']}", use_container_width=True):
                    st.session_state[f"show_doc_detail_{doc['id']}"] = True
            
            with col3:
                if st.button("🗑️ Delete", key=f"delete_{doc['id']}", use_container_width=True):
                    if doc_manager.delete_document(doc['id']):
                        st.success("✅ Document deleted")
                        st.rerun()
                    else:
                        st.error("❌ Failed to delete")
            
            # Show detail if requested
            if st.session_state.get(f"show_doc_detail_{doc['id']}", False):
                st.markdown("---")
                summary = doc_manager.get_document_summary(doc['id'])
                if summary and summary.get("metadata"):
                    st.json(summary["metadata"])


# ──────────────────────────────────────────────
# Document Search Component
# ──────────────────────────────────────────────
def render_document_search():
    """Render document search interface"""
    st.markdown("### 🔎 Search Documents")
    
    doc_manager = get_document_manager()
    company_name = st.session_state.get("company_name", "")
    
    # Search parameters
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input(
            "Search query",
            placeholder="e.g., revenue, bank statements, litigation, assets...",
            key="rag_search_query"
        )
    
    with col2:
        top_k = st.slider("Top results", 1, 10, 5)
    
    # Document type filter
    doc_types_available = [
        "Annual Report", "CIBIL Report", "GST Return", "Bank Statement",
        "Balance Sheet", "Audit Report", "Income Statement"
    ]
    selected_types = st.multiselect(
        "Filter by document type",
        doc_types_available,
        key="rag_filter_types"
    )
    
    # Search button
    if st.button("🔍 Search", use_container_width=True):
        if not search_query.strip():
            st.warning("Please enter a search query")
        else:
            with st.spinner("Searching documents..."):
                results = doc_manager.query_documents(
                    query=search_query,
                    company_name=company_name if company_name else None,
                    document_types=selected_types if selected_types else None,
                    top_k=top_k
                )
            
            if not results:
                st.info("No relevant documents found for this query")
            else:
                st.success(f"Found {len(results)} relevant sections")
                
                for idx, result in enumerate(results, 1):
                    with st.expander(
                        f"📑 Result {idx} — {result['metadata'].get('file_name', 'Unknown')} "
                        f"({result['metadata'].get('document_type', 'Unknown')})"
                    ):
                        # Content
                        st.markdown("**Relevant Content:**")
                        st.write(result['content'])
                        
                        # Metadata
                        with st.expander("📋 Metadata"):
                            for key, value in result['metadata'].items():
                                if key not in ['timestamp', 'total_chunks']:
                                    st.write(f"**{key}:** {value}")


# ──────────────────────────────────────────────
# Metrics Extraction Component
# ──────────────────────────────────────────────
def render_metrics_extraction():
    """Render automated metrics extraction from documents"""
    st.markdown("### 📊 Extract Financial Metrics")
    
    doc_manager = get_document_manager()
    company_name = st.session_state.get("company_name", "")
    
    if not company_name:
        st.info("Enter a company name to extract metrics from its documents")
        return
    
    documents = doc_manager.list_company_documents(company_name)
    
    if not documents:
        st.warning("No documents found for this company")
        return
    
    if st.button("🤖 Extract Metrics from All Documents", use_container_width=True):
        with st.spinner("Extracting metrics from all documents..."):
            metrics_queries = {
                "💰 Revenue": "company revenue turnover sales annual",
                "📊 CIBIL Score": "CIBIL credit score rating",
                "🏦 Bank Inflow": "bank inflow deposits transactions cash inflow",
                "⚖️ Litigation": "lawsuit litigation nclt writ petition dispute case",
                "📅 Company Age": "established founded incorporated year",
                "🧾 GST Revenue": "GST revenue registration GST returns",
                "📈 Assets": "total assets balance sheet assets",
                "💳 Liabilities": "liabilities debts obligations"
            }
            
            all_metrics = {}
            
            for metric_name, query_text in metrics_queries.items():
                results = doc_manager.query_documents(
                    query=query_text,
                    company_name=company_name,
                    top_k=3
                )
                
                all_metrics[metric_name] = {
                    "found": len(results) > 0,
                    "count": len(results),
                    "sources": [r['metadata'].get('file_name') for r in results[:2]] if results else []
                }
            
            # Display results
            st.markdown("---")
            st.markdown("#### Extraction Results")
            
            found_metrics = {k: v for k, v in all_metrics.items() if v['found']}
            missing_metrics = {k: v for k, v in all_metrics.items() if not v['found']}
            
            if found_metrics:
                st.success(f"✅ Found {len(found_metrics)} metrics in documents")
                for metric_name, info in found_metrics.items():
                    st.write(
                        f"{metric_name} — **{info['count']} references** | Sources: {', '.join(info['sources'])}"
                    )
            
            if missing_metrics:
                st.warning(f"⚠️ Could not find {len(missing_metrics)} metrics:")
                for metric_name in missing_metrics.keys():
                    st.write(f"• {metric_name}")


# ──────────────────────────────────────────────
# Data Update Component
# ──────────────────────────────────────────────
def render_data_updates():
    """Render interface for updating document-based data"""
    st.markdown("### ✏️ Update Document Data")
    
    doc_manager = get_document_manager()
    company_name = st.session_state.get("company_name", "")
    
    documents = doc_manager.list_company_documents(company_name)
    
    if not documents:
        st.warning("No documents available")
        return
    
    # Select document
    doc_options = {f"{d['file_name']} ({d['type']})": d['id'] for d in documents}
    selected_doc_display = st.selectbox("Select document to update", doc_options.keys())
    selected_doc_id = doc_options[selected_doc_display]
    
    # Get current data
    current_doc = doc_manager.get_document_summary(selected_doc_id)
    
    st.markdown("**Current Metadata:**")
    if current_doc and current_doc.get('metadata'):
        col1, col2 = st.columns(2)
        with col1:
            for key in list(current_doc['metadata'].keys())[:5]:
                st.write(f"• {key}: {current_doc['metadata'][key]}")
    
    # Update form
    st.markdown("---")
    st.markdown("**Add/Update Data:**")
    
    update_key = st.text_input("Field name", placeholder="e.g., verified_revenue")
    update_value = st.text_input("Value", placeholder="e.g., 500 Cr")
    
    if st.button("💾 Save Update", use_container_width=True):
        if update_key and update_value:
            success = doc_manager.update_document_data(
                selected_doc_id,
                {update_key: update_value}
            )
            
            if success:
                st.success(f"✅ Updated {update_key} = {update_value}")
            else:
                st.error("❌ Failed to update")
        else:
            st.warning("Please fill in both field name and value")


# ──────────────────────────────────────────────
# Main RAG Dashboard
# ──────────────────────────────────────────────
def render_rag_dashboard():
    """Render complete RAG management dashboard"""
    apply_rag_css()
    
    st.markdown("""
    <div class='rag-container'>
    <h1>🚀 RAG Document Intelligence</h1>
    <p>Upload, search, and analyze company documents using AI-powered retrieval</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📤 Upload",
        "📚 Manage",
        "🔎 Search",
        "📊 Metrics",
        "✏️ Update"
    ])
    
    with tab1:
        render_document_upload()
    
    with tab2:
        render_document_management()
    
    with tab3:
        render_document_search()
    
    with tab4:
        render_metrics_extraction()
    
    with tab5:
        render_data_updates()


if __name__ == "__main__":
    render_rag_dashboard()
