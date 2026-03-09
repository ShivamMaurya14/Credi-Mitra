"""
RAG (Retrieval-Augmented Generation) Module for Credi-Mitra
Unified module combining document management, embedding, retrieval, and LLM integration.

Components:
- ChromaDBManager: Vector DB operations
- PDFProcessor: PDF text extraction  
- DocumentManager: High-level document API
- RAG Tools: LangGraph tool definitions for agent integration
- RAG UI: Streamlit dashboard components
"""

import os
import json
import uuid
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import hashlib

import chromadb
import pdfplumber
import pypdf
import streamlit as st
from langchain_core.tools import tool

# ═══════════════════════════════════════════════
# PART 1: CHROMA DB MANAGER
# ═══════════════════════════════════════════════

class ChromaDBManager:
    """Vector database management with persistent storage"""
    
    def __init__(self, db_path="./chroma_db"):
        self.db_path = db_path
        os.makedirs(db_path, exist_ok=True)
        self.client = chromadb.PersistentClient(path=db_path)
        self._init_collections()
    
    def _init_collections(self):
        """Initialize document collections"""
        self.documents = self.client.get_or_create_collection(
            name="credit_documents",
            metadata={"description": "Document chunks for search"},
            distance_function="cosine"
        )
        self.metadata = self.client.get_or_create_collection(
            name="document_metadata",
            metadata={"description": "Document metadata"},
            distance_function="cosine"
        )
    
    def add_document(self, doc_id: str, company: str, doc_type: str, 
                     content: str, file_name: str, metadata: Dict = None) -> bool:
        """Add document with chunks and metadata"""
        try:
            metadata = metadata or {}
            chunks = self._chunk_text(content, chunk_size=1000, overlap=100)
            
            for idx, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_chunk_{idx}"
                self.documents.add(
                    ids=[chunk_id],
                    documents=[chunk],
                    metadatas=[{
                        "doc_id": doc_id, "company": company, "type": doc_type,
                        "file": file_name, "chunk": idx, "timestamp": datetime.now().isoformat()
                    }]
                )
            
            # Store full metadata
            self.metadata.add(
                ids=[doc_id],
                documents=[content[:500]],
                metadatas=[{
                    "company": company, "type": doc_type, "file": file_name,
                    "created": datetime.now().isoformat(),
                    **metadata
                }]
            )
            return True
        except Exception as e:
            print(f"Error adding document: {e}")
            return False
    
    def search_documents(self, query: str, company: str = None, 
                        doc_type: str = None, top_k: int = 5) -> List[Dict]:
        """Semantic search across documents"""
        try:
            where_filter = {}
            if company:
                where_filter["company"] = company
            if doc_type:
                where_filter["type"] = doc_type
            
            results = self.documents.query(
                query_texts=[query],
                n_results=top_k,
                where=where_filter if where_filter else None
            )
            
            return [{
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "similarity": 1 - results["distances"][0][i] if results["distances"] else 0
            } for i in range(len(results["ids"][0]))]
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def list_documents(self, company: str = None) -> List[Dict]:
        """List all documents, optionally filtered by company"""
        try:
            all_docs = self.metadata.get(
                where={"company": company} if company else None
            )
            return [{
                "id": all_docs["ids"][i],
                "company": all_docs["metadatas"][i].get("company"),
                "type": all_docs["metadatas"][i].get("type"),
                "file": all_docs["metadatas"][i].get("file")
            } for i in range(len(all_docs["ids"]))]
        except Exception as e:
            print(f"List error: {e}")
            return []
    
    def update_document_metadata(self, doc_id: str, updates: Dict) -> bool:
        """Update document metadata"""
        try:
            current = self.metadata.get(ids=[doc_id])
            if not current["ids"]:
                return False
            
            metadata = current["metadatas"][0]
            metadata.update(updates)
            self.metadata.update(ids=[doc_id], metadatas=[metadata])
            return True
        except Exception as e:
            print(f"Update error: {e}")
            return False
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete document and all chunks"""
        try:
            chunks = self.documents.get(where={"doc_id": doc_id})
            if chunks["ids"]:
                self.documents.delete(ids=chunks["ids"])
            self.metadata.delete(ids=[doc_id])
            return True
        except Exception as e:
            print(f"Delete error: {e}")
            return False
    
    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """Split text into chunks with overlap"""
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            start = end - overlap
        return chunks


# ═══════════════════════════════════════════════
# PART 2: PDF PROCESSOR
# ═══════════════════════════════════════════════

class PDFProcessor:
    """Extract and process PDF content"""
    
    @staticmethod
    def extract_text(pdf_path: str) -> str:
        """Extract text from PDF with fallback methods"""
        text = ""
        
        # Try pdfplumber first
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""
            return text
        except:
            pass
        
        # Fallback to pypdf
        try:
            with open(pdf_path, 'rb') as f:
                reader = pypdf.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() or ""
            return text
        except Exception as e:
            print(f"PDF extraction failed: {e}")
            return ""
    
    @staticmethod
    def extract_metadata(pdf_path: str) -> Dict:
        """Get PDF metadata"""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                return {
                    "pages": len(pdf.pages),
                    "created": pdf.metadata.get("CreationDate", "N/A"),
                    "author": pdf.metadata.get("Author", "Unknown")
                }
        except:
            return {"pages": 0, "created": "N/A", "author": "Unknown"}


# ═══════════════════════════════════════════════
# PART 3: DOCUMENT MANAGER (HIGH-LEVEL API)
# ═══════════════════════════════════════════════

_db_manager = None
_doc_storage = "./documents_storage"

def get_document_manager():
    """Get singleton DocumentManager instance"""
    global _db_manager
    if _db_manager is None:
        _db_manager = ChromaDBManager()
    return DocumentManager(_db_manager)

class DocumentManager:
    """High-level API for document operations"""
    
    def __init__(self, db_manager: ChromaDBManager):
        self.db = db_manager
        os.makedirs(_doc_storage, exist_ok=True)
    
    def upload_pdf(self, pdf_file_path: str, company_name: str, 
                   doc_type: str, metadata: Dict = None) -> Dict:
        """Upload and process PDF"""
        try:
            doc_id = str(uuid.uuid4())
            file_name = os.path.basename(pdf_file_path)
            
            # Extract text
            text = PDFProcessor.extract_text(pdf_file_path)
            if not text.strip():
                return {"status": "error", "message": "Could not extract text"}
            
            # Extract metrics
            structured = self._extract_metrics(text)
            
            # Store in DB
            meta = metadata or {}
            meta.update(structured)
            self.db.add_document(doc_id, company_name, doc_type, text, file_name, meta)
            
            return {
                "status": "success",
                "doc_id": doc_id,
                "file": file_name,
                "metrics": structured
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def search_documents(self, query: str, company: str = None, 
                        doc_type: str = None, top_k: int = 5) -> List[Dict]:
        """Search across documents"""
        return self.db.search_documents(query, company, doc_type, top_k)
    
    def list_documents(self, company: str = None) -> List[Dict]:
        """List documents"""
        return self.db.list_documents(company)
    
    def update_document_data(self, doc_id: str, updates: Dict) -> bool:
        """Update document"""
        return self.db.update_document_metadata(doc_id, updates)
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete document"""
        return self.db.delete_document(doc_id)
    
    def get_document_summary(self, doc_id: str) -> Dict:
        """Get document details"""
        docs = self.db.metadata.get(ids=[doc_id])
        if not docs["ids"]:
            return {}
        return docs["metadatas"][0] if docs["metadatas"] else {}
    
    @staticmethod
    def _extract_metrics(text: str) -> Dict:
        """Extract key financial metrics using regex"""
        patterns = {
            "revenue": r"revenue[:\s]+(?:₹|rs\.?)?\s*([\d,.]+)\s*(?:cr|crore)",
            "cibil": r"cibil[:\s]*(\d+)",
            "gst": r"gst[:\s]*(?:₹|rs\.?)?\s*([\d,.]+)\s*(?:cr|crore)",
            "litigation": r"(?:nclt|litigation|case)[:\s]*(\d+)",
        }
        
        metrics = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, text.lower())
            if match:
                metrics[key] = match.group(1)
        return metrics


# ═══════════════════════════════════════════════
# PART 4: LANGRAPH TOOLS (AGENT INTEGRATION)
# ═══════════════════════════════════════════════

@tool
def search_company_documents(query: str, company_name: str = "", 
                            document_type: str = "") -> str:
    """Search documents by semantic query"""
    mgr = get_document_manager()
    results = mgr.search_documents(query, company_name, document_type)
    return json.dumps({"results": results, "count": len(results)})

@tool
def get_company_documents_list(company_name: str = "") -> str:
    """List company documents"""
    mgr = get_document_manager()
    docs = mgr.list_documents(company_name)
    return json.dumps({"documents": docs, "total": len(docs)})

@tool
def extract_key_metrics_from_db(company_name: str = "") -> str:
    """Extract metrics from documents"""
    mgr = get_document_manager()
    docs = mgr.list_documents(company_name)
    metrics = {}
    for doc in docs:
        summary = mgr.get_document_summary(doc["id"])
        metrics.update({k: v for k, v in summary.items() 
                       if k in ["revenue", "cibil", "gst", "litigation"]})
    return json.dumps({"metrics": metrics})

@tool
def update_document_findings(doc_id: str, findings: str) -> str:
    """Update document with findings"""
    try:
        mgr = get_document_manager()
        updates = json.loads(findings)
        success = mgr.update_document_data(doc_id, updates)
        return json.dumps({"success": success})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})

@tool
def get_document_summary(doc_id: str) -> str:
    """Get document summary"""
    mgr = get_document_manager()
    summary = mgr.get_document_summary(doc_id)
    return json.dumps({"summary": summary})

def get_rag_tools():
    """Export all RAG tools for agent"""
    return [
        search_company_documents,
        get_company_documents_list,
        extract_key_metrics_from_db,
        update_document_findings,
        get_document_summary
    ]


# ═══════════════════════════════════════════════
# PART 5: STREAMLIT UI COMPONENTS
# ═══════════════════════════════════════════════

def render_rag_dashboard():
    """Render RAG dashboard with 5 tabs"""
    st.title("📚 RAG Document Intelligence")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📤 Upload", "📚 Manage", "🔎 Search", "📊 Metrics", "✏️ Update"])
    
    with tab1:
        render_upload_tab()
    with tab2:
        render_manage_tab()
    with tab3:
        render_search_tab()
    with tab4:
        render_metrics_tab()
    with tab5:
        render_update_tab()

def render_upload_tab():
    """Upload documents"""
    st.subheader("Upload Company Documents")
    company = st.text_input("Company Name")
    doc_type = st.selectbox("Document Type", ["Annual Report", "CIBIL", "GST", "Bank Statements", "Other"])
    file = st.file_uploader("Upload PDF", type="pdf")
    
    if st.button("🚀 Upload & Index", use_container_width=True):
        if not company or not file:
            st.error("Please fill all fields")
            return
        
        with st.spinner("Processing..."):
            mgr = get_document_manager()
            result = mgr.upload_pdf(file.name, company, doc_type)
            if result.get("status") == "success":
                st.success(f"✅ Uploaded: {result.get('file')}")
                st.json(result.get("metrics"))
            else:
                st.error(f"Error: {result.get('message')}")

def render_manage_tab():
    """List and manage documents"""
    st.subheader("Manage Documents")
    mgr = get_document_manager()
    
    company = st.text_input("Filter by Company (optional)")
    docs = mgr.list_documents(company)
    
    if docs:
        for doc in docs:
            col1, col2 = st.columns([4, 1])
            col1.write(f"📄 {doc['file']} ({doc['type']}) - {doc['company']}")
            if col2.button("🗑️", key=doc['id']):
                mgr.delete_document(doc['id'])
                st.rerun()
    else:
        st.info("No documents found")

def render_search_tab():
    """Search documents"""
    st.subheader("Search Documents")
    query = st.text_input("Search query", placeholder="e.g., 'revenue', 'CIBIL', 'bank inflow'")
    company = st.text_input("Filter by Company (optional)")
    
    if st.button("🔍 Search", use_container_width=True):
        mgr = get_document_manager()
        results = mgr.search_documents(query, company)
        st.write(f"Found {len(results)} results:")
        for r in results:
            st.write(f"**{r['metadata'].get('file', 'Unknown')}** ({r['similarity']:.2%})")
            st.caption(r['content'][:200] + "...")

def render_metrics_tab():
    """Extract metrics"""
    st.subheader("Extract Financial Metrics")
    company = st.text_input("Company Name")
    
    if st.button("🤖 Extract Metrics", use_container_width=True):
        mgr = get_document_manager()
        docs = mgr.list_documents(company)
        all_metrics = {}
        for doc in docs:
            summary = mgr.get_document_summary(doc['id'])
            all_metrics.update(summary)
        
        st.json(all_metrics) if all_metrics else st.info("No metrics extracted")

def render_update_tab():
    """Update document findings"""
    st.subheader("Update Document Findings")
    mgr = get_document_manager()
    
    company = st.text_input("Company Name")
    docs = mgr.list_documents(company)
    
    if docs:
        doc_options = {f"{d['file']} ({d['type']})": d['id'] for d in docs}
        selected = st.selectbox("Select Document", list(doc_options.keys()))
        doc_id = doc_options[selected]
        
        field = st.text_input("Field to Update")
        value = st.text_input("New Value")
        
        if st.button("💾 Save", use_container_width=True):
            mgr.update_document_data(doc_id, {field: value})
            st.success("✅ Updated")
    else:
        st.info("No documents found")
