"""
RAG (Retrieval-Augmented Generation) Module for Credi-Mitra
Unified module for document management and vector retrieval using Pinecone Cloud.
"""

import os
import json
import uuid
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import hashlib
import nest_asyncio
import streamlit as st

import pdfplumber
import pypdf
from langchain_core.tools import tool
from llama_parse import LlamaParse
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from pinecone import Pinecone, ServerlessSpec
except ImportError:
    Pinecone = None

# Apply nest_asyncio for LlamaParse
nest_asyncio.apply()

# ═══════════════════════════════════════════════
# PART 1: PINECONE DB MANAGER
# ═══════════════════════════════════════════════

class PineconeDBManager:
    """Vector database management via Pinecone Cloud"""
    
    def __init__(self, index_name="credi-mitra", model_choice=None):
        self.api_key = os.environ.get("PINECONE_API_KEY")
        if not self.api_key:
            raise ValueError("PINECONE_API_KEY not found in environment")
        
        if Pinecone is None:
            raise ImportError("pinecone-client is not installed. Run 'pip install pinecone-client'")

        self.pc = Pinecone(api_key=self.api_key)
        self.index_name = index_name
        
        # ── Resolve Embedding Model Provider ──
        # We strictly honor suffixes if present, otherwise fallback to keyword detection.
        model_choice_str = str(model_choice).lower() if model_choice else ""
        
        if "(openai)" in model_choice_str:
            provider = "openai"
        elif "(google)" in model_choice_str:
            provider = "google"
        elif "(groq)" in model_choice_str:
            provider = "google" # Default Groq reasoning to Google embeddings
        else:
            # Fallback keyword detection
            provider = "google"
            if "openai" in model_choice_str or "gpt" in model_choice_str:
                # Special Case: skip "gpt-oss" which is Groq-hosted
                if "gpt-oss" not in model_choice_str:
                    provider = "openai"
            elif "google" in model_choice_str or "gemini" in model_choice_str:
                provider = "google"
        
        gemini_api_key = os.environ.get("gemini_api_key") or os.environ.get("GOOGLE_API_KEY")
        openai_api_key = os.environ.get("OPENAI_API_KEY")

        if provider == "openai":
            # OpenAI embeddings (text-embedding-3-small) → 1536-dim
            self.embedding_function = OpenAIEmbeddings(
                model="text-embedding-3-small", 
                openai_api_key=openai_api_key
            )
            self.dimension = 1536
        else:
            # Google Gemini embeddings (models/gemini-embedding-001) → 3072-dim
            self.embedding_function = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                google_api_key=gemini_api_key,
                task_type="retrieval_query"
            )
            self.dimension = 3072

        # ── Handle Pinecone Index Recreation ──
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        
        should_create = False
        if self.index_name not in existing_indexes:
            should_create = True
        else:
            # Check if dimension matches
            try:
                index_info = self.pc.describe_index(self.index_name)
                if index_info.dimension != self.dimension:
                    print(f"[RAG] Dimension mismatch ({index_info.dimension} vs {self.dimension}). Recreating index...")
                    self.pc.delete_index(self.index_name)
                    # Wait for deletion
                    import time
                    for _ in range(10):
                        if self.index_name not in [idx.name for idx in self.pc.list_indexes()]:
                            break
                        time.sleep(2)
                    should_create = True
            except Exception:
                should_create = True
        
        if should_create:
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
            # Wait for index to be ready
            import time
            while not self.pc.describe_index(self.index_name).status['ready']:
                time.sleep(1)
        
        self.index = self.pc.Index(self.index_name)

    def reset_database(self):
        """Delete all vectors in the index"""
        try:
            # Ignore errors if namespace is empty/not found (404)
            self.index.delete(delete_all=True)
            return True
        except Exception as e:
            # Check if it's a 404 namespace error and skip it
            if "404" in str(e) or "Namespace not found" in str(e):
                return True
            print(f"Error resetting Pinecone: {e}")
            return False

    def add_document(self, doc_id: str, company: str, doc_type: str, 
                      content: str, file_name: str, metadata: Optional[Dict] = None) -> bool:
        """Add document with chunks and metadata to Pinecone"""
        try:
            metadata = metadata or {}
            
            # Use RecursiveCharacterTextSplitter for better semantic chunking
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=150,
                length_function=len,
                is_separator_regex=False,
            )
            chunks = text_splitter.split_text(content)
            
            vectors = []
            for idx, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_chunk_{idx}"
                # Handle potential embedding errors
                try:
                    embedding = self.embedding_function.embed_query(chunk)
                except Exception as e:
                    print(f"[RAG] Embedding failed for chunk {idx}: {e}")
                    continue
                
                vectors.append({
                    "id": chunk_id,
                    "values": embedding,
                    "metadata": {
                        "text": chunk,
                        "doc_id": doc_id, 
                        "company": company, 
                        "type": doc_type,
                        "file": file_name, 
                        "chunk": idx, 
                        "timestamp": datetime.now().isoformat(),
                        **metadata
                    }
                })
            
            # Upsert in batches of 100
            for i in range(0, len(vectors), 100):
                self.index.upsert(vectors=vectors[i:i+100])
            
            return True
        except Exception as e:
            print(f"Error adding to Pinecone: {e}")
            return False

    def search_documents(self, query: str, company: str = None, 
                        doc_type: str = None, top_k: int = 5) -> List[Dict]:
        """Search Pinecone"""
        try:
            filter_dict = {}
            if company: filter_dict["company"] = company
            if doc_type: filter_dict["type"] = doc_type
            
            embedding = self.embedding_function.embed_query(query)
            
            results = self.index.query(
                vector=embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict if filter_dict else None
            )
            
            output = []
            for res in results.get("matches", []):
                output.append({
                    "id": res["id"],
                    "content": res["metadata"].get("text", ""),
                    "metadata": res["metadata"],
                    "similarity": res["score"]
                })
            return output
        except Exception as e:
            print(f"Pinecone search error: {e}")
            if hasattr(e, 'body'):
                print(f"Error body: {e.body}")
            return []

    def add_web_result(self, result_id: str, company: str, content: str, metadata: Optional[Dict] = None) -> bool:
        """Add web result to Pinecone"""
        try:
            embedding = self.embedding_function.embed_query(content)
            self.index.upsert(vectors=[{
                "id": result_id,
                "values": embedding,
                "metadata": {
                    "text": content,
                    "company": company,
                    "type": "web_search",
                    "timestamp": datetime.now().isoformat(),
                    **(metadata or {})
                }
            }])
            return True
        except Exception as e:
            print(f"Error adding web result to Pinecone: {e}")
            return False

    def search_web_results(self, query: str, company: Optional[str] = None, top_k: int = 5) -> List[Dict]:
        """Search across stored web findings in Pinecone"""
        try:
            filter_dict = {"type": "web_search"}
            if company: filter_dict["company"] = company
            
            embedding = self.embedding_function.embed_query(query)
            results = self.index.query(
                vector=embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict
            )
            
            output = []
            for res in results.get("matches", []):
                output.append({
                    "id": res["id"],
                    "content": res["metadata"].get("text", ""),
                    "metadata": res["metadata"],
                    "similarity": res["score"]
                })
            return output
        except Exception as e:
            print(f"Pinecone web search error: {e}")
            return []

    def list_documents(self, company: str = None) -> List[Dict]:
        """List documents by querying metadata for unique doc_ids"""
        try:
            # Query with a dummy vector and high top_k to get a representative sample of docs
            # Or better, fetch by prefix if the doc_id pattern allows.
            # For brevity in this RAG module, we query by metadata.
            results = self.index.query(
                vector=[0.0] * 768, # dummy vector
                top_k=1000,
                include_metadata=True,
                filter={"company": company} if company else None
            )
            
            unique_docs = {}
            for res in results.get("matches", []):
                meta = res["metadata"]
                doc_id = meta.get("doc_id")
                if doc_id and doc_id not in unique_docs:
                    unique_docs[doc_id] = {
                        "id": doc_id,
                        "company": meta.get("company"),
                        "type": meta.get("type"),
                        "file": meta.get("file")
                    }
            return list(unique_docs.values())
        except Exception as e:
            print(f"Pinecone list error: {e}")
            return []

    def update_document_metadata(self, doc_id: str, updates: Dict) -> bool:
        """Partial updates in Pinecone are limited, so we skip for now or re-upsert if needed"""
        return False

    def delete_document(self, doc_id: str) -> bool:
        """Delete all chunks for a document"""
        try:
            self.index.delete(filter={"doc_id": doc_id})
            return True
        except Exception as e:
            print(f"Pinecone delete error: {e}")
            return False

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """Split text into chunks using RecursiveCharacterTextSplitter"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,
            is_separator_regex=False,
        )
        return text_splitter.split_text(str(text))


# ═══════════════════════════════════════════════
# PART 2: PDF PROCESSOR
# ═══════════════════════════════════════════════

class PDFProcessor:
    """Extract and process PDF content"""
    
    @staticmethod
    def extract_text(pdf_path: str) -> str:
        """Extract text from PDF using LlamaParse with fallback"""
        llama_key = os.environ.get("llama_cloud_key") or os.environ.get("LLAMA_CLOUD_API_KEY")
        
        if llama_key:
            try:
                parser = LlamaParse(api_key=llama_key, result_type="markdown", verbose=False)
                parsed_docs = parser.load_data(pdf_path)
                if parsed_docs:
                    return "\n".join([d.text for d in parsed_docs])
            except Exception as e:
                print(f"LlamaParse failed: {e}. Falling back to standard extraction.")

        # Fallback 1: pdfplumber
        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""
            if text.strip():
                return text
        except:
            pass
        
        # Fallback 2: pypdf
        try:
            with open(pdf_path, 'rb') as f:
                reader = pypdf.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() or ""
            return text
        except Exception as e:
            print(f"All PDF extraction methods failed: {e}")
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
_LAST_MODEL_PROVIDER = None

def get_document_manager(model_choice=None):
    """Get singleton DocumentManager instance using Pinecone Cloud.
    Recreates manager if switching between Google and OpenAI to ensure correct embedding dimensions.
    """
    global _db_manager, _LAST_MODEL_PROVIDER
    
    # Auto-resolve from session state if not provided
    if model_choice is None:
        model_choice = st.session_state.get("selected_model")
    
    # Identify provider for current choice
    m_lower = str(model_choice).lower() if model_choice else ""
    
    if "(openai)" in m_lower:
        current_provider = "openai"
    elif "(google)" in m_lower or "(groq)" in m_lower:
        current_provider = "google"
    else:
        # Fallback keyword detection
        if "openai" in m_lower or "gpt" in m_lower:
            if "gpt-oss" in m_lower: current_provider = "google"
            else: current_provider = "openai"
        else:
            current_provider = "google"
    
    # If no manager exists OR provider changed, recreate manager
    if _db_manager is None or current_provider != _LAST_MODEL_PROVIDER:
        _db_manager = PineconeDBManager(model_choice=model_choice)
        _LAST_MODEL_PROVIDER = current_provider
        print(f"[CREDI-MITRA] Vector Store initialized for {current_provider.upper()} embeddings")
        
    return DocumentManager(_db_manager)

class DocumentManager:
    """High-level API for document operations"""
    
    def __init__(self, db_manager: PineconeDBManager):
        self.db = db_manager
        self.dimension = db_manager.dimension
        os.makedirs(os.environ.get("DOC_STORAGE_PATH", "./documents_storage"), exist_ok=True)

    def reset_session(self):
        """Clear all database data and remove cached text files"""
        self.db.reset_database()
        # Also clear temp if needed
        import shutil
        if os.path.exists("temp"):
            shutil.rmtree("temp")
            os.makedirs("temp", exist_ok=True)
        return True
    
    def upload_pdf(self, pdf_file_path: str, company_name: str, 
                   doc_type: str, metadata: Optional[Dict] = None) -> Dict:
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
                "text_content": text,
                "metrics": structured
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def search_documents(self, query: str, company: str = None, 
                        doc_type: str = None, top_k: int = 5) -> List[Dict]:
        """Search across documents"""
        return self.db.search_documents(query, company, doc_type, top_k)
    
    def search_web_results(self, query: str, company: str = None, top_k: int = 5) -> List[Dict]:
        """Search across web findings"""
        return self.db.search_web_results(query, company, top_k)

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
        """Get document details from Pinecone metadata"""
        results = self.db.index.query(
            vector=[0.0] * self.dimension,
            top_k=1,
            include_metadata=True,
            filter={"doc_id": doc_id}
        )
        if results.get("matches"):
            return results["matches"][0]["metadata"]
        return {}
    
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
    """Search uploaded documents by semantic query."""
    mgr = get_document_manager()
    results = mgr.search_documents(query, company_name, document_type)
    return json.dumps({"results": results, "count": len(results)})

@tool
def search_analyzed_web_findings(query: str, company_name: str = "") -> str:
    """Search pre-analyzed web research results for a company.
    Use this to find specific mentions of litigation, NCLT, or sentiment from previous web crawls.
    """
    mgr = get_document_manager()
    results = mgr.search_web_results(query, company_name)
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
        if summary:
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
        search_analyzed_web_findings,
        get_company_documents_list,
        extract_key_metrics_from_db,
        update_document_findings,
        get_document_summary
    ]
