"""
RAG Manager — Chroma DB Integration for Document Storage & Retrieval
Handles PDF ingestion, text extraction, embedding, retrieval, and updates
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
from chromadb.config import Settings
import pdfplumber
import pypdf
from fpdf import FPDF

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
CHROMA_DB_PATH = "./chroma_db"
DOCUMENTS_STORAGE_PATH = "./documents_storage"

# Create directories if they don't exist
os.makedirs(CHROMA_DB_PATH, exist_ok=True)
os.makedirs(DOCUMENTS_STORAGE_PATH, exist_ok=True)


# ──────────────────────────────────────────────
# Chroma DB Initialization
# ──────────────────────────────────────────────
class ChromaDBManager:
    """Manages Chroma DB operations for document storage and retrieval"""
    
    def __init__(self, db_path=CHROMA_DB_PATH):
        """Initialize Chroma DB client and collections"""
        self.db_path = db_path
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Initialize collections
        self._init_collections()
    
    def _init_collections(self):
        """Initialize or get existing collections"""
        try:
            self.documents_collection = self.client.get_or_create_collection(
                name="credit_documents",
                metadata={"description": "Credit documents and financial info"},
                distance_function="cosine"
            )
            
            self.metadata_collection = self.client.get_or_create_collection(
                name="document_metadata",
                metadata={"description": "Document metadata and features"},
                distance_function="cosine"
            )
            
            print("✅ Chroma DB collections initialized")
        except Exception as e:
            print(f"❌ Error initializing collections: {e}")
    
    def add_document(self, 
                     doc_id: str,
                     company_name: str,
                     document_type: str,
                     content: str,
                     file_name: str,
                     metadata: Dict[str, Any] = None) -> bool:
        """
        Add a document to Chroma DB
        
        Args:
            doc_id: Unique document ID
            company_name: Name of the company
            document_type: Type of document (e.g., 'Annual Report', 'CIBIL')
            content: Full text content of document
            file_name: Original file name
            metadata: Additional metadata
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if metadata is None:
                metadata = {}
            
            # Chunk the content for better retrieval
            chunks = self._chunk_text(content)
            
            # Add to documents collection
            for idx, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_chunk_{idx}"
                
                self.documents_collection.add(
                    ids=[chunk_id],
                    documents=[chunk],
                    metadatas=[{
                        "doc_id": doc_id,
                        "company_name": company_name,
                        "document_type": document_type,
                        "file_name": file_name,
                        "chunk_index": idx,
                        "total_chunks": len(chunks),
                        "timestamp": datetime.now().isoformat(),
                        **metadata
                    }]
                )
            
            # Add metadata entry
            self.metadata_collection.add(
                ids=[doc_id],
                documents=[f"Metadata for {document_type} - {company_name}"],
                metadatas=[{
                    "doc_id": doc_id,
                    "company_name": company_name,
                    "document_type": document_type,
                    "file_name": file_name,
                    "content_length": len(content),
                    "chunk_count": len(chunks),
                    "upload_timestamp": datetime.now().isoformat(),
                    **metadata
                }]
            )
            
            print(f"✅ Document added: {doc_id} ({len(chunks)} chunks)")
            return True
        
        except Exception as e:
            print(f"❌ Error adding document: {e}")
            return False
    
    def search_documents(self, 
                        query: str,
                        company_name: Optional[str] = None,
                        document_types: Optional[List[str]] = None,
                        top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search documents using semantic similarity
        
        Args:
            query: Search query
            company_name: Filter by company (optional)
            document_types: Filter by document types (optional)
            top_k: Number of top results to return
        
        Returns:
            List of relevant document chunks with metadata
        """
        try:
            # Build where filter if needed
            where_filter = None
            if company_name or document_types:
                where_filter = {}
                if company_name:
                    where_filter["company_name"] = company_name
                if document_types:
                    where_filter["document_type"] = {"$in": document_types}
            
            results = self.documents_collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where_filter
            )
            
            # Format results
            formatted_results = []
            if results and results['documents'] and len(results['documents']) > 0:
                for idx, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                    formatted_results.append({
                        "rank": idx + 1,
                        "content": doc,
                        "metadata": metadata,
                        "similarity_score": results['distances'][0][idx] if 'distances' in results else None
                    })
            
            return formatted_results
        
        except Exception as e:
            print(f"❌ Error searching documents: {e}")
            return []
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get full document content by ID"""
        try:
            results = self.documents_collection.get(
                ids=[f"{doc_id}_chunk_0"],
                include=["documents", "metadatas"]
            )
            
            if results and results['metadatas']:
                return {
                    "id": doc_id,
                    "metadata": results['metadatas'][0]
                }
            return None
        except Exception as e:
            print(f"❌ Error retrieving document: {e}")
            return None
    
    def update_document_metadata(self, doc_id: str, updates: Dict[str, Any]) -> bool:
        """Update metadata for a document"""
        try:
            # Get existing metadata
            existing = self.get_document_by_id(doc_id)
            if not existing:
                return False
            
            # Update metadata collection
            updated_metadata = {**existing["metadata"], **updates}
            
            self.metadata_collection.update(
                ids=[doc_id],
                metadatas=[updated_metadata]
            )
            
            print(f"✅ Document metadata updated: {doc_id}")
            return True
        
        except Exception as e:
            print(f"❌ Error updating metadata: {e}")
            return False
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and all its chunks"""
        try:
            # Get all chunks for this document
            all_items = self.documents_collection.get(
                where={"doc_id": doc_id}
            )
            
            if all_items and all_items['ids']:
                # Delete from documents collection
                self.documents_collection.delete(ids=all_items['ids'])
                
                # Delete from metadata collection
                self.metadata_collection.delete(ids=[doc_id])
            
            print(f"✅ Document deleted: {doc_id}")
            return True
        
        except Exception as e:
            print(f"❌ Error deleting document: {e}")
            return False
    
    def list_documents(self, company_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all documents or filter by company"""
        try:
            where_filter = None
            if company_name:
                where_filter = {"company_name": company_name}
            
            results = self.metadata_collection.get(where=where_filter)
            
            documents = []
            if results and results['metadatas']:
                for metadata in results['metadatas']:
                    documents.append({
                        "id": metadata.get("doc_id"),
                        "company": metadata.get("company_name"),
                        "type": metadata.get("document_type"),
                        "file_name": metadata.get("file_name"),
                        "upload_date": metadata.get("upload_timestamp"),
                        "content_length": metadata.get("content_length"),
                        "chunks": metadata.get("chunk_count")
                    })
            
            return documents
        
        except Exception as e:
            print(f"❌ Error listing documents: {e}")
            return []
    
    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk.strip())
            start = end - overlap
        
        return chunks if chunks else [text]


# ──────────────────────────────────────────────
# PDF Document Processor
# ──────────────────────────────────────────────
class PDFProcessor:
    """Process PDF files and extract text content"""
    
    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> str:
        """
        Extract text from PDF using multiple methods
        
        Args:
            pdf_path: Path to PDF file
        
        Returns:
            Extracted text content
        """
        text_content = ""
        
        try:
            # Try pdfplumber first (better for tables)
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    # Extract text
                    text_content += page.extract_text() or ""
                    text_content += "\n"
                    
                    # Try to extract tables
                    tables = page.extract_tables()
                    if tables:
                        text_content += "\n[TABLE]\n"
                        for table in tables:
                            for row in table:
                                text_content += " | ".join(str(cell) if cell else "" for cell in row) + "\n"
                        text_content += "[/TABLE]\n"
            
            if text_content.strip():
                return text_content
            
        except Exception as e:
            print(f"⚠ pdfplumber failed: {e}, trying pypdf...")
        
        try:
            # Fallback to pypdf
            with open(pdf_path, 'rb') as f:
                pdf_reader = pypdf.PdfReader(f)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text_content += page.extract_text() or ""
                    text_content += "\n"
            
            return text_content
        
        except Exception as e:
            print(f"❌ Error extracting PDF: {e}")
            return ""
    
    @staticmethod
    def extract_metadata_from_pdf(pdf_path: str) -> Dict[str, Any]:
        """Extract metadata from PDF"""
        metadata = {}
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                num_pages = len(pdf.pages)
                metadata["num_pages"] = num_pages
                metadata["file_size_kb"] = os.path.getsize(pdf_path) / 1024
        except Exception as e:
            print(f"⚠ Could not extract PDF metadata: {e}")
        
        return metadata


# ──────────────────────────────────────────────
# Document Manager (High-level API)
# ──────────────────────────────────────────────
class DocumentManager:
    """High-level document management interface"""
    
    def __init__(self, db_path=CHROMA_DB_PATH, storage_path=DOCUMENTS_STORAGE_PATH):
        """Initialize document manager"""
        self.chroma_db = ChromaDBManager(db_path)
        self.storage_path = storage_path
        self.pdf_processor = PDFProcessor()
        
        os.makedirs(storage_path, exist_ok=True)
    
    def upload_pdf(self,
                   pdf_file,
                   company_name: str,
                   document_type: str,
                   extract_structured_data: bool = True) -> Dict[str, Any]:
        """
        Upload and process a PDF document
        
        Args:
            pdf_file: File object or path
            company_name: Name of the company
            document_type: Type of document
            extract_structured_data: Whether to extract key data points
        
        Returns:
            Result dictionary with processing status
        """
        result = {
            "status": "processing",
            "doc_id": None,
            "file_name": None,
            "text_content": None,
            "structured_data": None,
            "error": None
        }
        
        try:
            # Generate document ID
            doc_id = f"doc_{int(datetime.now().timestamp())}_{uuid.uuid4().hex[:8]}"
            
            # Save file
            if isinstance(pdf_file, str):
                file_path = pdf_file
                file_name = os.path.basename(file_path)
            else:
                file_name = pdf_file.name
                file_path = os.path.join(self.storage_path, file_name)
                with open(file_path, 'wb') as f:
                    f.write(pdf_file.read())
            
            result["file_name"] = file_name
            
            # Extract text
            text_content = self.pdf_processor.extract_text_from_pdf(file_path)
            if not text_content.strip():
                result["error"] = "No text could be extracted from PDF"
                result["status"] = "failed"
                return result
            
            result["text_content"] = text_content
            
            # Extract structured data
            structured_data = None
            if extract_structured_data:
                structured_data = self._extract_structured_data(text_content)
            
            result["structured_data"] = structured_data
            
            # Add to Chroma DB
            metadata_dict = {
                "document_type": document_type,
                "file_path": file_path,
                **(structured_data or {})
            }
            
            success = self.chroma_db.add_document(
                doc_id=doc_id,
                company_name=company_name,
                document_type=document_type,
                content=text_content,
                file_name=file_name,
                metadata=metadata_dict
            )
            
            if success:
                result["status"] = "success"
                result["doc_id"] = doc_id
            else:
                result["status"] = "failed"
                result["error"] = "Failed to add document to database"
        
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
        
        return result
    
    def query_documents(self, 
                       query: str,
                       company_name: Optional[str] = None,
                       document_types: Optional[List[str]] = None,
                       top_k: int = 5) -> List[Dict[str, Any]]:
        """Query documents by semantic similarity"""
        return self.chroma_db.search_documents(
            query=query,
            company_name=company_name,
            document_types=document_types,
            top_k=top_k
        )
    
    def get_document_summary(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get complete document summary"""
        return self.chroma_db.get_document_by_id(doc_id)
    
    def list_company_documents(self, company_name: str) -> List[Dict[str, Any]]:
        """List all documents for a company"""
        return self.chroma_db.list_documents(company_name)
    
    def update_document_data(self, doc_id: str, updates: Dict[str, Any]) -> bool:
        """Update document metadata with extracted/verified data"""
        return self.chroma_db.update_document_metadata(doc_id, updates)
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document"""
        return self.chroma_db.delete_document(doc_id)
    
    def _extract_structured_data(self, text: str) -> Dict[str, Any]:
        """Extract key financial metrics from text"""
        data = {}
        
        # CIBIL Score
        cibil_match = re.search(r'(?:CIBIL|credit\s*score)[:\s]*(\d{3})', text, re.IGNORECASE)
        if cibil_match:
            data["cibil_score"] = int(cibil_match.group(1))
        
        # Revenue
        revenue_match = re.search(
            r'(?:revenue|turnover|sales)[:\s]*(?:Rs\.?|INR|₹)?\s*([\d,.]+)\s*(?:Cr|Crore|L|Lakh)?',
            text, re.IGNORECASE
        )
        if revenue_match:
            data["revenue"] = revenue_match.group(1)
        
        # Bank Inflow
        inflow_match = re.search(
            r'(?:inflow|bank.*inflow|total.*inflow)[:\s]*(?:Rs\.?|INR|₹)?\s*([\d,.]+)',
            text, re.IGNORECASE
        )
        if inflow_match:
            data["bank_inflow"] = inflow_match.group(1)
        
        # Company Age
        age_match = re.search(r'(?:established|founded|incorporated)[:\s]*(\d{4})', text, re.IGNORECASE)
        if age_match:
            year = int(age_match.group(1))
            data["company_age_years"] = datetime.now().year - year
        
        # Litigation
        litigation_count = len(re.findall(
            r'(?:lawsuit|litigation|nclt|writ|petition|case|dispute)',
            text, re.IGNORECASE
        ))
        data["litigation_count"] = litigation_count
        
        return data if data else None


# ──────────────────────────────────────────────
# Global Instance
# ──────────────────────────────────────────────
_document_manager = None

def get_document_manager() -> DocumentManager:
    """Get or create global document manager instance"""
    global _document_manager
    if _document_manager is None:
        _document_manager = DocumentManager()
    return _document_manager
