"""
RAG Integration Tools for LangGraph Agent
Provides tools for document retrieval and semantic search during credit analysis
"""

import json
import os
from typing import Dict, List, Any, Optional, Annotated
import re

from langchain_core.tools import tool
import streamlit as st

from rag_manager import get_document_manager

# ──────────────────────────────────────────────
# RAG Retrieval Tools
# ──────────────────────────────────────────────

@tool
def search_company_documents(query: str, 
                            company_name: Optional[str] = None,
                            document_types: Optional[List[str]] = None,
                            top_results: int = 5) -> str:
    """
    Search uploaded company documents using semantic similarity.
    Use this to find relevant financial data, reports, or information.
    
    Args:
        query: What you're looking for (e.g., "revenue", "bank statements", "litigation")
        company_name: Filter by company name (optional)
        document_types: Filter by doc types e.g. ["Annual Report", "CIBIL"] (optional)
        top_results: Number of results to return (1-10)
    
    Returns:
        JSON with search results including document chunks and metadata
    """
    try:
        doc_manager = get_document_manager()
        
        # Get company name from session if not provided
        if not company_name:
            company_name = st.session_state.get("company_name")
        
        results = doc_manager.query_documents(
            query=query,
            company_name=company_name,
            document_types=document_types,
            top_k=top_results
        )
        
        # Format results for readable output
        if not results:
            return json.dumps({
                "status": "no_results",
                "message": "No relevant documents found for this query",
                "query": query,
                "company": company_name
            })
        
        formatted_results = []
        for result in results:
            formatted_results.append({
                "document": result["metadata"].get("file_name", "Unknown"),
                "document_type": result["metadata"].get("document_type", "Unknown"),
                "relevant_content": result["content"][:500],
                "full_metadata": result["metadata"],
                "similarity_score": result.get("similarity_score")
            })
        
        return json.dumps({
            "status": "success",
            "total_results": len(formatted_results),
            "results": formatted_results,
            "query": query
        }, indent=2)
    
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": str(e),
            "query": query
        })


@tool
def get_company_documents_list(company_name: Optional[str] = None) -> str:
    """
    Get list of all documents uploaded for a company.
    Use this to understand what documents are available for analysis.
    
    Args:
        company_name: Company name (uses session if not provided)
    
    Returns:
        List of documents with metadata
    """
    try:
        doc_manager = get_document_manager()
        
        if not company_name:
            company_name = st.session_state.get("company_name")
        
        if not company_name:
            return json.dumps({
                "status": "warning",
                "message": "No company name provided or found in session"
            })
        
        documents = doc_manager.list_company_documents(company_name)
        
        if not documents:
            return json.dumps({
                "status": "no_documents",
                "company": company_name,
                "message": "No documents found for this company"
            })
        
        return json.dumps({
            "status": "success",
            "company": company_name,
            "total_documents": len(documents),
            "documents": documents
        }, indent=2)
    
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": str(e)
        })


@tool
def extract_key_metrics_from_db(company_name: Optional[str] = None,
                                document_types: Optional[List[str]] = None) -> str:
    """
    Extract key financial metrics from all stored documents.
    This searches for common metrics like revenue, CIBIL, bank inflow, litigation.
    
    Args:
        company_name: Filter by company (optional)
        document_types: Filter by document types (optional)
    
    Returns:
        Extracted metrics from documents
    """
    try:
        doc_manager = get_document_manager()
        
        if not company_name:
            company_name = st.session_state.get("company_name")
        
        # Search for common metrics
        metric_queries = {
            "revenue": "company revenue turnover sales",
            "cibil": "CIBIL credit score rating",
            "bank_inflow": "bank inflow deposits transactions",
            "litigation": "lawsuit litigation nclt dispute",
            "company_age": "established founded incorporated",
            "gst": "GST revenue registration",
            "assets": "assets total assets balance sheet",
            "liabilities": "liabilities debts",
        }
        
        extracted_metrics = {}
        
        for metric_name, query_text in metric_queries.items():
            results = doc_manager.query_documents(
                query=query_text,
                company_name=company_name,
                document_types=document_types,
                top_k=3
            )
            
            if results:
                extracted_metrics[metric_name] = {
                    "found": True,
                    "sources": [r["metadata"].get("file_name") for r in results[:2]],
                    "references": [r["content"][:200] for r in results[:1]]
                }
            else:
                extracted_metrics[metric_name] = {"found": False}
        
        return json.dumps({
            "status": "success",
            "company": company_name,
            "metrics": extracted_metrics
        }, indent=2)
    
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": str(e)
        })


@tool
def update_document_findings(doc_id: str, findings: Dict[str, Any]) -> str:
    """
    Update findings/analysis results for a document in the database.
    Use this to save analysis results, verified data, or user corrections.
    
    Args:
        doc_id: Document ID
        findings: Dictionary of key-value findings to store
                 e.g., {"verified_revenue": "100Cr", "risk_level": "low"}
    
    Returns:
        Confirmation of update
    """
    try:
        doc_manager = get_document_manager()
        
        success = doc_manager.update_document_data(doc_id, findings)
        
        if success:
            return json.dumps({
                "status": "success",
                "message": f"Document {doc_id} updated with findings",
                "findings_updated": findings
            })
        else:
            return json.dumps({
                "status": "failed",
                "message": f"Could not update document {doc_id}"
            })
    
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": str(e)
        })


@tool
def get_document_summary(doc_id: str) -> str:
    """
    Get complete summary and metadata for a specific document.
    
    Args:
        doc_id: Document ID
    
    Returns:
        Document summary and metadata
    """
    try:
        doc_manager = get_document_manager()
        
        doc_summary = doc_manager.get_document_summary(doc_id)
        
        if not doc_summary:
            return json.dumps({
                "status": "not_found",
                "doc_id": doc_id
            })
        
        return json.dumps({
            "status": "success",
            "document": doc_summary
        }, indent=2)
    
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": str(e)
        })


# ──────────────────────────────────────────────
# Helper function to add tools to agent
# ──────────────────────────────────────────────
def get_rag_tools() -> List:
    """
    Get list of all RAG tools for agent graph
    
    Returns:
        List of tool functions
    """
    return [
        search_company_documents,
        get_company_documents_list,
        extract_key_metrics_from_db,
        update_document_findings,
        get_document_summary,
    ]
