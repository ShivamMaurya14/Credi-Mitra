# 🏗️ Complete RAG Architecture Guide

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                     CREDI-MITRA WITH RAG SYSTEM                      │
└─────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                         User Interface Layer                          │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Streamlit Frontend (app.py + rag_ui.py)                    │   │
│  ├─────────────────────────────────────────────────────────────┤   │
│  │                                                             │   │
│  │  Dashboard                    RAG Dashboard               │   │
│  │  ┌──────────────────┐        ┌──────────────────┐        │   │
│  │  │ Quick Actions    │        │ Upload Tab       │        │   │
│  │  │ • Analysis       │        │ • Pick Company   │        │   │
│  │  │ • RAG Docs ←────────────→ │ • Select Type    │        │   │
│  │  │ • Settings       │        │ • Upload PDFs    │        │   │
│  │  └──────────────────┘        └──────────────────┘        │   │
│  │                                                             │   │
│  │  ┌──────────────────┐        ┌──────────────────┐        │   │
│  │  │                  │        │ Search Tab       │        │   │
│  │  │                  │        │ • Query Input    │        │   │
│  │  │                  │        │ • Top K Results  │        │   │
│  │  │                  │        │ • Metadata View  │        │   │
│  │  └──────────────────┘        └──────────────────┘        │   │
│  │                                                             │   │
│  │  ┌──────────────────┐        ┌──────────────────┐        │   │
│  │  │                  │        │ Metrics Tab      │        │   │
│  │  │                  │        │ • Auto Extract   │        │   │
│  │  │                  │        │ • All Metrics    │        │   │
│  │  │                  │        │ • Found/Missing  │        │   │
│  │  └──────────────────┘        └──────────────────┘        │   │
│  │                                                             │   │
│  │  ┌──────────────────┐        ┌──────────────────┐        │   │
│  │  │                  │        │ Manage Tab       │        │   │
│  │  │                  │        │ • List Docs      │        │   │
│  │  │                  │        │ • View Detail    │        │   │
│  │  │                  │        │ • Delete         │        │   │
│  │  └──────────────────┘        └──────────────────┘        │   │
│  │                                                             │   │
│  │  ┌──────────────────┐        ┌──────────────────┐        │   │
│  │  │  Chat Interface  │        │ Update Tab       │        │   │
│  │  │  • Agent Running │        │ • Select Doc     │        │   │
│  │  │  • Messages      │        │ • Edit Field     │        │   │
│  │  │  • Tool Output   │        │ • Save Value     │        │   │
│  │  └──────────────────┘        └──────────────────┘        │   │
│  │                                                             │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
                                   ↓
                         
┌──────────────────────────────────────────────────────────────────────┐
│                      Agent Orchestration Layer                        │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  LangGraph Agent (agent_graph.py)                          │    │
│  ├────────────────────────────────────────────────────────────┤    │
│  │                                                            │    │
│  │  Original 7 Tools:                                         │    │
│  │  ✓ list_uploaded_documents                               │    │
│  │  ✓ analyze_document                                      │    │
│  │  ✓ extract_document_data                                 │    │
│  │  ✓ crawl_web_for_litigation                              │    │
│  │  ✓ extract_numerical_features                            │    │
│  │  ✓ run_xgboost_scorer                                    │    │
│  │  ✓ generate_cam_report                                   │    │
│  │                                                            │    │
│  │  NEW RAG Tools: [get_rag_tools()]                         │    │
│  │  ✓ search_company_documents ────────────┐                │    │
│  │  ✓ get_company_documents_list ──────────┤                │    │
│  │  ✓ extract_key_metrics_from_db ────────┤                │    │
│  │  ✓ update_document_findings ───────────┤                │    │
│  │  ✓ get_document_summary ──────────────┤                │    │
│  │                                         │                │    │
│  │  LLM (Groq/Gemini) makes decisions     │                │    │
│  │  Calls appropriate tools               │                │    │
│  │  Processes results                     │                │    │
│  │                                         │                │    │
│  └────────────────────────────────────────┼────────────────┘    │
│                                            │                      │
└────────────────────────────────────────────┼──────────────────────┘
                                             ↓

┌──────────────────────────────────────────────────────────────────────┐
│                         RAG Core Layer                                │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  rag_tools.py — Tool Implementations                       │    │
│  ├────────────────────────────────────────────────────────────┤    │
│  │                                                            │    │
│  │  search_company_documents()                               │    │
│  │  ├─ Query semantically                                   │    │
│  │  ├─ Get document manager                                 │    │
│  │  └─ Return top-k results ────────────┐                   │    │
│  │                                       ↓                   │    │
│  │  get_company_documents_list()        DocumentManager     │    │
│  │  ├─ List by company                  (rag_manager.py)    │    │
│  │  └─ Return metadata ──────────────┐                       │    │
│  │                                    │  ┌─────────────┐    │    │
│  │  extract_key_metrics_from_db()     ├─→│ ChromaDB    │    │    │
│  │  ├─ Search for metrics patterns     │  │ Operations│    │    │
│  │  └─ Multi-query extraction ────────┤  ├─────────────┤    │    │
│  │                                    │  │ PDF         │    │    │
│  │  update_document_findings()        │  │ Processor   │    │    │
│  │  ├─ Update metadata                │  │             │    │    │
│  │  └─ Persist to DB ─────────────────┤  └─────────────┘    │    │
│  │                                    │                      │    │
│  │  get_document_summary()            │                      │    │
│  │  ├─ Retrieve doc info              │                      │    │
│  │  └─ Return full summary ───────────┘                      │    │
│  │                                                            │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
                                   ↓

┌──────────────────────────────────────────────────────────────────────┐
│                    RAG Management Core Layer                          │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  rag_manager.py — Core Management System                   │    │
│  ├────────────────────────────────────────────────────────────┤    │
│  │                                                            │    │
│  │  ChromaDBManager Class:                                   │    │
│  │  ├─ add_document()        → Store in Chroma              │    │
│  │  ├─ search_documents()    → Semantic query                │    │
│  │  ├─ get_document_by_id()  → Retrieve                      │    │
│  │  ├─ update_doc_metadata() → Modify                        │    │
│  │  ├─ delete_document()     → Remove                        │    │
│  │  ├─ list_documents()      → Browse                        │    │
│  │  └─ _chunk_text()         → Split for embedding           │    │
│  │                                                            │    │
│  │  PDFProcessor Class:                                      │    │
│  │  ├─ extract_text_from_pdf()    → pdfplumber + pypdf      │    │
│  │  │  ├─ Text extraction                                   │    │
│  │  │  ├─ Table detection                                   │    │
│  │  │  └─ Fallback methods                                  │    │
│  │  └─ extract_metadata_from_pdf() → Pages, size            │    │
│  │                                                            │    │
│  │  DocumentManager Class:                                  │    │
│  │  ├─ upload_pdf()            → Full pipeline              │    │
│  │  ├─ query_documents()       → Semantic search            │    │
│  │  ├─ get_document_summary()  → Full info                  │    │
│  │  ├─ list_company_documents()→ Company docs               │    │
│  │  ├─ update_document_data()  → Save findings              │    │
│  │  ├─ delete_document()       → Remove                     │    │
│  │  └─ _extract_structured_data()→ Auto-extract metrics    │    │
│  │                                                            │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
                                   ↓

┌──────────────────────────────────────────────────────────────────────┐
│                      Vector Database Layer                            │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  Chroma DB (Persistent Vector Store)                       │    │
│  ├────────────────────────────────────────────────────────────┤    │
│  │                                                            │    │
│  │  Collection 1: credit_documents                            │    │
│  │  ├─ Purpose: Document chunks for search                   │    │
│  │  ├─ Fields:                                                │    │
│  │  │  ├─ id: "doc_123456_chunk_0"                          │    │
│  │  │  ├─ document: Text chunk (1000 chars)                 │    │
│  │  │  └─ metadata:                                          │    │
│  │  │     ├─ doc_id                                         │    │
│  │  │     ├─ company_name                                   │    │
│  │  │     ├─ document_type                                  │    │
│  │  │     ├─ file_name                                      │    │
│  │  │     ├─ chunk_index                                    │    │
│  │  │     ├─ timestamp                                      │    │
│  │  │     └─ custom fields                                  │    │
│  │  │                                                         │    │
│  │  │ Search: query → embeddings → cosine similarity       │    │
│  │  │                                                         │    │
│  │  ├─ Embedding Model: all-MiniLM-L6-v2 (384 dims)        │    │
│  │  └─ Distance Metric: Cosine Similarity                   │    │
│  │                                                            │    │
│  │  Collection 2: document_metadata                           │    │
│  │  ├─ Purpose: Full document information                   │    │
│  │  ├─ Fields:                                                │    │
│  │  │  ├─ id: "doc_123456"                                  │    │
│  │  │  ├─ document: "Metadata for [type]"                  │    │
│  │  │  └─ metadata:                                          │    │
│  │  │     ├─ doc_id                                         │    │
│  │  │     ├─ company_name                                   │    │
│  │  │     ├─ document_type                                  │    │
│  │  │     ├─ file_name                                      │    │
│  │  │     ├─ content_length                                 │    │
│  │  │     ├─ chunk_count                                    │    │
│  │  │     ├─ cibil_score (auto-extracted)                 │    │
│  │  │     ├─ revenue                                        │    │
│  │  │     ├─ bank_inflow                                    │    │
│  │  │     ├─ company_age_years                              │    │
│  │  │     ├─ litigation_count                               │    │
│  │  │     └─ custom fields                                  │    │
│  │  │                                                         │    │
│  │  └─ Filtering: BY company_name, document_type          │    │
│  │                                                            │    │
│  │  Storage Location: ./chroma_db/ (persistent)            │    │
│  │  Auto-created on first upload                           │    │
│  │  Persists across app restarts                           │    │
│  │                                                            │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
                                   ↓

┌──────────────────────────────────────────────────────────────────────┐
│                       Storage Layer                                   │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  File System:                                                         │
│                                                                       │
│  ./chroma_db/                    Vector database (Persistent)        │
│  ├─ Index files & metadata                                          │
│  └─ Collection data                                                  │
│                                                                       │
│  ./documents_storage/            Original uploaded PDFs              │
│  ├─ Annual_Report_2024.pdf                                          │
│  ├─ CIBIL_Report.pdf                                                │
│  └─ Bank_Statements.pdf                                             │
│                                                                       │
│  ./uploads/                      Application-specific folders        │
│  └─ Company_Name_AppNo/                                             │
│     ├─ Application_Form.pdf                                         │
│     ├─ analysis_jsons/          Extracted analysis                  │
│     └─ ...                                                           │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

### Upload → Store → Search Flow

```
1. USER UPLOADS PDF
   └─→ app.py (rag_ui.py)

2. PROCESS PDF
   └─→ rag_manager.py
       └─→ DocumentManager.upload_pdf()
           ├─→ PDFProcessor.extract_text_from_pdf()
           │   ├─ Try pdfplumber (tables + text)
           │   └─ Fallback to pypdf
           ├─→ Extract metadata
           ├─→ Chunk text (1000 chars, 100 overlap)
           ├─→ Extract structured data (regex patterns)
           │   ├─ CIBIL score
           │   ├─ Revenue
           │   ├─ Bank inflow
           │   ├─ Company age
           │   ├─ Litigation count
           │   └─ GST revenue
           └─→ Store in Chroma DB

3. STORAGE IN CHROMA
   └─→ ChromaDBManager.add_document()
       ├─→ credit_documents collection
       │   └─ One entry per chunk
       │       ├─ ID: doc_id_chunk_n
       │       ├─ document: text
       │       └─ metadata: all fields
       └─→ document_metadata collection
           └─ One entry per document
               ├─ ID: doc_id
               ├─ metadata: full info + extracted features
               └─ for filtering & retrieval

4. SEMANTIC SEARCH
   └─→ Agent calls search_company_documents("revenue")
       └─→ rag_tools.py
           └─→ DocumentManager.query_documents()
               └─→ ChromaDBManager.search_documents()
                   ├─ Query embedding via sentence-transformers
                   ├─ Cosine similarity against all chunks
                   ├─ Apply filters (company, doc_type)
                   ├─ Sort by similarity
                   └─ Return top-k results

5. RESULTS TO AGENT
   └─→ Format as JSON with:
       ├─ content: relevant text chunks
       ├─ metadata: source info
       ├─ similarity_score: relevance
       └─ rank: order

6. VERIFICATION & UPDATE
   └─→ Agent or analyst reviews
       └─→ Update with corrections
           └─→ update_document_findings()
               └─→ ChromaDBManager.update_document_metadata()
                   └─→ Persisted to Chroma DB

```

---

## Integration Points with Existing System

### How RAG Connects to LangGraph Agent

```
┌─────────────────────────┐
│  LangGraph State        │
├─────────────────────────┤
│ • messages: [...]       │
│ • thread_id: uuid       │
│ • company_name: str     │
│ • features: {...}       │
│ • ml_decision: {...}    │
├─────────────────────────┤
│ ALL_TOOLS:              │
│ ├─ Existing 7 tools     │
│ └─ NEW: 5 RAG tools ←──┐│
└─────────────────────────┘│
                           │
                           ↓ Agent needs info
                           
                    search_company_documents()
                           │
                     Uses: company_name, query
                           │
                           ↓
                    
                    DocumentManager.query()
                           │
                     Returns: top_k results
                           │
                           ↓
                           
                    Agent processes results
                    Extracts features
                    Updates state
                           │
                           ↓
                           
                    update_document_findings()
                           │
                    Saves: verified_values
                           │
                           ↓
                           
                    Continues with XGBoost
                    Generates CAM report
```

---

## Feature Extraction Pipeline

```
┌──────────────────────────┐
│ Uploaded Documents       │
└──────────────────────────┘
         ↓
    [Auto-extract via regex patterns]
         ↓
┌──────────────────────────┐
│ Structured Data Found:   │
├──────────────────────────┤
│ • CIBIL Score            │
│ • Revenue                │
│ • Bank Inflow            │
│ • Company Age            │
│ • Litigation Count       │
│ • GST Revenue            │
│ • Assets                 │
│ • Liabilities            │
└──────────────────────────┘
         ↓
    [Semantic search for metrics]
         ↓
┌──────────────────────────┐
│ Query Examples:          │
├──────────────────────────┤
│ "company revenue"    → Revenue field
│ "CIBIL credit"       → CIBIL Score
│ "bank inflow"        → Bank Inflow
│ "litigation dispute" → Litigation Count
│ "company age years"  → Company Age
└──────────────────────────┘
         ↓
    [Results + sources]
         ↓
┌──────────────────────────┐
│ Extracted Metrics Tab    │
├──────────────────────────┤
│ ✓ Revenue (found)        │
│ ✓ CIBIL (found)          │
│ ✗ Bank Inflow (missing)  │
│ ...                      │
└──────────────────────────┘
```

---

## Data Persistence

**What persists:**
- ✅ Chroma DB collections (./chroma_db/)
- ✅ Uploaded PDFs (./documents_storage/)
- ✅ Extracted metadata
- ✅ Updated findings
- ✅ Company associations

**What doesn't persist:**
- ❌ Chat history (app restart)
- ❌ Temporary embeddings (regenerated on search)
- ❌ Session state (except in Streamlit cache)

**Restart behavior:**
- App restarts → DB accessible
- Upload same company again → Old docs still there
- Search queries → Use existing stored documents

---

## Performance Characteristics

| Operation | Time | Notes |
|-----------|------|-------|
| Upload PDF | 2-5s | Varies with size |
| Text Extract | 1-2s | pdfplumber method |
| Chunking | <0.5s | Fast local operation |
| Embedding | 0.5-1s | First time + cached |
| Search Query | 0.5-1s | Cosine similarity |
| Store to DB | 0.2-0.5s | Batch operation |
| Metrics Extract | 1-2s | Multi-query search |
| Update Metadata | 0.2s | atomic operation |

---

## Security Considerations

**Data stored locally:**
- No cloud upload required
- Full control over data
- GDPR compliant (no external processing)

**Embedding model:**
- Runs locally
- No model queries sent externally
- Embeddings stored in local DB

**PDF storage:**
- Stored in workspace filesystem
- Accessible only to process
- Can be encrypted at filesystem level

