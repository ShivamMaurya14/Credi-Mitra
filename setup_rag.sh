#!/bin/bash
# CREDI-MITRA RAG System — Quick Start Script

echo "🚀 CREDI-MITRA RAG Implementation Setup Guide"
echo "=============================================="
echo ""

# Check Python
echo "✓ Checking Python installation..."
python --version
echo ""

# Step 1: Install dependencies
echo "📦 Step 1: Installing RAG dependencies..."
echo "Running: pip install chromadb sentence-transformers"
echo ""
pip install chromadb sentence-transformers pdfplumber pypdf --quiet

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully!"
else
    echo "❌ Failed to install dependencies"
    echo "Try manually: pip install chromadb sentence-transformers -r requirements.txt"
    exit 1
fi
echo ""

# Step 2: Verify installation
echo "✓ Verifying installations..."
python -c "import chromadb; print('✅ Chroma DB installed')" 2>/dev/null
python -c "import sentence_transformers; print('✅ Sentence Transformers installed')" 2>/dev/null
python -c "import pdfplumber; print('✅ PDF Processing installed')" 2>/dev/null
echo ""

# Step 3: Create necessary directories
echo "📁 Creating storage directories..."
mkdir -p ./chroma_db
mkdir -p ./documents_storage
mkdir -p ./uploads
mkdir -p ./temp_storage
echo "✅ Directories ready"
echo ""

# Step 4: Show file status
echo "📂 Files created for RAG system:"
echo "   ✓ rag_manager.py - Core RAG management"
echo "   ✓ rag_tools.py - LangGraph integration tools"
echo "   ✓ rag_ui.py - Streamlit dashboard"
echo "   ✓ RAG_IMPLEMENTATION.md - Detailed documentation"
echo "   ✓ RAG_SETUP_SUMMARY.md - Implementation summary"
echo ""

# Step 5: Ready to run
echo "🎯 Ready to run Credi-Mitra with RAG!"
echo ""
echo "To start the application:"
echo "   streamlit run app.py"
echo ""
echo "Then:"
echo "   1. Log in to dashboard"
echo "   2. Click '📚 RAG Document Intelligence'"
echo "   3. Upload your PDF documents"
echo "   4. Use semantic search and metrics extraction"
echo ""

echo "✨ RAG system fully configured!"
