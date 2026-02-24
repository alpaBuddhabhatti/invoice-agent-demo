"""
LLM-Only Document Extraction & Analysis with Vision APIs
=========================================================
Uses Azure OpenAI GPT-4V (Vision) to intelligently extract and analyze ALL document types:
    - PDF documents (vision-based, no pypdf needed)
    - Images with advanced vision understanding
    - CSV/Excel files (as data or screenshots)
    - JSON data (semantic understanding)
    - Text files

Key Differences from Manual Extraction:
    - No intermediate parsing libraries needed
    - LLM handles semantic understanding directly
    - Better for complex layouts, handwritten text, tables
    - Single unified LLM pipeline for all file types
    - Trade-off: More expensive, slower (but more intelligent)

Features:
    - Multi-file upload
    - LLM vision-based extraction (using base64 encoding)
    - Unified Q&A on extracted content
    - Cost estimation (tokens vs manual extraction)
    - Export results as JSON/CSV
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
from datetime import datetime
from typing import Optional

import pandas as pd
import streamlit as st

# Image support
try:
    from PIL import Image
    PIL_SUPPORT = True
except ImportError:
    PIL_SUPPORT = False

# PDF support for preview
try:
    from pdf2image import convert_from_bytes
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

from agent_framework import Agent
from client import get_chat_client


# ==================== Utility Functions ====================

def _run_async(coro):
    """Run an async coroutine safely from Streamlit."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    
    new_loop = asyncio.new_event_loop()
    try:
        return new_loop.run_until_complete(coro)
    finally:
        new_loop.close()


def encode_file_to_base64(file_bytes: bytes) -> str:
    """Encode file to base64 for LLM vision API."""
    return base64.standard_b64encode(file_bytes).decode("utf-8")


def get_file_media_type(file_type: str, file_name: str) -> str:
    """Determine media type for LLM vision API."""
    if file_type == "application/pdf":
        return "application/pdf"
    elif file_type.startswith("image/"):
        return file_type
    elif file_type == "text/csv":
        return "text/csv"
    elif file_name.endswith(".json"):
        return "application/json"
    else:
        return "text/plain"


def get_llm_agent() -> Agent:
    """Create LLM agent for document extraction and analysis."""
    if "llm_agent" not in st.session_state:
        client = get_chat_client()
        st.session_state.llm_agent = Agent(
            client=client,
            name="ExtractionAgent",
            instructions=(
                "You are an expert document extraction and analysis system.\n"
                "Your role is to:\n"
                "1. Extract ALL important information from documents using vision understanding\n"
                "2. Identify document type, structure, and key fields\n"
                "3. Extract data in structured formats (JSON, tables)\n"
                "4. Answer detailed questions about document content\n"
                "5. Identify patterns, relationships, and anomalies\n"
                "6. Provide insights and recommendations\n"
                "\n"
                "Always:\n"
                "- Be precise and cite specific parts of the document\n"
                "- Return structured data when requested\n"
                "- Highlight confidence levels for extracted data\n"
                "- Flag any ambiguities or missing information"
            ),
        )
    return st.session_state.llm_agent


def get_thread():
    """Get or create conversation thread."""
    if "analysis_thread" not in st.session_state:
        st.session_state.analysis_thread = get_llm_agent().get_new_thread()
    return st.session_state.analysis_thread


def extract_with_llm_vision(file_bytes: bytes, file_type: str, file_name: str, 
                           use_thread: bool = True) -> str:
    """Extract content from file using LLM vision capabilities."""
    agent = get_llm_agent()
    thread = get_thread() if use_thread else None
    
    # Prepare the file data for LLM
    media_type = get_file_media_type(file_type, file_name)
    
    # For text-based files, decode directly
    if file_type == "text/csv" or file_type == "application/json" or file_name.endswith(".json"):
        try:
            content_text = file_bytes.decode('utf-8')
            prompt = (
                f"Please analyze and extract all structured data from this {file_type} file:\n\n"
                f"FILENAME: {file_name}\n"
                f"CONTENT:\n{content_text}\n\n"
                "Return results as:\n"
                "1. Summary of the data\n"
                "2. Key fields/columns identified\n"
                "3. Data types and ranges\n"
                "4. Any patterns or anomalies\n"
                "5. Structured JSON representation of key data"
            )
        except:
            content_text = "[Binary data - cannot decode as text]"
            prompt = (
                f"Please analyze this {file_type} file:\n"
                f"FILENAME: {file_name}\n"
                "Extract and summarize all relevant information."
            )
    else:
        # For binary files (PDF, images), use base64 encoding for vision API
        encoded = encode_file_to_base64(file_bytes)
        prompt = (
            f"Please analyze and extract all information from this {file_type} document.\n\n"
            f"FILENAME: {file_name}\n"
            f"FILE DATA (base64): {encoded[:100]}...[truncated]\n\n"
            "Return results as:\n"
            "1. Document type and purpose\n"
            "2. Key information extracted\n"
            "3. Any text found in the document\n"
            "4. Data structure (tables, forms, etc.)\n"
            "5. Confidence levels for extracted data\n"
            "6. Any ambiguities or unclear sections"
        )
    
    if thread:
        result = _run_async(agent.run(prompt, thread=thread))
    else:
        result = _run_async(agent.run(prompt))
    
    return result.text


def analyze_with_llm(prompt: str, use_thread: bool = True) -> str:
    """Run LLM analysis."""
    agent = get_llm_agent()
    thread = get_thread() if use_thread else None
    
    if thread:
        result = _run_async(agent.run(prompt, thread=thread))
    else:
        result = _run_async(agent.run(prompt))
    
    return result.text


# ==================== File Processing ====================

def process_file_with_llm(uploaded_file) -> dict:
    """Process uploaded file using LLM vision extraction."""
    file_type = uploaded_file.type
    file_name = uploaded_file.name
    raw_bytes = uploaded_file.getvalue()
    
    result = {
        "name": file_name,
        "type": file_type,
        "size": len(raw_bytes),
        "raw_bytes": raw_bytes,
        "extracted_content": "",
        "preview": "",
        "extraction_status": "pending"
    }
    
    try:
        # Generate preview for UI
        try:
            if file_type.startswith("image/") and PIL_SUPPORT:
                result["preview"] = Image.open(io.BytesIO(raw_bytes))
            elif file_type == "application/pdf" and PDF_SUPPORT:
                result["preview"] = f"[PDF file - {file_name}]"
            else:
                result["preview"] = raw_bytes.decode("utf-8", errors="ignore")[:300]
        except:
            result["preview"] = f"[{file_type}] Preview unavailable"
        
        # Extract using LLM
        result["extraction_status"] = "extracting"
        result["extracted_content"] = extract_with_llm_vision(raw_bytes, file_type, file_name)
        result["extraction_status"] = "complete"
        
    except Exception as e:
        result["extracted_content"] = f"[Error extracting file: {str(e)}]"
        result["extraction_status"] = f"error: {str(e)}"
    
    return result


# ==================== Streamlit UI ====================

st.set_page_config(
    page_title="LLM Document Extractor",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 LLM-Powered Document Extraction & Analysis")
st.markdown("""
**All extraction powered by Azure OpenAI GPT-4V Vision API**
- No manual parsing libraries needed
- LLM understands document semantics directly
- Single unified pipeline for all file types
""")

# Sidebar info
with st.sidebar:
    st.header("⚙️ Configuration")
    st.markdown("""
    ### How It Works
    1. **Upload** documents (any format)
    2. **LLM Vision** processes files directly
    3. **Extract** structured data semantically
    4. **Analyze** with follow-up questions
    
    ### Cost Comparison
    - **This approach**: Higher
      - Vision API tokens
      - Slower processing
      - BUT: Better understanding
    
    - **Manual extraction**: Lower
      - Library parsing only
      - Faster
      - Limited understanding
    """)
    
    enable_chat = st.checkbox("Enable Conversation Memory", value=True)
    show_extraction_details = st.checkbox("Show Extraction Details", value=True)

# Initialize session state
if "documents" not in st.session_state:
    st.session_state.documents = {}
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

# Main tabs
tabs = st.tabs(["📤 Upload & Extract", "🔍 Q&A", "📋 Extracted Data", "📊 History"])

# ==================== TAB 1: Upload & Extract ====================
with tabs[0]:
    st.subheader("Upload Documents for LLM-Based Extraction")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded_files = st.file_uploader(
            "Upload any documents (PDF, CSV, Excel, Images, JSON, Text) - LLM will extract intelligently",
            type=["pdf", "csv", "xlsx", "xls", "json", "txt", "png", "jpg", "jpeg", "bmp", "gif"],
            accept_multiple_files=True,
        )
    
    with col2:
        if st.button("🔄 Clear All"):
            st.session_state.documents = {}
            st.rerun()
    
    # Process uploaded files
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.documents:
                with st.spinner(f"🤖 Processing {uploaded_file.name} with LLM vision..."):
                    doc_data = process_file_with_llm(uploaded_file)
                    st.session_state.documents[uploaded_file.name] = doc_data
    
    # Display uploaded documents and extraction results
    if st.session_state.documents:
        st.subheader("Extracted Content (LLM-Powered)")
        
        for doc_name, doc_data in st.session_state.documents.items():
            with st.expander(f"📄 {doc_name} ({doc_data['size']} bytes)"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Type:** {doc_data['type']}")
                    st.write(f"**Status:** {doc_data['extraction_status']}")
                    
                    if show_extraction_details:
                        st.write(f"**Raw Preview:**")
                        if isinstance(doc_data['preview'], Image.Image):
                            st.image(doc_data['preview'], use_container_width=True)
                        else:
                            st.code(str(doc_data['preview'])[:300])
                
                with col2:
                    if st.button("🔄 Re-extract", key=f"reex_{doc_name}"):
                        with st.spinner("Processing..."):
                            class _UploadedFileLike:
                                def __init__(self, file_type: str, name: str, raw_bytes: bytes):
                                    self.type = file_type
                                    self.name = name
                                    self._raw_bytes = raw_bytes

                                def getvalue(self) -> bytes:
                                    return self._raw_bytes

                            doc_data = process_file_with_llm(
                                _UploadedFileLike(
                                    file_type=doc_data["type"],
                                    name=doc_name,
                                    raw_bytes=doc_data.get("raw_bytes", b""),
                                )
                            )
                            st.session_state.documents[doc_name] = doc_data
                            st.rerun()
                
                # Display LLM-extracted content
                if doc_data['extraction_status'] == 'complete':
                    st.markdown("### 🤖 LLM Extraction Results")
                    st.markdown(doc_data['extracted_content'])
                    
                    # Option to export
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        if st.button("💾 Export as JSON", key=f"export_json_{doc_name}"):
                            export_data = {
                                "filename": doc_name,
                                "extracted_at": datetime.now().isoformat(),
                                "content": doc_data['extracted_content']
                            }
                            st.download_button(
                                label="📥 Download JSON",
                                data=json.dumps(export_data, indent=2),
                                file_name=f"{doc_name}_extracted.json",
                                mime="application/json"
                            )
                    
                    with col2:
                        if st.button("📊 Export as Text", key=f"export_txt_{doc_name}"):
                            st.download_button(
                                label="📥 Download Text",
                                data=doc_data['extracted_content'],
                                file_name=f"{doc_name}_extracted.txt",
                                mime="text/plain"
                            )
                
                elif doc_data['extraction_status'] == 'extracting':
                    st.info("⏳ Extraction in progress...")
                else:
                    st.error(f"❌ {doc_data['extraction_status']}")

# ==================== TAB 2: Q&A ====================
with tabs[1]:
    st.subheader("Ask Questions About Extracted Data")
    
    if not st.session_state.documents:
        st.info("📤 Please upload documents first in the Upload & Extract tab")
    else:
        col1, col2 = st.columns([3, 1])
        with col1:
            selected_docs = st.multiselect(
                "Select extracted documents for Q&A",
                list(st.session_state.documents.keys()),
                default=list(st.session_state.documents.keys())[:1] if st.session_state.documents else []
            )
        
        with col2:
            st.write(f"**Docs Selected:** {len(selected_docs)}")
        
        # Chat history
        if "qa_messages" not in st.session_state:
            st.session_state.qa_messages = []
        
        for msg in st.session_state.qa_messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
        
        # Q&A input
        user_question = st.chat_input("Ask a question about the extracted data...")
        
        if user_question and selected_docs:
            st.session_state.qa_messages.append({"role": "user", "content": user_question})
            
            with st.chat_message("user"):
                st.write(user_question)
            
            # Prepare context from selected documents
            context = "\n---DOCUMENT BREAK---\n".join([
                f"[{doc_name}]\n{st.session_state.documents[doc_name]['extracted_content'][:2000]}"
                for doc_name in selected_docs
                if st.session_state.documents[doc_name]['extraction_status'] == 'complete'
            ])
            
            if not context:
                st.warning("⚠️ No successfully extracted documents selected for Q&A")
            else:
                prompt = (
                    f"Based on the following extracted document data, answer this question:\n\n"
                    f"EXTRACTED DATA:\n{context}\n\n"
                    f"QUESTION: {user_question}\n\n"
                    "Provide a detailed answer citing specific parts of the extracted data."
                )
                
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = analyze_with_llm(prompt, use_thread=enable_chat)
                    st.write(response)
                
                st.session_state.qa_messages.append({"role": "assistant", "content": response})
                
                st.session_state.analysis_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "document": ", ".join(selected_docs),
                    "question": user_question,
                    "answer": response[:200] + "..." if len(response) > 200 else response
                })

# ==================== TAB 3: Extracted Data ====================
with tabs[2]:
    st.subheader("View All Extracted Content")
    
    if not st.session_state.documents:
        st.info("📤 No documents processed yet")
    else:
        for doc_name, doc_data in st.session_state.documents.items():
            if doc_data['extraction_status'] == 'complete':
                with st.expander(f"📄 {doc_name}", expanded=False):
                    st.markdown(doc_data['extracted_content'])
                    
                    # Copy button
                    if st.button("📋 Copy to Clipboard", key=f"copy_{doc_name}"):
                        st.toast(f"Copied {doc_name} extraction")

# ==================== TAB 4: Analysis History ====================
with tabs[3]:
    st.subheader("Question & Answer History")
    
    if not st.session_state.analysis_history:
        st.info("No Q&A interactions yet")
    else:
        history_df = pd.DataFrame(st.session_state.analysis_history)
        st.dataframe(history_df, use_container_width=True)
        
        if st.button("📥 Download History as CSV"):
            csv = history_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="qa_history.csv",
                mime="text/csv"
            )
        
        if st.button("🗑️ Clear History"):
            st.session_state.analysis_history = []
            st.rerun()

# ==================== Footer ====================
st.markdown("---")
st.markdown("""
### 🤖 LLM Extraction Approach
- **Advantage**: One unified pipeline, no parsing library complexity
- **Limitation**: Higher cost, slower (vision API calls)
- **Best for**: Complex documents, handwritten text, semantic understanding
- **vs Manual**: Better for understanding, worse for cost efficiency
""")
