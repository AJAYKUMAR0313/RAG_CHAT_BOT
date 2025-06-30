import streamlit as st
import os
import tempfile
from typing import List, Dict, Any
import pandas as pd
from utils.pdf_processor import PDFProcessor
from utils.vector_store import VectorStore
from utils.chat_handler import ChatHandler
from dotenv import load_dotenv
load_dotenv()
import os


# Configure page
st.set_page_config(
    page_title="Paves Technologies AI Assistant",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_handler" not in st.session_state:
    st.session_state.chat_handler = None
if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = []

def initialize_components():
    """Initialize the vector store and chat handler"""
    try:
        if st.session_state.vector_store is None:
            st.session_state.vector_store = VectorStore()
        
        if st.session_state.chat_handler is None:
            groq_api_key = os.getenv("GROQ_API_KEY", "")
            if not groq_api_key:
                st.error("⚠️ GROQ_API_KEY environment variable not found. Please set your Groq API key.")
                st.stop()
            st.session_state.chat_handler = ChatHandler(groq_api_key)
    except Exception as e:
        st.error(f"Error initializing components: {str(e)}")
        st.stop()

def process_uploaded_files(uploaded_files):
    """Process uploaded PDF files and add to vector store"""
    if not uploaded_files:
        return
    
    pdf_processor = PDFProcessor()
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, uploaded_file in enumerate(uploaded_files):
        if uploaded_file.name in st.session_state.documents_processed:
            st.warning(f"📄 {uploaded_file.name} already processed. Skipping...")
            continue
            
        status_text.text(f"Processing {uploaded_file.name}...")
        
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Extract text from PDF
            text_chunks = pdf_processor.extract_text_chunks(tmp_file_path, uploaded_file.name)
            
            if text_chunks:
                # Add to vector store
                st.session_state.vector_store.add_documents(text_chunks)
                st.session_state.documents_processed.append(uploaded_file.name)
                st.success(f"✅ Successfully processed {uploaded_file.name} ({len(text_chunks)} chunks)")
            else:
                st.warning(f"⚠️ No text extracted from {uploaded_file.name}")
            
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
        except Exception as e:
            st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
        
        progress_bar.progress((i + 1) / len(uploaded_files))
    
    status_text.empty()
    progress_bar.empty()

def main():
    # Initialize components
    initialize_components()
    
    # Header with company branding
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image("assets/paves_logo.svg", width=80)
    with col2:
        st.title("🏗️ Paves Technologies AI Assistant")
        st.markdown("*Your intelligent companion for company documentation and knowledge*")
    
    # Sidebar for document management
    with st.sidebar:
        st.header("📚 Document Management")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload PDF Documents",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload company documents, manuals, or reports to enhance the AI's knowledge base"
        )
        
        if uploaded_files:
            if st.button("🔄 Process Documents", type="primary"):
                process_uploaded_files(uploaded_files)
        
        st.divider()
        
        # Document status
        st.subheader("📋 Processed Documents")
        if st.session_state.documents_processed:
            for doc in st.session_state.documents_processed:
                st.text(f"✅ {doc}")
        else:
            st.info("No documents processed yet")
        
        # Clear documents
        if st.session_state.documents_processed:
            if st.button("🗑️ Clear All Documents", type="secondary"):
                st.session_state.vector_store = VectorStore()
                st.session_state.documents_processed = []
                st.session_state.messages = []
                st.rerun()
        
        st.divider()
        
        # Company information
        st.markdown("""
        ### 🏢 About Paves Technologies
        Your AI assistant is specialized in Paves Technologies' 
        operations, projects, and documentation. Upload relevant 
        documents to enhance response accuracy.
        """)
    
    # Main chat interface
    st.header("💬 Chat with Your Documents")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("📖 Sources"):
                    for source in message["sources"]:
                        st.text(f"• {source}")
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about Paves Technologies..."):
        # Check if documents are loaded
        # if not st.session_state.documents_processed:
        #     st.warning("⚠️ Please upload and process some documents first to get meaningful responses.")
        #     return
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Get relevant documents
                    relevant_docs = st.session_state.vector_store.similarity_search(prompt, k=5)
                    
                    # Generate response
                    response, sources = st.session_state.chat_handler.generate_response(
                        prompt, relevant_docs
                    )
                    
                    # Display response
                    st.markdown(response)
                    
                    # Display sources
                    if sources:
                        with st.expander("📖 Sources"):
                            for source in sources:
                                st.text(f"• {source}")
                    
                    # Add to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response,
                        "sources": sources
                    })
                    
                except Exception as e:
                    error_msg = f"❌ Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_msg
                    })

if __name__ == "__main__":
    main()
