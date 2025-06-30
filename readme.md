# Paves Technologies AI Assistant

## Overview

This is a Streamlit-based AI assistant application designed specifically for Paves Technologies, a construction and infrastructure company. The application enables users to chat with company documents through PDF upload and processing, utilizing RAG (Retrieval-Augmented Generation) architecture to provide context-aware responses about company operations, projects, and services.

## System Architecture

The application follows a modular architecture with the following key design principles:

**Problem**: Need for an intelligent document query system for construction company documentation
**Solution**: RAG-based chat interface with PDF processing capabilities
**Architecture Choice**: Streamlit frontend with modular backend services for scalability and maintainability

### Core Components:
- **Frontend**: Streamlit web interface for user interaction
- **Document Processing**: PDF text extraction and chunking
- **Vector Storage**: FAISS-based similarity search with sentence transformers
- **Chat Interface**: Groq API integration for LLM responses
- **Session Management**: Streamlit session state for conversation continuity

## Key Components

### 1. Main Application (`app.py`)
- **Purpose**: Entry point and UI orchestration
- **Responsibilities**: Page configuration, session state management, component initialization
- **Key Features**: Wide layout, sidebar navigation, progress tracking for file uploads

### 2. PDF Processor (`utils/pdf_processor.py`)
- **Purpose**: Extract and process text from PDF documents
- **Technologies**: PyPDF2 and pdfplumber for dual extraction methods
- **Features**: 
  - Fallback extraction strategy (pdfplumber as primary, PyPDF2 as backup)
  - Text cleaning and normalization
  - Configurable chunking (1000 chars with 200 char overlap)

### 3. Vector Store (`utils/vector_store.py`)
- **Purpose**: Semantic search capability using vector embeddings
- **Technology Stack**: FAISS for indexing, SentenceTransformers for embeddings
- **Model**: all-MiniLM-L6-v2 (lightweight, efficient for semantic similarity)
- **Persistence**: Local file storage with pickle serialization
- **Storage Location**: `data/vector_store/` directory

### 4. Chat Handler (`utils/chat_handler.py`)
- **Purpose**: LLM integration and conversation management
- **API Provider**: Groq (using mixtral-8x7b-32768 model)
- **Features**: 
  - Specialized system prompt for construction industry context
  - Context formatting from retrieved documents
  - Professional tone and industry terminology

## Data Flow

1. **Document Upload**: User uploads PDF files through Streamlit interface
2. **Text Extraction**: PDFProcessor extracts and cleans text using dual-method approach
3. **Chunking**: Text split into overlapping chunks for better retrieval
4. **Embedding**: Chunks converted to vectors using SentenceTransformer
5. **Storage**: Vectors stored in FAISS index with metadata persistence
6. **Query Processing**: User questions converted to embeddings for similarity search
7. **Context Retrieval**: Relevant document chunks retrieved based on similarity
8. **Response Generation**: Groq API generates responses using retrieved context
9. **Display**: Conversation displayed through Streamlit chat interface

## External Dependencies

### Core Libraries:
- **Streamlit**: Web application framework
- **FAISS**: Vector similarity search
- **SentenceTransformers**: Text embedding generation
- **Groq**: LLM API for chat completion
- **PyPDF2 & pdfplumber**: PDF text extraction
- **pandas**: Data manipulation
- **numpy**: Numerical operations

### API Requirements:
- **Groq API Key**: Required environment variable `GROQ_API_KEY`

### Model Dependencies:
- **all-MiniLM-L6-v2**: Sentence transformer model (auto-downloaded)
- **mixtral-8x7b-32768**: Groq's LLM model

## Deployment Strategy

### Environment Setup:
- Python environment with required dependencies
- Groq API key configuration
- Local storage directory creation for vector persistence

### File Structure:
```
/
├── app.py                 # Main application
├── utils/
│   ├── pdf_processor.py   # PDF handling
│   ├── vector_store.py    # Vector operations
│   └── chat_handler.py    # Chat logic
└── data/
    └── vector_store/      # Persistent storage
        ├── faiss_index.bin
        └── metadata.pkl
```

### Scalability Considerations:
- Modular architecture allows easy component replacement
- FAISS supports scaling to larger document collections
- Session state management handles multiple concurrent users
- Local storage can be replaced with cloud solutions

## Changelog

- June 29, 2025. Initial setup

## User Preferences

Preferred communication style: Simple, everyday language.