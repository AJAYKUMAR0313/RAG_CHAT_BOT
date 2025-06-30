import PyPDF2
import pdfplumber
from typing import List, Dict, Any
import re
import logging

class PDFProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize PDF processor with chunking parameters
        
        Args:
            chunk_size: Maximum size of each text chunk
            chunk_overlap: Overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.logger = logging.getLogger(__name__)
    
    def extract_text_with_pypdf2(self, pdf_path: str) -> str:
        """Extract text using PyPDF2"""
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            self.logger.error(f"Error extracting text with PyPDF2: {str(e)}")
            return ""
    
    def extract_text_with_pdfplumber(self, pdf_path: str) -> str:
        """Extract text using pdfplumber (more accurate for complex layouts)"""
        try:
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text
        except Exception as e:
            self.logger.error(f"Error extracting text with pdfplumber: {str(e)}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters that might interfere with processing
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\'\/\\]', '', text)
        
        # Normalize line breaks
        text = text.replace('\n', ' ').replace('\r', ' ')
        
        return text.strip()
    
    def create_chunks(self, text: str, source: str) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Input text to chunk
            source: Source document name
            
        Returns:
            List of text chunks with metadata
        """
        if not text:
            return []
        
        chunks = []
        words = text.split()
        
        # Calculate words per chunk based on average word length
        avg_word_length = 5  # Approximate average word length
        words_per_chunk = self.chunk_size // avg_word_length
        overlap_words = self.chunk_overlap // avg_word_length
        
        start = 0
        chunk_id = 0
        
        while start < len(words):
            end = min(start + words_per_chunk, len(words))
            chunk_text = ' '.join(words[start:end])
            
            if chunk_text.strip():  # Only add non-empty chunks
                chunks.append({
                    'text': chunk_text,
                    'source': source,
                    'chunk_id': chunk_id,
                    'start_word': start,
                    'end_word': end
                })
                chunk_id += 1
            
            # Move start position with overlap
            start = max(start + words_per_chunk - overlap_words, start + 1)
            
            # Break if we're not making progress
            if start >= end:
                break
        
        return chunks
    
    def extract_text_chunks(self, pdf_path: str, filename: str) -> List[Dict[str, Any]]:
        """
        Main method to extract text chunks from PDF
        
        Args:
            pdf_path: Path to PDF file
            filename: Original filename for metadata
            
        Returns:
            List of text chunks with metadata
        """
        # Try pdfplumber first (better accuracy)
        text = self.extract_text_with_pdfplumber(pdf_path)
        
        # Fallback to PyPDF2 if pdfplumber fails
        if not text or len(text.strip()) < 50:
            text = self.extract_text_with_pypdf2(pdf_path)
        
        if not text or len(text.strip()) < 50:
            self.logger.warning(f"Very little or no text extracted from {filename}")
            return []
        
        # Clean and chunk the text
        cleaned_text = self.clean_text(text)
        chunks = self.create_chunks(cleaned_text, filename)
        
        self.logger.info(f"Extracted {len(chunks)} chunks from {filename}")
        return chunks
