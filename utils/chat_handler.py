import os
from typing import List, Dict, Any, Tuple
import json
import logging
from groq import Groq

class ChatHandler:
    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        """
        Initialize chat handler with Groq API
        
        Args:
            api_key: Groq API key
            model: Model name to use for chat completion
        """
        self.api_key = api_key
        self.model = model
        self.logger = logging.getLogger(__name__)
        
        try:
            self.client = Groq(api_key=api_key)
        except Exception as e:
            self.logger.error(f"Error initializing Groq client: {str(e)}")
            raise
    
    def create_system_prompt(self) -> str:
        """Create system prompt for Paves Technologies context"""
        return """You are an AI assistant specialized in Paves Technologies, a leading construction and infrastructure company. 

Your role is to:
1. Answer questions about Paves Technologies' operations, projects, services, and documentation
2. Provide accurate, professional, and helpful responses based on the provided context
3. Use industry-appropriate terminology and maintain a professional tone
4. When referencing information, be specific about which documents or sections you're drawing from
5. If you cannot find relevant information in the provided context, clearly state this limitation

Company Context:
- Paves Technologies specializes in construction, infrastructure development, and engineering services
- Focus on innovative solutions, quality delivery, and sustainable practices
- Serve both public and private sector clients
- Emphasize safety, efficiency, and technical excellence

Guidelines:
- Always prioritize accuracy over completeness
- Cite specific sources when available
- Use professional construction and engineering terminology
- Be concise but comprehensive in your responses
- If asked about information not in the provided documents, clearly state the limitation"""
    
    def format_context(self, relevant_docs: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents into context for the LLM
        
        Args:
            relevant_docs: List of relevant document chunks
            
        Returns:
            Formatted context string
        """
        if not relevant_docs:
            return "No relevant documents found."
        
        context_parts = []
        for i, doc in enumerate(relevant_docs, 1):
            source = doc.get('source', 'Unknown Source')
            text = doc.get('text', '')
            similarity = doc.get('similarity_score', 0)
            
            context_parts.append(f"""
Document {i} (Source: {source}, Relevance: {similarity:.3f}):
{text}
---
""")
        
        return "\n".join(context_parts)
    
    def generate_response(self, query: str, relevant_docs: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
        """
        Generate response using Groq API with RAG context
        
        Args:
            query: User query
            relevant_docs: List of relevant document chunks
            
        Returns:
            Tuple of (response, sources)
        """
        try:
            # Format context from retrieved documents
            context = self.format_context(relevant_docs)
            
            # Create messages for chat completion
            messages = [
                {
                    "role": "system",
                    "content": self.create_system_prompt()
                },
                {
                    "role": "user",
                    "content": f"""Based on the following context from Paves Technologies documents, please answer this question: {query}

Context:
{context}

Instructions:
- Use only the information provided in the context above
- If the context doesn't contain enough information to answer the question, say so clearly
- Cite the specific documents you're referencing
- Maintain a professional tone appropriate for Paves Technologies
- Be concise but thorough"""
                }
            ]
            
            # Call Groq API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,  # Lower temperature for more consistent responses
                max_tokens=1024,
                top_p=0.9
            )
            
            # Extract response content
            answer = response.choices[0].message.content.strip()
            
            # Extract unique sources from relevant documents
            sources = list(set(doc.get('source', 'Unknown') for doc in relevant_docs))
            
            return answer, sources
            
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            error_response = f"I apologize, but I encountered an error while processing your request: {str(e)}. Please try again or contact support if the issue persists."
            return error_response, []
    
    def validate_api_key(self) -> bool:
        """
        Validate if the Groq API key is working
        
        Returns:
            True if API key is valid, False otherwise
        """
        try:
            # Make a simple test call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            return True
        except Exception as e:
            self.logger.error(f"API key validation failed: {str(e)}")
            return False
