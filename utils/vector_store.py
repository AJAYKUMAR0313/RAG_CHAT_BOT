import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Any, Tuple
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class VectorStore:
    def __init__(self, storage_path: str = "data/vector_store", max_features: int = 5000):
        """
        Initialize FAISS vector store with TF-IDF vectorization
        
        Args:
            storage_path: Path to store vector index and metadata
            max_features: Maximum number of features for TF-IDF vectorizer
        """
        self.storage_path = storage_path
        self.max_features = max_features
        self.index_path = os.path.join(storage_path, "faiss_index.bin")
        self.metadata_path = os.path.join(storage_path, "metadata.pkl")
        self.vectorizer_path = os.path.join(storage_path, "vectorizer.pkl")
        self.logger = logging.getLogger(__name__)
        
        # Create storage directory if it doesn't exist
        os.makedirs(storage_path, exist_ok=True)
        
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        
        # Initialize variables
        self.embedding_dim = max_features
        self.index = None
        self.metadata = []
        self.is_fitted = False
        
        # Load existing index if available
        self.load_index()
    
    def save_index(self):
        """Save FAISS index, vectorizer and metadata to disk"""
        try:
            if self.index is not None:
                faiss.write_index(self.index, self.index_path)
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)
            with open(self.vectorizer_path, 'wb') as f:
                pickle.dump(self.vectorizer, f)
            self.logger.info(f"Vector store saved to {self.storage_path}")
        except Exception as e:
            self.logger.error(f"Error saving vector store: {str(e)}")
    
    def load_index(self):
        """Load FAISS index, vectorizer and metadata from disk"""
        try:
            if (os.path.exists(self.index_path) and 
                os.path.exists(self.metadata_path) and 
                os.path.exists(self.vectorizer_path)):
                
                self.index = faiss.read_index(self.index_path)
                with open(self.metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
                with open(self.vectorizer_path, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                self.is_fitted = True
                if self.index:
                    self.embedding_dim = self.index.d
                self.logger.info(f"Loaded {len(self.metadata)} documents from vector store")
            else:
                self.logger.info("No existing vector store found, starting fresh")
        except Exception as e:
            self.logger.error(f"Error loading vector store: {str(e)}")
            # Reset to empty state if loading fails
            self.index = None
            self.metadata = []
            self.is_fitted = False
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        Add documents to the vector store
        
        Args:
            documents: List of document dictionaries with 'text' and metadata
        """
        if not documents:
            return
        
        try:
            # Extract texts for embedding
            texts = [doc['text'] for doc in documents]
            
            # Combine with existing texts if any
            all_texts = []
            if self.metadata:
                all_texts.extend([doc['text'] for doc in self.metadata])
            all_texts.extend(texts)
            
            # Fit/refit the vectorizer on all texts
            tfidf_matrix = self.vectorizer.fit_transform(all_texts)
            self.is_fitted = True
            
            # Update embedding dimension
            self.embedding_dim = tfidf_matrix.shape[1]
            
            # Create new FAISS index with correct dimensions
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            
            # Add all vectors to FAISS index
            vectors = tfidf_matrix.toarray().astype(np.float32)
            self.index.add(vectors)
            
            # Store metadata
            self.metadata.extend(documents)
            
            # Save to disk
            self.save_index()
            
            self.logger.info(f"Added {len(documents)} documents to vector store")
            
        except Exception as e:
            self.logger.error(f"Error adding documents to vector store: {str(e)}")
            raise
    
    def similarity_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents
        
        Args:
            query: Search query
            k: Number of top results to return
            
        Returns:
            List of similar documents with metadata and similarity scores
        """
        if not query.strip():
            return []
        
        if not self.is_fitted or self.index is None:
            self.logger.warning("Vector store is empty or not fitted")
            return []
        
        if self.index.ntotal == 0:
            self.logger.warning("Vector store is empty")
            return []
        
        try:
            # Generate query embedding using the fitted vectorizer
            query_tfidf = self.vectorizer.transform([query])
            query_vector = query_tfidf.toarray().astype(np.float32)
            
            # Ensure k doesn't exceed available documents
            k = min(k, self.index.ntotal)
            
            # Search FAISS index
            distances, indices = self.index.search(query_vector, k)
            
            # Prepare results
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.metadata):  # Ensure index is valid
                    doc = self.metadata[idx].copy()
                    # Convert L2 distance to similarity score (closer to 1 is more similar)
                    doc['similarity_score'] = float(1 / (1 + distance))
                    doc['rank'] = i + 1
                    results.append(doc)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error during similarity search: {str(e)}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        return {
            'total_documents': self.index.ntotal if self.index else 0,
            'embedding_dimension': self.embedding_dim,
            'vectorizer_type': 'TF-IDF',
            'unique_sources': len(set(doc.get('source', 'unknown') for doc in self.metadata)),
            'is_fitted': self.is_fitted
        }
    
    def clear(self):
        """Clear all documents from the vector store"""
        self.index = None
        self.metadata = []
        self.is_fitted = False
        
        # Reset vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        
        # Remove saved files
        for path in [self.index_path, self.metadata_path, self.vectorizer_path]:
            if os.path.exists(path):
                os.remove(path)
        
        self.logger.info("Vector store cleared")