"""
RAG Document Manager for Structural Crack Analysis
Handles document storage, embedding, and retrieval
"""
import os
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from pathlib import Path

try:
    from .crack_types import CrackType, CrackCharacteristics
except ImportError:
    from crack_types import CrackType, CrackCharacteristics


class CrackRAGDocumentManager:
    """Manages documents and embeddings for crack analysis RAG system"""
    
    def __init__(self, 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 data_dir: str = "./data"):
        """
        Initialize the document manager
        
        Args:
            embedding_model: Name of the sentence transformer model
            data_dir: Directory to store embeddings and documents
        """
        self.embedding_model = SentenceTransformer(embedding_model)
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Storage for documents and metadata
        self.documents: List[str] = []
        self.document_metadata: List[Dict] = []
        self.index: Optional[faiss.Index] = None
        
        # File paths for persistence
        self.documents_file = self.data_dir / "documents.json"
        self.metadata_file = self.data_dir / "metadata.json"
        self.index_file = self.data_dir / "faiss_index.pkl"
        
        # Load existing data if available
        self.load_data()
    
    def add_document(self, 
                    content: str, 
                    crack_type: Optional[CrackType] = None,
                    source: str = "manual",
                    title: str = "",
                    **metadata) -> None:
        """
        Add a single document to the knowledge base
        
        Args:
            content: The document content
            crack_type: Associated crack type (optional)
            source: Source of the document
            title: Document title
            **metadata: Additional metadata
        """
        # Prepare metadata
        doc_metadata = {
            "title": title,
            "source": source,
            "crack_type": crack_type.value if crack_type else None,
            "doc_id": len(self.documents),
            **metadata
        }
        
        # Add to storage
        self.documents.append(content)
        self.document_metadata.append(doc_metadata)
        
        # Generate embedding and update index
        self._update_index([content])
        
        print(f"Added document: {title or content[:50]}...")
    
    def add_documents_batch(self, 
                           documents: List[Dict]) -> None:
        """
        Add multiple documents in batch
        
        Args:
            documents: List of document dictionaries with 'content' and optional metadata
        """
        contents = []
        
        for doc in documents:
            content = doc.get("content", "")
            crack_type_str = doc.get("crack_type")
            crack_type = CrackType(crack_type_str) if crack_type_str else None
            
            doc_metadata = {
                "title": doc.get("title", ""),
                "source": doc.get("source", "batch"),
                "crack_type": crack_type.value if crack_type else None,
                "doc_id": len(self.documents) + len(contents),
                **{k: v for k, v in doc.items() if k not in ["content", "title", "source", "crack_type"]}
            }
            
            self.documents.append(content)
            self.document_metadata.append(doc_metadata)
            contents.append(content)
        
        # Update index with all new documents
        self._update_index(contents)
        
        print(f"Added {len(documents)} documents in batch")
    
    def _update_index(self, new_contents: List[str]) -> None:
        """Update FAISS index with new document embeddings"""
        if not new_contents:
            return
            
        # Generate embeddings
        embeddings = self.embedding_model.encode(new_contents)
        embeddings = embeddings.astype('float32')
        
        # Create or update index
        if self.index is None:
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
    
    def retrieve_documents(self, 
                          query: str, 
                          k: int = 5,
                          crack_type_filter: Optional[CrackType] = None) -> Tuple[List[str], List[Dict], List[float]]:
        """
        Retrieve most relevant documents for a query
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            crack_type_filter: Filter by specific crack type
            
        Returns:
            Tuple of (documents, metadata, scores)
        """
        if self.index is None or len(self.documents) == 0:
            return [], [], []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query]).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search in index
        scores, indices = self.index.search(query_embedding, min(k * 2, len(self.documents)))
        
        # Filter results
        filtered_docs = []
        filtered_metadata = []
        filtered_scores = []
        
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for invalid indices
                continue
                
            metadata = self.document_metadata[idx]
            
            # Apply crack type filter if specified
            if crack_type_filter and metadata.get("crack_type") != crack_type_filter.value:
                continue
            
            filtered_docs.append(self.documents[idx])
            filtered_metadata.append(metadata)
            filtered_scores.append(float(score))
            
            if len(filtered_docs) >= k:
                break
        
        return filtered_docs, filtered_metadata, filtered_scores
    
    def get_crack_specific_context(self, crack_type: CrackType, query: str = "") -> str:
        """
        Get context specific to a crack type
        
        Args:
            crack_type: The crack type to get context for
            query: Additional query for context refinement
            
        Returns:
            Formatted context string
        """
        # Get crack characteristics
        crack_info = CrackCharacteristics.get_crack_info(crack_type)
        
        # Build context
        context_parts = [
            f"Crack Type: {crack_type.value.title()}",
            f"Description: {crack_info.get('description', 'N/A')}",
            f"Common Locations: {', '.join(crack_info.get('common_locations', []))}",
            f"Typical Causes: {', '.join(crack_info.get('typical_causes', []))}"
        ]
        
        # Add retrieved documents if query provided
        if query:
            docs, metadata, scores = self.retrieve_documents(
                query, k=3, crack_type_filter=crack_type
            )
            if docs:
                context_parts.append("\\nRelevant Documentation:")
                for i, (doc, meta) in enumerate(zip(docs, metadata)):
                    context_parts.append(f"{i+1}. {doc[:200]}...")
        
        return "\\n".join(context_parts)
    
    def save_data(self) -> None:
        """Save documents, metadata, and index to disk"""
        # Save documents and metadata
        with open(self.documents_file, 'w') as f:
            json.dump(self.documents, f, indent=2)
        
        with open(self.metadata_file, 'w') as f:
            json.dump(self.document_metadata, f, indent=2, default=str)
        
        # Save FAISS index
        if self.index is not None:
            with open(self.index_file, 'wb') as f:
                pickle.dump(faiss.serialize_index(self.index), f)
        
        print(f"Saved {len(self.documents)} documents to {self.data_dir}")
    
    def load_data(self) -> None:
        """Load documents, metadata, and index from disk"""
        try:
            # Load documents
            if self.documents_file.exists():
                with open(self.documents_file, 'r') as f:
                    self.documents = json.load(f)
            
            # Load metadata
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    self.document_metadata = json.load(f)
            
            # Load FAISS index
            if self.index_file.exists():
                with open(self.index_file, 'rb') as f:
                    index_data = pickle.load(f)
                    self.index = faiss.deserialize_index(index_data)
            
            if self.documents:
                print(f"Loaded {len(self.documents)} documents from {self.data_dir}")
                
        except Exception as e:
            print(f"Error loading data: {e}")
            self.documents = []
            self.document_metadata = []
            self.index = None
    
    def get_stats(self) -> Dict:
        """Get statistics about the document collection"""
        crack_type_counts = {}
        for metadata in self.document_metadata:
            crack_type = metadata.get("crack_type", "unknown")
            crack_type_counts[crack_type] = crack_type_counts.get(crack_type, 0) + 1
        
        return {
            "total_documents": len(self.documents),
            "crack_type_distribution": crack_type_counts,
            "index_size": self.index.ntotal if self.index else 0
        }
