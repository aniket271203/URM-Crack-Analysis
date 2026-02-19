"""
Enhanced Retrieval Mechanisms for Crack Analysis RAG
Implements hybrid search, re-ranking, and query expansion
"""
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

try:
    from .crack_types import CrackType
except ImportError:
    from crack_types import CrackType


@dataclass
class SearchResult:
    """Represents a search result with content and metadata"""
    content: str
    metadata: Dict
    score: float
    rank: int = 0


class EnhancedRetriever:
    """Enhanced retrieval with query expansion and re-ranking"""
    
    # Domain-specific query expansion mappings
    QUERY_EXPANSIONS = {
        "diagonal": ["diagonal", "shear", "settlement", "inclined", "slant", "angular"],
        "vertical": ["vertical", "longitudinal", "up-down", "thermal", "shrinkage"],
        "horizontal": ["horizontal", "lateral", "transverse", "flexural", "bending"],
        "step": ["step", "stepped", "stair", "staircase", "zigzag", "masonry joint"],
        "x-shaped": ["x-shaped", "cross", "diagonal pair", "seismic", "shear cross"],
        
        "crack": ["crack", "fissure", "fracture", "split", "break", "opening"],
        "cause": ["cause", "reason", "origin", "source", "mechanism", "etiology"],
        "repair": ["repair", "fix", "remediation", "restoration", "treatment", "correction"],
        "inspection": ["inspection", "assessment", "evaluation", "examination", "survey"],
        "monitoring": ["monitoring", "tracking", "observation", "surveillance", "watching"],
        
        "masonry": ["masonry", "brick", "block", "mortar", "stone", "CMU"],
        "concrete": ["concrete", "cement", "reinforced concrete", "prestressed"],
        "wall": ["wall", "partition", "panel", "facade"],
        "beam": ["beam", "girder", "lintel", "joist"],
        "foundation": ["foundation", "footing", "basement", "substructure"],
    }
    
    # Keywords indicating structural importance
    STRUCTURAL_KEYWORDS = [
        "structural", "load", "bearing", "failure", "collapse", "safety",
        "critical", "dangerous", "severe", "major", "significant"
    ]
    
    # Keywords indicating repair/maintenance
    MAINTENANCE_KEYWORDS = [
        "repair", "fix", "remediation", "treatment", "maintenance",
        "restoration", "epoxy", "injection", "seal", "patch"
    ]
    
    def expand_query(self, query: str) -> List[str]:
        """
        Expand query with domain-specific synonyms
        
        Args:
            query: Original query string
            
        Returns:
            List of expanded query terms
        """
        query_lower = query.lower()
        expanded_terms = set(query.split())
        
        # Add expansions for matched keywords
        for keyword, expansions in self.QUERY_EXPANSIONS.items():
            if keyword in query_lower:
                expanded_terms.update(expansions)
        
        return list(expanded_terms)
    
    def extract_crack_type_from_query(self, query: str) -> Optional[CrackType]:
        """
        Extract crack type from query if mentioned
        
        Args:
            query: Search query
            
        Returns:
            CrackType if found, None otherwise
        """
        query_lower = query.lower()
        
        for crack_type in CrackType:
            if crack_type.value in query_lower:
                return crack_type
            
            # Check for variations
            if crack_type == CrackType.X_SHAPED and ("x shaped" in query_lower or "cross" in query_lower):
                return crack_type
            if crack_type == CrackType.STEP and "stepped" in query_lower:
                return crack_type
        
        return None
    
    def score_relevance(self, result: SearchResult, query: str, expanded_terms: List[str]) -> float:
        """
        Calculate enhanced relevance score
        
        Args:
            result: Search result to score
            query: Original query
            expanded_terms: Expanded query terms
            
        Returns:
            Enhanced relevance score
        """
        content_lower = result.content.lower()
        score = result.score  # Start with embedding similarity score
        
        # Boost for exact query match
        if query.lower() in content_lower:
            score *= 1.3
        
        # Boost for multiple expanded term matches
        term_matches = sum(1 for term in expanded_terms if term in content_lower)
        score *= (1.0 + 0.1 * term_matches)
        
        # Boost for structural importance keywords
        if any(kw in content_lower for kw in self.STRUCTURAL_KEYWORDS):
            score *= 1.2
        
        # Boost for crack type match
        query_crack_type = self.extract_crack_type_from_query(query)
        if query_crack_type:
            result_crack_types = result.metadata.get("crack_types", [])
            if query_crack_type.value in result_crack_types:
                score *= 1.4
        
        # Boost for document type relevance
        doc_type = result.metadata.get("document_type", "")
        if "standard" in doc_type or "handbook" in doc_type:
            score *= 1.15  # Standards and handbooks are authoritative
        
        # Boost for topic match
        query_topics = []
        if "repair" in query.lower() or "fix" in query.lower():
            query_topics.append("repair")
        if "inspect" in query.lower() or "assess" in query.lower():
            query_topics.append("inspection")
        if "cause" in query.lower() or "reason" in query.lower():
            query_topics.append("causes")
        
        result_topics = result.metadata.get("topics", [])
        if any(topic in result_topics for topic in query_topics):
            score *= 1.25
        
        return score
    
    def rerank_results(self, 
                      results: List[Tuple[str, Dict, float]], 
                      query: str,
                      top_k: int = 5) -> List[SearchResult]:
        """
        Re-rank search results based on enhanced relevance scoring
        
        Args:
            results: List of (content, metadata, score) tuples
            query: Original search query
            top_k: Number of top results to return
            
        Returns:
            Re-ranked list of SearchResult objects
        """
        # Convert to SearchResult objects
        search_results = [
            SearchResult(content=content, metadata=metadata, score=score)
            for content, metadata, score in results
        ]
        
        # Expand query
        expanded_terms = self.expand_query(query)
        
        # Calculate enhanced scores
        for result in search_results:
            result.score = self.score_relevance(result, query, expanded_terms)
        
        # Sort by enhanced score
        search_results.sort(key=lambda x: x.score, reverse=True)
        
        # Assign ranks
        for rank, result in enumerate(search_results[:top_k], 1):
            result.rank = rank
        
        return search_results[:top_k]
    
    def format_context(self, 
                      results: List[SearchResult], 
                      max_length: int = 4000,
                      include_metadata: bool = True) -> str:
        """
        Format retrieval results into context string
        
        Args:
            results: List of SearchResult objects
            max_length: Maximum length of context in characters
            include_metadata: Whether to include metadata in formatting
            
        Returns:
            Formatted context string
        """
        if not results:
            return "No relevant documents found."
        
        context_parts = ["=== Relevant Context from Knowledge Base ===\n"]
        current_length = len(context_parts[0])
        
        for result in results:
            # Format metadata
            metadata_str = ""
            if include_metadata:
                source_info = []
                if "filename" in result.metadata:
                    source_info.append(f"Source: {result.metadata['filename']}")
                if "chunk_id" in result.metadata:
                    chunk_info = f"Chunk {result.metadata['chunk_id'] + 1}"
                    if "total_chunks" in result.metadata:
                        chunk_info += f"/{result.metadata['total_chunks']}"
                    source_info.append(chunk_info)
                if "topics" in result.metadata and result.metadata["topics"]:
                    source_info.append(f"Topics: {', '.join(result.metadata['topics'][:3])}")
                
                if source_info:
                    metadata_str = f" [{' | '.join(source_info)}]"
            
            # Format result
            result_text = f"\n[Document {result.rank}]{metadata_str}\n"
            result_text += f"Relevance: {result.score:.3f}\n"
            result_text += f"{result.content}\n"
            result_text += "-" * 80 + "\n"
            
            # Check if adding this would exceed max length
            if current_length + len(result_text) > max_length:
                # Truncate the content to fit
                available_space = max_length - current_length - 200  # Reserve space for truncation notice
                if available_space > 200:
                    truncated_content = result.content[:available_space] + "...[truncated]"
                    result_text = f"\n[Document {result.rank}]{metadata_str}\n"
                    result_text += f"Relevance: {result.score:.3f}\n"
                    result_text += f"{truncated_content}\n"
                    result_text += "-" * 80 + "\n"
                    context_parts.append(result_text)
                break
            
            context_parts.append(result_text)
            current_length += len(result_text)
        
        return "".join(context_parts)
    
    def hybrid_search(self,
                     document_manager,
                     query: str,
                     k: int = 10,
                     final_k: int = 5,
                     crack_type_filter: Optional[CrackType] = None) -> Tuple[List[str], List[Dict], List[float]]:
        """
        Perform hybrid search with retrieval and re-ranking
        
        Args:
            document_manager: The document manager instance
            query: Search query
            k: Number of initial results to retrieve
            final_k: Number of final results after re-ranking
            crack_type_filter: Optional crack type filter
            
        Returns:
            Tuple of (documents, metadata, scores)
        """
        # Retrieve initial results
        docs, metadata, scores = document_manager.retrieve_documents(
            query=query,
            k=k,
            crack_type_filter=crack_type_filter
        )
        
        if not docs:
            return [], [], []
        
        # Combine into tuples for re-ranking
        results = list(zip(docs, metadata, scores))
        
        # Re-rank
        reranked = self.rerank_results(results, query, top_k=final_k)
        
        # Extract back into separate lists
        final_docs = [r.content for r in reranked]
        final_metadata = [r.metadata for r in reranked]
        final_scores = [r.score for r in reranked]
        
        return final_docs, final_metadata, final_scores
