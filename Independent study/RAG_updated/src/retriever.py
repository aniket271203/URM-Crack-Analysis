from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.documents import Document

class SectionRetriever:
    def __init__(self, db_path="data/chroma_db", collection_name="fema_306"):
        self.embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        
        self.vector_db = Chroma(
            persist_directory=db_path,
            collection_name=collection_name,
            embedding_function=self.embedding_function
        )
        
    def query_with_score(self, query_text, n_results=10, filters=None):
        """
        Retrieve documents with similarity scores.
        
        Returns:
            List[Tuple[Document, float]]: List of (Document, score) tuples.
        """
        # Build filter dict
        chroma_filter = filters
        chroma_filter = filters
        # Chroma handles simple dicts as AND implicitly usually, but explicit $and is safer if supported.
        # User says nested $and was breaking it. Simplest fix: Just pass filters directly if it works.
        # If filters is {"a": 1, "b": 2}, Chroma standard is implicit AND.
        # So we remove the manual "$and" wrapping unless complex logic is needed.
        # Chroma requires explicit $and for multiple conditions in 'where' clause
        if filters and len(filters) > 1 and "$and" not in filters and "$or" not in filters:
             chroma_filter = {"$and": [{k: v} for k, v in filters.items()]}
             
        results = self.vector_db.similarity_search_with_score(
            query_text, 
            k=n_results, 
            filter=chroma_filter
        )
        return results

    def query(self, query_text, n_results=3, filters=None):
        # Wrapper for backward compatibility
        results_with_score = self.query_with_score(query_text, n_results, filters)
        return [doc for doc, _ in results_with_score]

    def get_failure_mode(self, mode_name):
        """Helper to find a specific failure mode definition."""
        return self.query(
            mode_name, 
            n_results=1, 
            filters={"is_failure_mode": True} # Removed potentially problematic nested filter
        )
    
if __name__ == "__main__":
    # Quick Test
    retriever = SectionRetriever()
    docs = retriever.query("diagonal cracking", filters={"is_failure_mode": True})
    
    print(f"Found {len(docs)} documents:")
    for doc in docs:
        print(f"[{doc.metadata.get('id')}] {doc.metadata.get('title')}")
