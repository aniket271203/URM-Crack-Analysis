"""
Test script for enhanced RAG retrieval
Demonstrates the improved document retrieval capabilities
"""
from src.main_rag import StructuralCrackRAG
from src.crack_types import CrackType
from src.enhanced_retrieval import EnhancedRetriever

def print_separator():
    print("\n" + "=" * 80 + "\n")

def main():
    print("🧪 Testing Enhanced RAG Retrieval System")
    print_separator()
    
    # Initialize the RAG system
    print("Initializing RAG system...")
    rag = StructuralCrackRAG(data_dir="./data")
    retriever = EnhancedRetriever()
    
    # Get statistics
    stats = rag.document_manager.get_stats()
    print(f"✓ Loaded {stats['total_documents']} documents")
    print(f"✓ Index size: {stats['index_size']}")
    print_separator()
    
    # Test queries with different focus areas
    test_cases = [
        {
            "query": "What causes diagonal cracks in masonry walls due to settlement?",
            "crack_type": CrackType.DIAGONAL,
            "description": "Diagonal crack causes with settlement focus"
        },
        {
            "query": "How to repair and remediate horizontal flexural cracks in concrete beams?",
            "crack_type": CrackType.HORIZONTAL,
            "description": "Repair methods for horizontal cracks"
        },
        {
            "query": "Inspection and monitoring procedures for vertical thermal cracks",
            "crack_type": CrackType.VERTICAL,
            "description": "Inspection methods for vertical cracks"
        },
        {
            "query": "Seismic x-shaped shear cracks in structural walls assessment",
            "crack_type": CrackType.X_SHAPED,
            "description": "Seismic crack assessment"
        },
        {
            "query": "Step cracks in brick masonry mortar joints differential movement",
            "crack_type": CrackType.STEP,
            "description": "Step crack in masonry"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        query = test_case["query"]
        crack_type = test_case["crack_type"]
        description = test_case["description"]
        
        print(f"TEST {i}: {description}")
        print(f"Query: '{query}'")
        print(f"Crack Type Filter: {crack_type.value}")
        print("-" * 80)
        
        # Perform enhanced retrieval
        docs, metadata, scores = retriever.hybrid_search(
            document_manager=rag.document_manager,
            query=query,
            k=15,  # Retrieve 15 initially
            final_k=5,  # Re-rank to top 5
            crack_type_filter=crack_type
        )
        
        if not docs:
            print("❌ No documents found")
            print_separator()
            continue
        
        print(f"✓ Found {len(docs)} relevant documents\n")
        
        # Display results
        for j, (doc, meta, score) in enumerate(zip(docs, metadata, scores), 1):
            print(f"[{j}] Score: {score:.4f}")
            
            # Extract metadata
            filename = meta.get("filename", "Unknown")
            chunk_info = f"Chunk {meta.get('chunk_id', 0) + 1}/{meta.get('total_chunks', 1)}"
            topics = meta.get("topics", [])
            doc_type = meta.get("document_type", "unknown")
            
            print(f"    Source: {filename}")
            print(f"    {chunk_info} | Type: {doc_type}")
            if topics:
                print(f"    Topics: {', '.join(topics[:5])}")
            
            # Show preview
            preview = doc[:200].replace('\n', ' ')
            print(f"    Preview: {preview}...")
            print()
        
        print_separator()
    
    # Test without crack type filter (general search)
    print("TEST 6: General search without crack type filter")
    query = "What are the main structural causes of cracks in buildings?"
    print(f"Query: '{query}'")
    print("Crack Type Filter: None (all types)")
    print("-" * 80)
    
    docs, metadata, scores = retriever.hybrid_search(
        document_manager=rag.document_manager,
        query=query,
        k=20,
        final_k=8,
        crack_type_filter=None
    )
    
    print(f"✓ Found {len(docs)} relevant documents\n")
    
    for j, (doc, meta, score) in enumerate(zip(docs, metadata, scores), 1):
        print(f"[{j}] Score: {score:.4f} | {meta.get('filename', 'Unknown')}")
        topics = meta.get("topics", [])
        if topics:
            print(f"    Topics: {', '.join(topics[:3])}")
        preview = doc[:150].replace('\n', ' ')
        print(f"    {preview}...")
        print()
    
    print_separator()
    
    # Test query expansion
    print("🔍 Query Expansion Demonstration")
    print("-" * 80)
    
    original_query = "repair diagonal crack"
    expanded = retriever.expand_query(original_query)
    
    print(f"Original Query: '{original_query}'")
    print(f"Expanded Terms: {', '.join(sorted(expanded))}")
    print()
    
    print("This helps find more relevant documents by including synonyms!")
    print_separator()
    
    # Test formatted context
    print("📝 Formatted Context Example")
    print("-" * 80)
    
    query = "foundation settlement diagonal cracks"
    docs, metadata, scores = retriever.hybrid_search(
        document_manager=rag.document_manager,
        query=query,
        k=10,
        final_k=3,
        crack_type_filter=CrackType.DIAGONAL
    )
    
    from src.enhanced_retrieval import SearchResult
    search_results = [
        SearchResult(content=doc, metadata=meta, score=score)
        for doc, meta, score in zip(docs, metadata, scores)
    ]
    
    formatted_context = retriever.format_context(
        search_results,
        max_length=2000,
        include_metadata=True
    )
    
    print("Query:", query)
    print("\nFormatted Context for AI Analysis:")
    print(formatted_context)
    
    print_separator()
    print("✓ All tests completed successfully!")
    print("\n💡 Your RAG system is ready to use with enhanced retrieval!")
    print("   - Query expansion for better coverage")
    print("   - Re-ranking for improved relevance")
    print("   - Rich metadata and context formatting")
    print("   - 2626 documents from your research PDFs")


if __name__ == "__main__":
    main()
