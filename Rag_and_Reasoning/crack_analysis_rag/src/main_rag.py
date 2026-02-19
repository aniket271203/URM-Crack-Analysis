"""
Main Structural Crack Analysis RAG System
Orchestrates the complete crack analysis pipeline
"""
import os
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

try:
    from .crack_types import CrackType, CrackCharacteristics
    from .document_manager import CrackRAGDocumentManager
    from .gemini_analyzer import GeminiAnalyzer
    from .enhanced_retrieval import EnhancedRetriever
except ImportError:
    # Fallback to absolute imports when not run as package
    from crack_types import CrackType, CrackCharacteristics
    from document_manager import CrackRAGDocumentManager
    from gemini_analyzer import GeminiAnalyzer
    from enhanced_retrieval import EnhancedRetriever


class StructuralCrackRAG:
    """
    Main RAG system for structural crack analysis
    
    Combines document retrieval with Gemini AI analysis for comprehensive
    crack analysis including location identification and cause determination.
    """
    
    def __init__(self, 
                 data_dir: str = "./data",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 gemini_model: str = "gemini-2.5-flash-lite",
                 env_file: Optional[str] = None):
        """
        Initialize the crack analysis RAG system
        
        Args:
            data_dir: Directory for storing documents and embeddings
            embedding_model: Sentence transformer model for embeddings
            gemini_model: Gemini model for analysis
            env_file: Path to .env file (optional)
        """
        # Load environment variables
        if env_file:
            load_dotenv(env_file)
        else:
            load_dotenv()
        
        # Get configuration from environment or use defaults
        gemini_model = os.getenv("MODEL_NAME", gemini_model)
        embedding_model = os.getenv("EMBEDDING_MODEL", embedding_model)
        
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize components
        print("Initializing document manager...")
        self.document_manager = CrackRAGDocumentManager(
            embedding_model=embedding_model,
            data_dir=str(self.data_dir)
        )
        
        print("Initializing enhanced retriever...")
        self.enhanced_retriever = EnhancedRetriever()
        
        print("Initializing Gemini analyzer...")
        self.gemini_analyzer = GeminiAnalyzer(model_name=gemini_model)
        
        # Test connections
        if not self.gemini_analyzer.test_connection():
            raise RuntimeError("Failed to connect to Gemini API. Check your API key.")
        
        print("Structural Crack RAG system initialized successfully!")
    
    def add_document(self, 
                    content: str,
                    crack_type: Optional[CrackType] = None,
                    title: str = "",
                    source: str = "manual",
                    **metadata) -> None:
        """
        Add a document to the knowledge base
        
        Args:
            content: Document content
            crack_type: Associated crack type
            title: Document title
            source: Document source
            **metadata: Additional metadata
        """
        self.document_manager.add_document(
            content=content,
            crack_type=crack_type,
            title=title,
            source=source,
            **metadata
        )
        self.document_manager.save_data()
    
    def add_documents_from_file(self, file_path: str) -> None:
        """
        Add documents from a JSON file
        
        Args:
            file_path: Path to JSON file containing documents
        """
        import json
        
        with open(file_path, 'r') as f:
            documents = json.load(f)
        
        # Convert string crack types to enum
        for doc in documents:
            if 'crack_type' in doc and doc['crack_type']:
                try:
                    doc['crack_type'] = CrackType(doc['crack_type'])
                except ValueError:
                    doc['crack_type'] = None
        
        self.document_manager.add_documents_batch(documents)
        self.document_manager.save_data()
    
    def analyze_crack(self, 
                     image_path: str,
                     crack_type: CrackType,
                     use_rag: bool = True,
                     rag_query: Optional[str] = None,
                     additional_context: str = "") -> Dict[str, Any]:
        """
        Perform complete crack analysis
        
        Args:
            image_path: Path to crack image
            crack_type: Pre-classified crack type
            use_rag: Whether to use RAG document retrieval
            rag_query: Custom query for RAG (if None, uses crack type keywords)
            additional_context: Additional context for analysis
            
        Returns:
            Complete analysis results
        """
        print(f"Starting analysis for {crack_type.value} crack...")
        
        # Step 1: Location Analysis
        print("Step 1: Analyzing crack location and characteristics...")
        location_result = self.gemini_analyzer.analyze_crack_location(
            image_path=image_path,
            crack_type=crack_type,
            additional_context=additional_context
        )
        
        if not location_result["success"]:
            return {
                "success": False,
                "error": f"Location analysis failed: {location_result['error']}",
                "step_failed": "location_analysis"
            }
        
        # Step 2: RAG Context Retrieval (if enabled)
        rag_context = ""
        retrieved_docs = []
        
        if use_rag:
            print("Retrieving relevant documentation...")
            
            # Use custom query or generate from crack type
            if rag_query is None:
                keywords = CrackCharacteristics.get_analysis_keywords(crack_type)
                rag_query = f"{crack_type.value} crack causes and " + " ".join(keywords[:3])
            
            # Use enhanced retrieval
            docs, metadata, scores = self.enhanced_retriever.hybrid_search(
                document_manager=self.document_manager,
                query=rag_query,
                k=10,  # Retrieve more initially
                final_k=5,  # Re-rank to top 5
                crack_type_filter=crack_type
            )
            
            # Build context from retrieved documents using enhanced formatter
            if docs:
                # Convert to SearchResult format for enhanced formatter
                from enhanced_retrieval import SearchResult
                search_results = [
                    SearchResult(content=doc, metadata=meta, score=score)
                    for doc, meta, score in zip(docs, metadata, scores)
                ]
                
                # Format context nicely
                rag_context = self.enhanced_retriever.format_context(
                    search_results,
                    max_length=4000,
                    include_metadata=True
                )
                
                retrieved_docs = [{"content": doc, "metadata": meta, "score": score} 
                                for doc, meta, score in zip(docs, metadata, scores)]
                print(f"Retrieved and re-ranked {len(docs)} relevant documents")
            else:
                print("No relevant documents found in knowledge base")
        
        # Step 3: Cause Determination
        print("Step 2: Determining crack cause...")
        cause_result = self.gemini_analyzer.determine_crack_cause(
            image_path=image_path,
            crack_type=crack_type,
            location_analysis=location_result["analysis"],
            rag_context=rag_context,
            use_rag=use_rag
        )
        
        if not cause_result["success"]:
            return {
                "success": False,
                "error": f"Cause analysis failed: {cause_result['error']}",
                "step_failed": "cause_determination",
                "location_analysis": location_result
            }
        
        # Step 4: Generate Comprehensive Report
        print("Generating comprehensive report...")
        report_result = self.gemini_analyzer.generate_comprehensive_report(
            image_path=image_path,
            crack_type=crack_type,
            location_analysis=location_result["analysis"],
            cause_analysis=cause_result["analysis"]
        )
        
        # Compile final results
        final_result = {
            "success": True,
            "crack_type": crack_type.value,
            "image_path": image_path,
            "analysis_steps": {
                "step_1_location": location_result,
                "step_2_cause": cause_result,
                "step_3_report": report_result
            },
            "rag_info": {
                "used_rag": use_rag,
                "query": rag_query if use_rag else None,
                "retrieved_documents": retrieved_docs,
                "context_provided": bool(rag_context)
            },
            "summary": {
                "location_analysis": location_result["analysis"][:500] + "..." if len(location_result["analysis"]) > 500 else location_result["analysis"],
                "cause_analysis": cause_result["analysis"][:500] + "..." if len(cause_result["analysis"]) > 500 else cause_result["analysis"],
                "comprehensive_report": report_result["report"] if report_result["success"] else "Report generation failed"
            }
        }
        
        print("Analysis complete!")
        return final_result
    
    def quick_analysis(self, 
                      image_path: str,
                      crack_type: CrackType) -> Dict[str, Any]:
        """
        Perform quick analysis without RAG (using only Gemini knowledge)
        
        Args:
            image_path: Path to crack image
            crack_type: Pre-classified crack type
            
        Returns:
            Quick analysis results
        """
        return self.analyze_crack(
            image_path=image_path,
            crack_type=crack_type,
            use_rag=False
        )
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics and health info"""
        doc_stats = self.document_manager.get_stats()
        gemini_status = self.gemini_analyzer.test_connection()
        
        return {
            "document_manager": doc_stats,
            "gemini_connection": gemini_status,
            "supported_crack_types": [ct.value for ct in CrackType],
            "data_directory": str(self.data_dir)
        }
    
    def export_analysis_results(self, 
                               analysis_results: Dict[str, Any],
                               output_file: str) -> None:
        """
        Export analysis results to a file
        
        Args:
            analysis_results: Results from analyze_crack()
            output_file: Output file path (JSON or TXT)
        """
        import json
        from datetime import datetime
        
        # Add metadata
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "system_info": {
                "version": "1.0.0",
                "model": self.gemini_analyzer.model_name
            },
            "analysis_results": analysis_results
        }
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.suffix.lower() == '.json':
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
        else:
            # Export as formatted text
            with open(output_path, 'w') as f:
                f.write("STRUCTURAL CRACK ANALYSIS REPORT\\n")
                f.write("=" * 50 + "\\n\\n")
                f.write(f"Generated: {export_data['export_timestamp']}\\n")
                f.write(f"Crack Type: {analysis_results.get('crack_type', 'Unknown')}\\n")
                f.write(f"Image: {analysis_results.get('image_path', 'Unknown')}\\n\\n")
                
                if analysis_results.get("success"):
                    summary = analysis_results.get("summary", {})
                    f.write("COMPREHENSIVE REPORT:\\n")
                    f.write("-" * 20 + "\\n")
                    f.write(summary.get("comprehensive_report", "No report available"))
                else:
                    f.write(f"ANALYSIS FAILED: {analysis_results.get('error', 'Unknown error')}\\n")
        
        print(f"Analysis results exported to: {output_path}")


# Convenience function for quick setup
def create_crack_rag_system(data_dir: str = "./data", 
                           env_file: Optional[str] = None) -> StructuralCrackRAG:
    """
    Create and return a configured StructuralCrackRAG system
    
    Args:
        data_dir: Directory for storing data
        env_file: Path to .env file
        
    Returns:
        Configured StructuralCrackRAG instance
    """
    return StructuralCrackRAG(data_dir=data_dir, env_file=env_file)
