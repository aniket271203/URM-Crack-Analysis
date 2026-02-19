"""
Document Processing Script for Crack Analysis RAG
Processes PDF documents and adds them to the vector database with intelligent chunking
"""
import os
import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import re

try:
    import PyPDF2
    import pdfplumber
except ImportError:
    print("Installing required PDF processing libraries...")
    os.system(f"{sys.executable} -m pip install PyPDF2 pdfplumber")
    import PyPDF2
    import pdfplumber

try:
    from src.main_rag import StructuralCrackRAG
    from src.crack_types import CrackType
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from src.main_rag import StructuralCrackRAG
    from src.crack_types import CrackType


class DocumentProcessor:
    """Processes PDF documents for the RAG system"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize document processor
        
        Args:
            chunk_size: Target size of text chunks in characters
            chunk_overlap: Number of overlapping characters between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def extract_text_from_pdf(self, pdf_path: str) -> Tuple[str, Dict]:
        """
        Extract text from PDF with metadata
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Tuple of (extracted_text, metadata)
        """
        print(f"Processing: {os.path.basename(pdf_path)}")
        
        text = ""
        metadata = {
            "filename": os.path.basename(pdf_path),
            "pages": 0
        }
        
        # Try pdfplumber first (better text extraction)
        try:
            with pdfplumber.open(pdf_path) as pdf:
                metadata["pages"] = len(pdf.pages)
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                        
            if text.strip():
                print(f"  ✓ Extracted {len(text)} characters from {metadata['pages']} pages")
                return text, metadata
        except Exception as e:
            print(f"  ⚠ pdfplumber failed: {e}, trying PyPDF2...")
        
        # Fallback to PyPDF2
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                metadata["pages"] = len(pdf_reader.pages)
                
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                        
            print(f"  ✓ Extracted {len(text)} characters from {metadata['pages']} pages")
        except Exception as e:
            print(f"  ✗ Failed to extract text: {e}")
            
        return text, metadata
    
    def clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove page numbers and headers/footers patterns
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        # Fix hyphenated words at line breaks
        text = re.sub(r'-\s+', '', text)
        return text.strip()
    
    def chunk_text(self, text: str, metadata: Dict) -> List[Dict]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Text to chunk
            metadata: Metadata to include with each chunk
            
        Returns:
            List of chunk dictionaries
        """
        # Clean the text first
        text = self.clean_text(text)
        
        if len(text) <= self.chunk_size:
            return [{
                "content": text,
                "chunk_id": 0,
                "total_chunks": 1,
                **metadata
            }]
        
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            # Find the end of this chunk
            end = start + self.chunk_size
            
            # If not at the end, try to break at a sentence or paragraph
            if end < len(text):
                # Look for paragraph break
                next_para = text.find('\n\n', end - 100, end + 100)
                if next_para != -1:
                    end = next_para
                else:
                    # Look for sentence break
                    for punct in ['. ', '! ', '? ', '.\n']:
                        next_sent = text.find(punct, end - 50, end + 50)
                        if next_sent != -1:
                            end = next_sent + len(punct)
                            break
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunks.append({
                    "content": chunk_text,
                    "chunk_id": chunk_id,
                    "total_chunks": -1,  # Will be updated after all chunks are created
                    **metadata
                })
                chunk_id += 1
            
            # Move start position with overlap
            start = end - self.chunk_overlap
        
        # Update total_chunks for all chunks
        for chunk in chunks:
            chunk["total_chunks"] = len(chunks)
        
        print(f"  → Created {len(chunks)} chunks")
        return chunks
    
    def categorize_document(self, filename: str, text: str) -> Dict:
        """
        Automatically categorize document based on content and filename
        
        Args:
            filename: Document filename
            text: Document text content
            
        Returns:
            Dictionary with categorization info
        """
        filename_lower = filename.lower()
        text_lower = text.lower()
        
        categories = {
            "crack_types": [],
            "topics": [],
            "document_type": "general"
        }
        
        # Check for crack type mentions - BUT DON'T TAG GENERAL DOCS
        # Only tag if the document is SPECIFICALLY about that crack type
        crack_keywords = {
            CrackType.DIAGONAL: ["diagonal crack", "diagonal cracks", "diagonal cracking"],
            CrackType.VERTICAL: ["vertical crack", "vertical cracks", "vertical cracking"],
            CrackType.HORIZONTAL: ["horizontal crack", "horizontal cracks", "horizontal cracking"],
            CrackType.STEP: ["step crack", "step cracks", "stepped crack", "stair crack"],
            CrackType.X_SHAPED: ["x-shaped crack", "x shaped crack", "cross crack"]
        }
        
        # Only tag if crack type is mentioned frequently (not just once)
        for crack_type, keywords in crack_keywords.items():
            count = sum(text_lower.count(kw) for kw in keywords)
            if count >= 3:  # Must appear at least 3 times to be tagged
                categories["crack_types"].append(crack_type.value)
        
        # Check for topics
        topic_keywords = {
            "inspection": ["inspection", "assessment", "evaluation", "survey"],
            "repair": ["repair", "remediation", "restoration", "maintenance"],
            "causes": ["cause", "reason", "origin", "mechanism"],
            "monitoring": ["monitoring", "tracking", "measurement"],
            "masonry": ["masonry", "brick", "mortar", "CMU", "concrete block"],
            "concrete": ["concrete", "reinforced", "prestressed"],
            "structural": ["structural", "load", "stress", "strain"],
            "seismic": ["seismic", "earthquake", "lateral", "shake"]
        }
        
        for topic, keywords in topic_keywords.items():
            count = sum(1 for kw in keywords if kw in text_lower)
            if count >= 2:  # At least 2 keyword matches
                categories["topics"].append(topic)
        
        # Determine document type
        if any(kw in filename_lower for kw in ["fema", "standard", "code", "guideline"]):
            categories["document_type"] = "standard"
        elif any(kw in filename_lower for kw in ["manual", "handbook", "guide"]):
            categories["document_type"] = "handbook"
        elif any(kw in text_lower for kw in ["abstract", "introduction", "conclusion", "references"]):
            categories["document_type"] = "research_paper"
        elif "form" in filename_lower or "inspection" in filename_lower:
            categories["document_type"] = "inspection_form"
        
        return categories
    
    def process_pdf_directory(self, pdf_dir: str) -> List[Dict]:
        """
        Process all PDFs in a directory
        
        Args:
            pdf_dir: Directory containing PDF files
            
        Returns:
            List of processed document chunks
        """
        pdf_dir = Path(pdf_dir)
        pdf_files = list(pdf_dir.glob("*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in {pdf_dir}")
            return []
        
        print(f"\nFound {len(pdf_files)} PDF files to process\n")
        
        all_chunks = []
        
        for pdf_file in pdf_files:
            try:
                # Extract text
                text, metadata = self.extract_text_from_pdf(str(pdf_file))
                
                if not text.strip():
                    print(f"  ⚠ No text extracted, skipping...")
                    continue
                
                # Categorize document
                categories = self.categorize_document(pdf_file.name, text)
                metadata.update(categories)
                metadata["source"] = "pdf_corpus"
                
                # Create chunks
                chunks = self.chunk_text(text, metadata)
                all_chunks.extend(chunks)
                
                print(f"  Categories: {', '.join(categories['topics'] or ['general'])}")
                print()
                
            except Exception as e:
                print(f"  ✗ Error processing {pdf_file.name}: {e}\n")
                continue
        
        print(f"✓ Total chunks created: {len(all_chunks)}\n")
        return all_chunks


def main():
    """Main processing function"""
    print("=" * 70)
    print("Crack Analysis RAG - Document Processing")
    print("=" * 70)
    
    # Configuration
    pdf_dir = "../RAG_Docs"
    data_dir = "./data"
    chunk_size = 1000  # characters
    chunk_overlap = 200  # characters
    
    # Initialize processor
    processor = DocumentProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    # Process documents
    print("\n📄 Processing PDF documents...")
    chunks = processor.process_pdf_directory(pdf_dir)
    
    if not chunks:
        print("❌ No documents were processed successfully.")
        return
    
    # Save processed chunks to JSON for inspection
    chunks_file = Path(data_dir) / "processed_chunks.json"
    with open(chunks_file, 'w') as f:
        json.dump(chunks, f, indent=2)
    print(f"💾 Saved processed chunks to {chunks_file}")
    
    # Initialize RAG system
    print("\n🔧 Initializing RAG system...")
    try:
        rag_system = StructuralCrackRAG(data_dir=data_dir)
    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {e}")
        print("Make sure you have a .env file with GEMINI_API_KEY set.")
        return
    
    # Add documents to RAG system
    print(f"\n📚 Adding {len(chunks)} chunks to vector database...")
    
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        
        # Prepare batch for adding
        formatted_batch = []
        for chunk in batch:
            # Extract crack_type if present
            crack_types = chunk.get("crack_types", [])
            crack_type = crack_types[0] if crack_types else None
            
            doc_dict = {
                "content": chunk["content"],
                "title": f"{chunk['filename']} (Chunk {chunk['chunk_id'] + 1}/{chunk['total_chunks']})",
                "source": chunk.get("source", "pdf_corpus"),
                "crack_type": crack_type,
                "filename": chunk.get("filename"),
                "page_count": chunk.get("pages"),
                "chunk_id": chunk.get("chunk_id"),
                "total_chunks": chunk.get("total_chunks"),
                "document_type": chunk.get("document_type"),
                "topics": chunk.get("topics", [])
            }
            formatted_batch.append(doc_dict)
        
        try:
            rag_system.document_manager.add_documents_batch(formatted_batch)
            print(f"  ✓ Added batch {i // batch_size + 1}/{(len(chunks) - 1) // batch_size + 1}")
        except Exception as e:
            print(f"  ✗ Error adding batch: {e}")
    
    # Save everything
    print("\n💾 Saving to vector database...")
    rag_system.document_manager.save_data()
    
    # Display statistics
    print("\n📊 Database Statistics:")
    stats = rag_system.document_manager.get_stats()
    print(f"  Total documents: {stats['total_documents']}")
    print(f"  Index size: {stats['index_size']}")
    print(f"\n  Distribution by crack type:")
    for crack_type, count in stats['crack_type_distribution'].items():
        print(f"    {crack_type or 'general'}: {count}")
    
    # Test retrieval
    print("\n🔍 Testing retrieval...")
    test_queries = [
        "What causes diagonal cracks in masonry walls?",
        "How to repair vertical cracks in concrete?",
        "Inspection methods for structural cracks"
    ]
    
    for query in test_queries:
        print(f"\n  Query: '{query}'")
        docs, metadata, scores = rag_system.document_manager.retrieve_documents(query, k=3)
        for i, (doc, meta, score) in enumerate(zip(docs, metadata, scores), 1):
            print(f"    {i}. [{score:.3f}] {meta.get('title', 'Untitled')}")
            print(f"       {doc[:100]}...")
    
    print("\n" + "=" * 70)
    print("✓ Document processing complete!")
    print("=" * 70)
    print(f"\nYour RAG system now has {stats['total_documents']} documents in the vector database.")
    print("You can now use it for crack analysis with enhanced context retrieval.")


if __name__ == "__main__":
    main()
