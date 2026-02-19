"""
Structural Crack Analysis RAG System
Package initialization
"""
from .crack_types import CrackType, CrackCharacteristics
from .document_manager import CrackRAGDocumentManager
from .gemini_analyzer import GeminiAnalyzer
from .main_rag import StructuralCrackRAG, create_crack_rag_system

__version__ = "1.0.0"
__author__ = "Structural Engineering AI Team"

__all__ = [
    "CrackType",
    "CrackCharacteristics", 
    "CrackRAGDocumentManager",
    "GeminiAnalyzer",
    "StructuralCrackRAG",
    "create_crack_rag_system"
]
