from pydantic import BaseModel, Field
from typing import List, Optional

class Citation(BaseModel):
    section_id: str
    title: str
    text_snippet: str
    relevance_score: Optional[float] = None

class FailureMode(BaseModel):
    id: str
    name: str
    material: str
    type: str # e.g. "Sliding", "Cracking"
    description: str

class DiagnosisResult(BaseModel):
    failure_mode: FailureMode
    confidence_score: float = Field(..., description="0.0 to 1.0 score of confidence")
    reasoning: str = Field(..., description="Explanation of why this failure mode matches the input metrics")
    citations: List[Citation]
    damage_level: Optional[str] = Field(None, description="FEMA 306 Damage Level (e.g., 'Moderate')")
    severity: Optional[str] = Field(None, description="Rule-based severity tag (Fine, Moderate, Severe)")
    recommended_action: Optional[str] = None
    
    # Hybrid Scoring Details
    hybrid_score: Optional[float] = Field(None, description="Combined score: 0.5*LLM + 0.3*Retriever + 0.2*CrossEncoder")
    llm_confidence: Optional[float] = Field(None, description="Self-reported confidence from LLM (0-1)")
    retrieval_confidence: Optional[float] = Field(None, description="Softmax of retrieval similarity scores")
    cross_encoder_confidence: Optional[float] = Field(None, description="Normalized cross-encoder re-ranking score")
