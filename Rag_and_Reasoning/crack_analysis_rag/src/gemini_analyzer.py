"""
Gemini API Integration for Structural Crack Analysis
Handles image analysis and multi-step reasoning
"""
import os
import base64
from typing import Optional, Dict, Any, List
from pathlib import Path
import google.generativeai as genai
from PIL import Image
import io

try:
    from .crack_types import CrackType
except ImportError:
    from crack_types import CrackType


class GeminiAnalyzer:
    """Handles Gemini API interactions for crack analysis"""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.0-flash-exp"):
        """
        Initialize Gemini analyzer
        
        Args:
            api_key: Google API key (if None, loads from environment)
            model_name: Gemini model to use
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key not provided and GOOGLE_API_KEY not set in environment")
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)
        
        # Generation configuration
        self.generation_config = {
            "temperature": 0.3,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 4096,
        }
        
        # Safety settings (very permissive for technical/engineering analysis)
        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
    
    def _prepare_image(self, image_path: str) -> Image.Image:
        """
        Prepare image for Gemini API
        
        Args:
            image_path: Path to the image file
            
        Returns:
            PIL Image object
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = Image.open(image_path)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize if too large (Gemini has size limits)
        max_size = 4096
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        return image
    
    def analyze_crack_location(self, 
                              image_path: str, 
                              crack_type: CrackType,
                              additional_context: str = "") -> Dict[str, Any]:
        """
        Step 1: Analyze crack location and characteristics
        
        Args:
            image_path: Path to the crack image
            crack_type: Pre-classified crack type
            additional_context: Additional context for analysis
            
        Returns:
            Dictionary with location analysis results
        """
        image = self._prepare_image(image_path)
        
        prompt = f"""
You are a professional structural engineer conducting technical analysis of building infrastructure.

ENGINEERING ANALYSIS REQUEST:
Crack Type: {crack_type.value.upper()}

TASK: Perform a focused technical analysis of this {crack_type.value} crack.

Provide a concise structural engineering assessment:

1. CRACK LOCATION:
   - Precise position and dimensions
   - Orientation (angle from horizontal)
   - Pattern description

2. PHYSICAL CHARACTERISTICS:
   - Width (estimated in mm)
   - Length and extent
   - Through-thickness or surface crack

3. STRUCTURAL CONTEXT:
   - Material type and construction
   - Structural element affected
   - Visible reinforcement (if any)

4. CRITICAL OBSERVATIONS:
   - Severity indicators
   - Signs of movement
   - Immediate safety concerns

{additional_context}

Keep analysis focused and technical. Avoid excessive detail.
"""
        
        try:
            response = self.model.generate_content(
                [prompt, image],
                generation_config=self.generation_config,
                safety_settings=self.safety_settings
            )
            
            # Check if response was blocked
            if not response.text:
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'finish_reason'):
                        if candidate.finish_reason == 2:  # SAFETY
                            return {
                                "success": False,
                                "error": "Response blocked by safety filter. This is a technical engineering analysis and should be allowed.",
                                "crack_type": crack_type.value,
                                "step": "location_analysis",
                                "finish_reason": "SAFETY"
                            }
                        else:
                            return {
                                "success": False,
                                "error": f"Response incomplete. Finish reason: {candidate.finish_reason}",
                                "crack_type": crack_type.value,
                                "step": "location_analysis",
                                "finish_reason": candidate.finish_reason
                            }
                return {
                    "success": False,
                    "error": "Empty response from model",
                    "crack_type": crack_type.value,
                    "step": "location_analysis"
                }
            
            return {
                "success": True,
                "analysis": response.text,
                "crack_type": crack_type.value,
                "step": "location_analysis"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "crack_type": crack_type.value,
                "step": "location_analysis"
            }
    
    def determine_crack_cause(self, 
                             image_path: str,
                             crack_type: CrackType,
                             location_analysis: str,
                             rag_context: str = "",
                             use_rag: bool = True) -> Dict[str, Any]:
        """
        Step 2: Determine the exact cause of the crack
        
        Args:
            image_path: Path to the crack image
            crack_type: Pre-classified crack type
            location_analysis: Results from step 1 analysis
            rag_context: Context from RAG document retrieval
            use_rag: Whether to use RAG context or rely on model knowledge
            
        Returns:
            Dictionary with cause analysis results
        """
        image = self._prepare_image(image_path)
        
        # Build context section
        context_section = ""
        if use_rag and rag_context:
            context_section = f"""
RELEVANT TECHNICAL DOCUMENTATION:
{rag_context}

Use this documentation along with your analysis to determine the cause.
"""
        else:
            context_section = """
Based on your engineering knowledge and the visual evidence, determine the cause.
"""
        
        prompt = f"""
You are a professional structural engineer conducting root cause analysis for building infrastructure.

ENGINEERING ANALYSIS REQUEST:
Crack Type: {crack_type.value.upper()}

PREVIOUS STRUCTURAL ASSESSMENT:
{location_analysis}

{context_section}

TASK: Determine the SINGLE, DEFINITIVE root cause of this structural crack.

ANALYSIS REQUIREMENTS:
- Provide ONE primary cause only (not multiple possibilities)
- Be definitive and confident in your assessment
- Keep analysis concise but technically accurate
- Focus on the most likely engineering reason

Provide a focused engineering assessment with:

1. DEFINITIVE ROOT CAUSE:
   - Single primary engineering cause
   - Technical confidence: HIGH
   - Brief engineering explanation (2-3 sentences)

2. FAILURE MECHANISM:
   - How exactly this cause created the crack
   - Stress pattern that developed
   - Why it failed at this location

3. STRUCTURAL IMPACT:
   - Current safety level (High/Medium/Low risk)
   - Immediate structural concerns
   - Load capacity reduction

4. IMMEDIATE ACTIONS REQUIRED:
   - Most critical immediate step
   - Safety measures needed
   - Next engineering investigation

Keep the analysis precise, confident, and concise. Avoid listing multiple possible causes.
"""
        
        try:
            response = self.model.generate_content(
                [prompt, image],
                generation_config=self.generation_config,
                safety_settings=self.safety_settings
            )
            
            # Check if response was blocked
            if not response.text:
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'finish_reason'):
                        if candidate.finish_reason == 2:  # SAFETY
                            return {
                                "success": False,
                                "error": "Response blocked by safety filter. This is a technical engineering analysis and should be allowed.",
                                "crack_type": crack_type.value,
                                "step": "cause_determination",
                                "used_rag": use_rag,
                                "finish_reason": "SAFETY"
                            }
                        else:
                            return {
                                "success": False,
                                "error": f"Response incomplete. Finish reason: {candidate.finish_reason}",
                                "crack_type": crack_type.value,
                                "step": "cause_determination",
                                "used_rag": use_rag,
                                "finish_reason": candidate.finish_reason
                            }
                return {
                    "success": False,
                    "error": "Empty response from model",
                    "crack_type": crack_type.value,
                    "step": "cause_determination",
                    "used_rag": use_rag
                }
            
            return {
                "success": True,
                "analysis": response.text,
                "crack_type": crack_type.value,
                "step": "cause_determination",
                "used_rag": use_rag
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "crack_type": crack_type.value,
                "step": "cause_determination",
                "used_rag": use_rag
            }
    
    def generate_comprehensive_report(self,
                                    image_path: str,
                                    crack_type: CrackType,
                                    location_analysis: str,
                                    cause_analysis: str) -> Dict[str, Any]:
        """
        Generate a comprehensive crack analysis report
        
        Args:
            image_path: Path to the crack image
            crack_type: Crack type
            location_analysis: Step 1 results
            cause_analysis: Step 2 results
            
        Returns:
            Formatted comprehensive report
        """
        image = self._prepare_image(image_path)
        
        prompt = f"""
Generate a well-structured crack analysis report in PLAIN TEXT format (no markdown symbols).

CRACK TYPE: {crack_type.value.upper()}

LOCATION ANALYSIS:
{location_analysis}

CAUSE ANALYSIS:
{cause_analysis}

Create a professional report in PLAIN TEXT with these sections:

STRUCTURAL CRACK ANALYSIS REPORT
================================

1. CRACK SUMMARY
   - Basic crack information in simple English

2. WHAT WE FOUND
   - Key observations from the analysis

3. THE MAIN CAUSE
   - Single definitive reason for the crack
   - Why we are confident about this cause

4. HOW SERIOUS IS IT
   - Current danger level
   - What could happen if not fixed

5. WHAT TO DO NOW
   - Immediate steps needed
   - Who to contact

6. TECHNICAL DETAILS
   - Engineering information for professionals

Requirements:
- Use PLAIN TEXT formatting only (no *, #, or markdown)
- Write in clear, simple English
- Be definitive about the cause
- Keep it well-organized with clear sections
- Use proper spacing and indentation
"""
        
        try:
            response = self.model.generate_content(
                [prompt, image],
                generation_config=self.generation_config,
                safety_settings=self.safety_settings
            )
            
            return {
                "success": True,
                "report": response.text,
                "crack_type": crack_type.value,
                "step": "comprehensive_report"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "crack_type": crack_type.value,
                "step": "comprehensive_report"
            }
    
    def test_connection(self) -> bool:
        """Test if the Gemini API connection is working"""
        try:
            response = self.model.generate_content("Hello, this is a test.")
            return True
        except Exception as e:
            print(f"Gemini API connection test failed: {e}")
            return False
