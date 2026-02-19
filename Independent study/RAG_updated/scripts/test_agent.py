import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.retriever import SectionRetriever
from src.llm_service import GeminiLLMService
from src.structural_agent import StructuralAgent
import argparse
import json
from src.schema import DiagnosisResult

def print_diagnosis_result(result: DiagnosisResult):
    """Helper function to print the diagnosis result in a structured format."""
    print("-" * 50)
    print(f"Diagnosis: {result.failure_mode.name}")
    print(f"FAILURE MODE: {result.failure_mode.type.upper()}")
    print(f"Confidence: {result.confidence_score}")
    if result.hybrid_score is not None:
        print(f"HYBRID SCORE: {result.hybrid_score}")
        print(f"  > LLM Confidence: {result.llm_confidence}")
        print(f"  > Retrieval Conf (RC): {result.retrieval_confidence:.4f}")
        print(f"  > Cross-Encoder Conf (ReC): {result.cross_encoder_confidence:.4f}")
    print(f"Damage Level: {result.damage_level} ({result.severity})")
    print("-" * 50)
    print(result.reasoning)
    print("-" * 50)
    print("Citations:")
    for citation in result.citations:
        print(f"- [{citation.section_id}] {citation.title}")
        print(f"  Snippet: {citation.text_snippet}")
    print("-" * 50)

import time

def test_agent():
    parser = argparse.ArgumentParser(description="Test the Structural Agent with specific inputs or run default test cases.")
    parser.add_argument("--input", type=str, help="JSON string representing a single observation.")
    parser.add_argument("--file", type=str, help="Path to a JSON file containing an observation or a list of observations.")
    args = parser.parse_args()

    print("Initializing Structural Agent...")
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not found. Cannot run test.")
        return

    retriever = SectionRetriever()
    llm_service = GeminiLLMService(api_key=api_key)
    agent = StructuralAgent(retriever, llm_service)

    observations = []

    if args.input:
        try:
            obs = json.loads(args.input)
            observations.append(obs)
            print("Loaded observation from CLI input.")
        except json.JSONDecodeError as e:
            print(f"Error parsing input JSON: {e}")
            return
    elif args.file:
        try:
            with open(args.file, 'r') as f:
                content = json.load(f)
                if isinstance(content, list):
                    observations.extend(content)
                elif isinstance(content, dict):
                    observations.append(content)
                else:
                    print("Error: JSON file must contain a dictionary (single observation) or a list of dictionaries.")
                    return
            print(f"Loaded {len(observations)} observation(s) from file: {args.file}")
        except FileNotFoundError:
            print(f"Error: File not found: {args.file}")
            return
        except json.JSONDecodeError as e:
            print(f"Error parsing file JSON: {e}")
            return

    if observations:
        # Run tests for provided observations
        for i, obs in enumerate(observations):
            print(f"\n--- Custom Test Case {i+1} ---")
            print(f"Input: {json.dumps(obs, indent=2)}")
            result = agent.diagnose(obs)
            if result:
                print_diagnosis_result(result)
            else:
                print("No diagnosis found.")
            
            # Respect rate limits
            if i < len(observations) - 1:
                print("Waiting 35 seconds to request rate limits...")
                time.sleep(35)
    else:
        # Default Hardcoded Test Cases (Backward Compatibility)
        print("\nNo custom input provided. Running default test cases...")

        # Test Case 1: Diagonal Tension
        print("\n--- Test Case 1: Diagonal Tension (Target: Shear Failure) ---")
        obs1 = {
            "material": "URM",
            "orientation": "Diagonal",
            "width": "5mm",
            "location": "Center of Pier",
            "description": "Step-pattern cracks along mortar joints"
        }
        
        result1 = agent.diagnose(obs1)
        if result1:
            print_diagnosis_result(result1)
        else:
            print("No diagnosis found.")

        # Test Case 2: Wall-Pier Rocking
        print("\n--- Test Case 2: Wall-Pier Rocking ---")
        obs2 = {
            "material": "URM",
            "orientation": "Diagonal",
            "location": "Pier",
                "description": "Cracks open and close, no significant shear displacement",
            "width": "5mm" # Adding width for severity calculation
        }

        result2 = agent.diagnose(obs2)
        if result2:
            print_diagnosis_result(result2)
        else: 
            print("No diagnosis found.")

if __name__ == "__main__":
    test_agent()
