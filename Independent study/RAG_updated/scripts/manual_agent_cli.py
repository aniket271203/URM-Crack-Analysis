import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.retriever import SectionRetriever
from src.llm_service import GeminiLLMService
from src.structural_agent import StructuralAgent

def main():
    print("Initializing Structural Agent...")
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not found. Please checks your .env file or environment variables.")
        return

    try:
        retriever = SectionRetriever()
        llm_service = GeminiLLMService(api_key=api_key)
        agent = StructuralAgent(retriever, llm_service)
        print("Agent initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize agent: {e}")
        return

    print("\n=== FEMA 306 Structural Failure Mode Diagnostics CLI ===")
    print("Enter the requested crack observations. Press Ctrl+C to exit at any time.\n")

    try:
        while True:
            print("-" * 50)
            observations = {}
            
            # Helper to get input with optional default
            def get_input(prompt, default=None):
                p = f"{prompt} (default: {default}): " if default else f"{prompt}: "
                val = input(p).strip()
                return val if val else default

            observations['material'] = get_input("Material", "URM")
            observations['component'] = get_input("Component Type (e.g., Pier, Wall, Spandrel)", "Pier")
            observations['orientation'] = get_input("Crack Orientation (e.g., Diagonal, Horizontal, Vertical)")
            observations['width'] = get_input("Crack Width (e.g., 5mm, hairline)")
            observations['location'] = get_input("Crack Location (e.g., Center of Pier, Top/Bottom)")
            observations['description'] = get_input("Additional Description")

            # Remove empty fields
            observations = {k: v for k, v in observations.items() if v}

            if len(observations) < 2:
                print("\n[!] Please provide at least concrete details (Orientation, etc.) for a good diagnosis.")
                continue

            print("\nAnalyzing...")
            result = agent.diagnose(observations)

            if result:
                print(f"\n>>> DIAGNOSIS REPORT")
                print(f"Confidence Score: {result.confidence_score}")
                print("-" * 50)
                print(result.reasoning)
                print("-" * 50)
                print("Citations:")
                for citation in result.citations:
                    print(f"- [{citation.section_id}] {citation.title}")
                    print(f"  Snippet: {citation.text_snippet}")
                print("-" * 50)
                                
                print(f"\n--- Retrieved Content ({result.failure_mode.id}) ---")
                print(result.failure_mode.description)
                print("-------------------------")
            else:
                print("\n>>> No diagnosis could be determined with high confidence.")

            print("\n")
            
    except KeyboardInterrupt:
        print("\nExiting CLI. Goodbye!")

if __name__ == "__main__":
    main()
