import sys
import os

# Add project root to path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.retriever import SectionRetriever

def main():
    print("Initializing Retriever (loading embeddings)...")
    try:
        retriever = SectionRetriever()
    except Exception as e:
        print(f"Error initializing retriever: {e}")
        return

    print("\n" + "="*50)
    print("FEMA 306 Digital Inspector - Manual Query CLI")
    print("="*50)
    print("Type 'exit' or 'quit' to stop.")
    print("Type 'filters:on' to toggle failure mode filtering (Default: OFF).")
    
    filtering = False
    
    while True:
        mode_str = "[Failure Modes Only]" if filtering else "[All Sections]"
        query = input(f"\nEnter query {mode_str}: ").strip()
        
        if query.lower() in ['exit', 'quit']:
            break
            
        if query.lower() == 'filters:on':
            filtering = not filtering
            print(f"Filtering set to: {filtering}")
            continue
            
        if not query:
            continue
            
        filters = {"is_failure_mode": True} if filtering else None
        
        print(f"Searching for '{query}'...")
        results = retriever.query(query, n_results=3, filters=filters)
        
        if not results:
            print("No results found.")
        else:
            for i, doc in enumerate(results):
                title = doc.metadata.get('title', 'Unknown Title')
                # ID should now be present in metadata
                sec_id = doc.metadata.get('id', 'Unknown ID')
                
                tags = []
                if doc.metadata.get('is_failure_mode'):
                    tags.append(f"FAILURE MODE ({doc.metadata.get('mode_type')})")
                if doc.metadata.get('material'):
                    tags.append(doc.metadata.get('material'))

                tag_str = f" | Tags: {', '.join(tags)}" if tags else ""
                
                print(f"\n{i+1}. [{sec_id}] {title}{tag_str}")
                
                try:
                    import json
                    data = json.loads(doc.page_content)
                    
                    # Check for Guide format
                    if "damage_levels" in data:
                        print("   [Structured Guide Identified]")
                        print(f"   System: {data.get('system')} | Mode: {data.get('behavior_mode')}")
                        for lvl in data.get("damage_levels", [])[:1]: 
                            print(f"   Sample Level ({lvl.get('level')}): {lvl.get('criteria', [])[0] if lvl.get('criteria') else 'No criteria'}")
                        print("   ... (more levels available)")
                    
                    # Check for Procedure format (e.g. 7.3.2)
                    elif "description" in data:
                        print("   [Structured Procedure Identified]")
                        print(f"   Description: {data.get('description')}")
                        print(f"   Steps: {', '.join([k for k in data.keys() if k.startswith('step_')])}")
                    
                    else:
                        print(f"   [JSON Content]: {str(list(data.keys()))}")
                except:
                    # Text chunk
                    content_preview = doc.page_content.replace(title, '').strip()
                    print(f"   Snippet: {content_preview[:150]}...")


if __name__ == "__main__":
    main()
