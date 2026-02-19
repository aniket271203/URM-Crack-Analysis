import json
import os
import chromadb
from sentence_transformers import SentenceTransformer
import glob

# Try both locations, prioritize the processed one if it has files
INPUT_DIR = "data/processed/text_chunks"
# Fallback or check if user moved them
INPUT_DIR_ALT = "data/text_chunks"

DB_PATH = "data/chroma_db"
COLLECTION_NAME = "fema_306"

def create_embeddings():
    # Determine input directory
    if os.path.exists(INPUT_DIR) and os.listdir(INPUT_DIR):
        target_dir = INPUT_DIR
    elif os.path.exists(INPUT_DIR_ALT) and os.listdir(INPUT_DIR_ALT):
        target_dir = INPUT_DIR_ALT
    else:
        print(f"Error: No chunk directories found at {INPUT_DIR} or {INPUT_DIR_ALT}")
        return

    print(f"Loading chunks from {target_dir}...")
    
    # Get all JSON files
    chunk_files = glob.glob(os.path.join(target_dir, "*.json"))
    
    if not chunk_files:
        print("No JSON files found.")
        return

    chunks = []
    for cf in chunk_files:
        with open(cf, 'r', encoding='utf-8') as f:
            try:
                chunks.append(json.load(f))
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON: {cf}")

    print(f"Loaded {len(chunks)} chunks.")

    # Initialize model
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Initialize ChromaDB
    print(f"Initializing ChromaDB at {DB_PATH}...")
    client = chromadb.PersistentClient(path=DB_PATH)
    
    # Delete collection if exists to start fresh
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection '{COLLECTION_NAME}'")
    except:
        pass
        
    collection = client.create_collection(name=COLLECTION_NAME)

    ids = []
    documents = []
    metadatas = []
    embedding_texts = []
    
    for chunk in chunks:
        # Use document_id if available, else id, else title
        c_id = chunk.get("document_id") or chunk.get("id") or chunk.get("title")
        content = chunk.get("content", "")
        
        # Prepare two versions: 
        # 1. stored_doc: The JSON string (for the app to parse)
        # 2. semantic_text: The rich text (for the model to embed)
        
        stored_doc = content
        semantic_text = content
        
        # Fallback for structured chunks (legacy format without 'content' field)
        if not content:
            # 1. Stored Doc: Pure JSON
            stored_doc = json.dumps(chunk, indent=2)
            
            # 2. Semantic Text: Rich description for retrieval
            # Extract key fields explicitly to avoid JSON noise
            title = chunk.get("title", "")
            system = chunk.get("system", "")
            mode = chunk.get("behavior_mode", "")
            
            guidelines = ""
            if "identification_guidelines" in chunk:
                g = chunk["identification_guidelines"]
                guidelines = f"Identification: {g.get('by_observation', '')} {g.get('by_analysis', '')}"
            
            criteria = ""
            if "damage_levels" in chunk:
                # Collect criteria from all levels
                all_criteria = []
                for lvl in chunk["damage_levels"]:
                    c_list = lvl.get("criteria", [])
                    all_criteria.extend(c_list)
                    desc = lvl.get("typical_appearance_desc", "")
                    if desc: all_criteria.append(desc)
                criteria = "Damage Criteria: " + " ".join(all_criteria)
                
            desc_field = chunk.get("description", "")
            
            # Combine into a dense semantic string
            semantic_text = f"{title}. System: {system}. Mode: {mode}. {desc_field} {guidelines} {criteria}"
            
        if not c_id:
            print(f"Skipping chunk without ID: {chunk.keys()}")
            continue
            
        ids.append(c_id)
        documents.append(stored_doc)       # Store JSON
        embedding_texts.append(semantic_text) # Embed Rich Text
        
        # Prepare metadata (flat dict only)
        # copy relevant fields
        meta = {
            "title": chunk.get("title", ""),
            "section_id": chunk.get("metadata", {}).get("section_id", ""),
            "source": chunk.get("metadata", {}).get("source", ""),
            "id": c_id  # Explicitly store ID for retrieval
        }
        metadatas.append(meta)

    # Generate embeddings
    print("Generating embeddings from semantic text...")
    embeddings = model.encode(embedding_texts)
    
    # Add to ChromaDB
    print("Adding to vector database...")
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings.tolist()
    )

    print(f"Successfully added {len(ids)} documents to collection '{COLLECTION_NAME}'.")

if __name__ == "__main__":
    create_embeddings()

