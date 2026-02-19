import chromadb
import re

DB_PATH = "data/chroma_db"
COLLECTION_NAME = "fema_306"

def get_tags(id, title):
    tags = {
        "is_failure_mode": False,
        "mode_type": "General" # Default
    }

    # Identify Failure Modes based on keywords in title
    # Relevant for Chapter 7 URM modes
    # "Toe" covers Toe Crushing
    failure_keywords = [
        "Sliding", "Rocking", "Cracking", "Failure", "Buckling", 
        "Strut", "Yielding", "Slip", "Crushing", "Toe", "Tension"
    ]
    
    # Check if any keyword matches
    if any(k.lower() in title.lower() for k in failure_keywords):
        tags["is_failure_mode"] = True
        
        # Assign specific mode type
        if "Sliding" in title: tags["mode_type"] = "Sliding"
        elif "Rocking" in title: tags["mode_type"] = "Rocking"
        elif "Cracking" in title: tags["mode_type"] = "Cracking"
        elif "Buckling" in title: tags["mode_type"] = "Buckling"
        elif "Shear" in title: tags["mode_type"] = "Shear"
        elif "Flexural" in title: tags["mode_type"] = "Flexure"
        elif "Toe" in title: tags["mode_type"] = "Toe Failure"
        elif "Tension" in title: tags["mode_type"] = "Tension"
    
    return tags

def apply_tags():
    print(f"Connecting to ChromaDB at {DB_PATH}...")
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(COLLECTION_NAME)
    
    # Get all current data
    result = collection.get()
    ids = result['ids']
    metadatas = result['metadatas']
    
    print(f"Scanning {len(ids)} documents for tagging...")
    
    updated_ids = []
    updated_metadatas = []
    
    count_tagged = 0
    
    for i, id in enumerate(ids):
        # Handle duplicates with suffixes (e.g. 7.2.5_1) for logic, but keep ID same
        current_meta = metadatas[i]
        
        # Use existing metadata if available, else derive from ID/Title
        original_id = current_meta.get('section_id', id.split('_')[0])
        title = current_meta.get('title', '')

        new_tags = get_tags(original_id, title)
        
        # Merge tags
        current_meta.update(new_tags)
        
        # Remove old irrelevant keys if they exist (optional, but cleaner)
        if "material" in current_meta:
            del current_meta["material"]
        if "component_type" in current_meta and current_meta["component_type"] == "General":
             # We might want to keep component_type if it came from the structured chunks
             # But if it was added by previous run of this script as "General", maybe remove or ignore.
             pass

        updated_ids.append(id)
        updated_metadatas.append(current_meta)
        
        if new_tags["is_failure_mode"]:
            count_tagged += 1
            # print(f"Tagged {id} ({title}) -> {new_tags['mode_type']}")

    print(f"Updating metadata for {len(updated_ids)} documents...")
    print(f"Identified {count_tagged} failure mode sections.")
    
    collection.update(
        ids=updated_ids,
        metadatas=updated_metadatas
    )
    print("Metadata update complete.")

if __name__ == "__main__":
    apply_tags()
