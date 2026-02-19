import re
import json
import os
import uuid

INPUT_FILE = "data/processed/fema-306_raw_text.txt"
OUTPUT_DIR = "data/processed/text_chunks"

def chunk_chapter_7():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        content = f.read()

    # Regex to capture Section Number and Title
    # Capture group 1: Section ID (e.g., 7.2.1)
    # Capture group 3: Title (rest of the line)
    # We look for lines starting with "7.2." or similar
    # The previous grep showed "7.2.1 Non-Wall Components"
    section_pattern = re.compile(r'(^|\n)(7\.\d+(\.\d+)?)\s+(.+)')

    lines = content.split('\n')
    
    current_chunk = None
    accumulated_lines = []
    
    chunks_created = 0

    def save_chunk(chunk_data):
        if not chunk_data:
            return
        
        # Clean up content
        chunk_data["content"] = "\n".join(accumulated_lines).strip()
        
        # Create safe ID for filename (keep dots)
        section_id = chunk_data["metadata"]["section_id"]
        
        # Generate ID string: id_of_section_7_2_1
        id_str = f"id_of_section_{section_id.replace('.', '_')}"
        
        # Update the dictionary with the correct key and value
        chunk_data["document_id"] = id_str
        if "id" in chunk_data:
            del chunk_data["id"]
            
        # Create filename: chunk-section-7.2.1.json
        filename = f"chunk-section-{section_id}.json"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(chunk_data, f, indent=2)
        
        nonlocal chunks_created
        chunks_created += 1

    # Initial scan to find the first header? 
    # Or act as if the text before the first header is "Introduction"
    
    # We'll treat the start of the file as an implicit chunk until the first header
    current_chunk = {
        "id": str(uuid.uuid4()), # Will be replaced by document_id in save_chunk
        "title": "Chapter 7 Introduction",
        "content": "",
        "metadata": {"section_id": "7.2.intro", "source": "fema-306_raw_text.txt"}
    }

    for line in lines:
        stripped = line.strip()
        
        match = section_pattern.match(line)
        if match:
            # We found a new section header
            # 1. Save the previous chunk
            save_chunk(current_chunk)
            
            # 2. Start new chunk
            sec_id = match.group(2)
            title = match.group(4).strip()
            
            current_chunk = {
                "id": str(uuid.uuid4()), # Will be replaced
                "title": f"{sec_id} {title}",
                "content": "",
                "metadata": {
                    "section_id": sec_id,
                    "source": "fema-306_raw_text.txt"
                }
            }
            accumulated_lines = []
            
            # Include the header in the content? Usually yes.
            accumulated_lines.append(line)
        else:
            accumulated_lines.append(line)

    # Save the last chunk
    save_chunk(current_chunk)

    print(f"Successfully created {chunks_created} JSON chunk files in {OUTPUT_DIR}")

if __name__ == "__main__":
    chunk_chapter_7()
