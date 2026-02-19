import os

# Path to your dataset folder
folder = "../DATA_bricks"

# List all files
files = sorted(os.listdir(folder))

# Loop and rename
for i, filename in enumerate(files, start=1):
    old_path = os.path.join(folder, filename)
    
    # Extract extension (e.g. ".jpg")
    ext = os.path.splitext(filename)[1]
    
    # New name
    new_filename = f"brick_{i}{ext}"
    new_path = os.path.join(folder, new_filename)
    
    # Rename
    os.rename(old_path, new_path)

print("✅ Renaming completed!")
