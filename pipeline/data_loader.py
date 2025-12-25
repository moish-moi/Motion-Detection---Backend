# pipeline/data_loader.py

from pathlib import Path

def load_image_sequences(root_dir: str):
    root_path = Path(root_dir)
    sequence = []

    if not root_path.exists():
        print(f"❌ Error: Folder {root_path.absolute()} not found")
        return []

    # אוסף את כל קבצי ה-jpg מכל התיקיות הפנימיות
    image_files = sorted(list(root_path.glob("**/*.jpg")))
    
    for img_path in image_files:
        sequence.append({
            "path": str(img_path)
        })

    return sequence