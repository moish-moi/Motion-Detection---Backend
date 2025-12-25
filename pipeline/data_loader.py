# pipeline/data_loader.py

from pathlib import Path
from datetime import datetime

def load_image_sequences(root_dir: str):
    root_path = Path(root_dir)
    sequence = []

    if not root_path.exists():
        print(f"❌ שגיאה: התיקייה {root_path.absolute()} לא נמצאה")
        return []

    time_folders = sorted([f for f in root_path.iterdir() if f.is_dir()])
    
    for folder in time_folders:
        try:
            timestamp = datetime.strptime(folder.name, "%Y%m%d_%H%M%S")
        except ValueError:
            continue

        image_files = sorted(list(folder.glob("*.jpg")))
        
        for img_path in image_files:
            # שים לב: הורדנו את cv2.imread. אנחנו שומרים רק את הנתיב!
            sequence.append({
                "path": str(img_path),
                "timestamp": timestamp
            })

    return sequence