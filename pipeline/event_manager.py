# pipeline/event_manager.py

import cv2
import os
from datetime import datetime
from PIL import Image
from PIL.ExifTags import TAGS
from .motion_detector import detect_motion_between_frames
from .geometry_utils import extract_contours_and_boxes

def get_image_timestamp(image_path):
    """חילוץ זמן הצילום מה-EXIF של התמונה"""
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()
        if exif_data:
            for tag, value in exif_data.items():
                tag_name = TAGS.get(tag, tag)
                if tag_name == 'DateTimeOriginal' or tag_name == 'DateTime':
                    # פורמט EXIF סטנדרטי הוא "YYYY:MM:DD HH:MM:SS"
                    return datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
    except Exception:
        pass
    
    # גיבוי: אם אין EXIF, קח את זמן שינוי הקובץ מהמערכת
    mtime = os.path.getmtime(image_path)
    return datetime.fromtimestamp(mtime)

def extract_all_events(sequences):
    all_events = []
    if not sequences: return all_events

    prev_frame = cv2.imread(sequences[0]["path"])
    
    for i in range(1, len(sequences)):
        img_path = sequences[i]["path"]
        curr_frame = cv2.imread(img_path)
        if curr_frame is None: continue
        
        # חילוץ הזמן מהתמונה עצמה
        timestamp = get_image_timestamp(img_path)
        
        mask = detect_motion_between_frames(prev_frame, curr_frame)
        boxes = extract_contours_and_boxes(mask)
        
        if boxes:
            all_events.append({
                "timestamp": timestamp,
                "boxes": boxes
            })
            
        prev_frame = curr_frame
        
        if i % 50 == 0:
            print(f"Processed {i}/{len(sequences)} frames (Latest Time: {timestamp})...")
            
    return all_events