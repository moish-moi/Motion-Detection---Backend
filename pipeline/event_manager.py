# pipeline/event_manager.py

import cv2
from .motion_detector import detect_motion_between_frames
from .geometry_utils import extract_contours_and_boxes

def extract_all_events(sequences):
    all_events = []
    
    # טוענים את הפריים הראשון מראש
    prev_frame = cv2.imread(sequences[0]["path"])
    
    for i in range(1, len(sequences)):
        curr_frame = cv2.imread(sequences[i]["path"])
        if curr_frame is None: continue
        
        timestamp = sequences[i]["timestamp"]
        
        # עיבוד
        mask = detect_motion_between_frames(prev_frame, curr_frame)
        boxes = extract_contours_and_boxes(mask)
        
        if boxes:
            all_events.append({
                "timestamp": timestamp,
                "boxes": boxes
            })
            
        # חשוב מאוד: הפריים הנוכחי הופך לקודם, והזיכרון של הקודם משתחרר
        prev_frame = curr_frame
        
        # הדפסת התקדמות קלה כדי שלא תחשוב שזה נתקע
        if i % 10 == 0:
            print(f"Processed {i}/{len(sequences)} frames...")
            
    return all_events