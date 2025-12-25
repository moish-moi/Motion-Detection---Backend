# pipeline/heatmap_generator.py

import numpy as np

def build_heatmap(events, image_shape):
    """
    בונה מפת חום על סמך רשימת אירועים.
    :param events: רשימת הדיקשנריז של האירועים
    :param image_shape: גובה ורוחב התמונה (height, width)
    :return: מטריצת numpy עם ערכי הצטברות
    """
    # יצירת מטריצה ריקה בגודל התמונה
    heatmap = np.zeros(image_shape[:2], dtype=np.float32)
    
    for event in events:
        for (x, y, w, h) in event["boxes"]:
            # הוספת "חום" לאזור הקופסה במטריצה
            # אנחנו מוסיפים 1 לכל פיקסל בתוך המלבן
            heatmap[y:y+h, x:x+w] += 1
            
    return heatmap