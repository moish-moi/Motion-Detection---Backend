# pipeline/geometry_utils.py

import cv2

def extract_contours_and_boxes(motion_mask, min_area=500):
    """
    מחלץ מלבנים חוסמים מתוך מסיכת התנועה.
    :param motion_mask: התמונה השחורה-לבנה שקיבלנו מה-Motion Detector
    :param min_area: שטח מינימלי (בפיקסלים) כדי להתעלם מרעשים
    :return: רשימה של מלבנים [(x, y, w, h), ...]
    """
    # 1. מציאת קווי מתאר (Contours)
    # RETR_EXTERNAL - מוצא רק את הקונטורים החיצוניים (לא חורים בתוך אובייקטים)
    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bounding_boxes = []
    
    for cnt in contours:
        # 2. חישוב שטח הקונטור
        area = cv2.contourArea(cnt)
        
        # 3. סינון רעשים - רק אם הכתם גדול מספיק
        if area > min_area:
            # הפיכת הקונטור למלבן (Bounding Box)
            x, y, w, h = cv2.boundingRect(cnt)
            bounding_boxes.append((x, y, w, h))
            
    return bounding_boxes

def draw_boxes(image, boxes, color=(0, 255, 0), thickness=2):
    """
    פונקציית עזר לציור המלבנים על התמונה לצורך תצוגה
    """
    img_copy = image.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), color, thickness)
    return img_copy