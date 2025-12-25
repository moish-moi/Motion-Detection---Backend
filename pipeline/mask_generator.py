# pipeline/mask_generator.py

import cv2
import numpy as np

def heatmap_to_mask(heatmap, threshold_factor=0.2):
    """
    הופכת מפת חום למסכה בינארית (Mask).
    :param heatmap: מטריצת החום המצטברת
    :param threshold_factor: אחוז מהערך המקסימלי שמעליו אזור נחשב 'שגרתי'
    :return: מסכה שבה 255 (לבן) זה אזור שגרתי ו-0 (שחור) זה אזור נקי
    """
    # 1. מציאת הערך המקסימלי במפה (האזור הכי פעיל)
    max_val = np.max(heatmap)
    
    if max_val == 0:
        return np.zeros_like(heatmap, dtype=np.uint8)

    # 2. קביעת רף חיתוך (למשל: כל מה שזז יותר מ-20% מהמקסימום)
    limit = max_val * threshold_factor
    
    # 3. יצירת המסכה: 255 איפה שיש תנועה שגרתית, 0 איפה שלא
    _, binary_mask = cv2.threshold(heatmap, limit, 255, cv2.THRESH_BINARY)
    
    return binary_mask.astype(np.uint8)