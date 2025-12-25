# pipeline/motion_detector.py

import cv2
import numpy as np

def detect_motion_between_frames(prev_frame, curr_frame, threshold_val=25):
    """
    מזהה שינויים בין שתי תמונות עוקבות.
    :param prev_frame: התמונה הקודמת
    :param curr_frame: התמונה הנוכחית
    :param threshold_val: רגישות - ככל שהמספר נמוך יותר, המערכת תזהה תנועות קטנות יותר
    :return: תמונה שחורה-לבנה (Mask) שבה הלבן מייצג תנועה
    """
    # 1. המרה לגווני אפור (עיבוד מהיר יותר ופחות מושפע מרעשי צבע)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    # 2. טשטוש קל (Gaussian Blur) כדי להפחית רעשים דיגיטליים של המצלמה
    prev_blur = cv2.GaussianBlur(prev_gray, (21, 21), 0)
    curr_blur = cv2.GaussianBlur(curr_gray, (21, 21), 0)

    # 3. חיסור בין התמונות - מוצא את ההבדל המוחלט בין הפיקסלים
    frame_delta = cv2.absdiff(prev_blur, curr_blur)

    # 4. יצירת סף (Threshold) - כל מה שמעל ערך מסוים הופך ללבן (255), כל השאר שחור (0)
    _, thresh = cv2.threshold(frame_delta, threshold_val, 255, cv2.THRESH_BINARY)
    # 1. שחיקה (Erosion) - מוחק פיקסלים לבנים בודדים (רעש)
    kernel_small = np.ones((3,3), np.uint8)
    thresh = cv2.erode(thresh, kernel_small, iterations=1)
    # 2. הרחבה (Dilation) - מחבר חלקים קרובים של אובייקט אחד
    kernel_large = np.ones((5,5), np.uint8)
    thresh = cv2.dilate(thresh, kernel_large, iterations=4) # הגדלנו איטרציות
    # 5. הרחבת האזורים הלבנים (Dilate) כדי לחבר נקודות קטנות לאובייקטים ברורים
    thresh = cv2.dilate(thresh, None, iterations=2)

    return thresh