# pipeline/motion_detector.py

import cv2
import numpy as np

def detect_motion_between_frames(prev_frame, curr_frame, threshold_val=25, global_rejection_ratio=0.5):
    # 1. המרה לגווני אפור
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    # 2. טשטוש
    prev_blur = cv2.GaussianBlur(prev_gray, (21, 21), 0)
    curr_blur = cv2.GaussianBlur(curr_gray, (21, 21), 0)

    # 3. חיסור
    frame_delta = cv2.absdiff(prev_blur, curr_blur)

    # 4. יצירת סף (Threshold)
    _, thresh = cv2.threshold(frame_delta, threshold_val, 255, cv2.THRESH_BINARY)

    # --- השינוי החדש: מנגנון ה-Global Rejection ---
    motion_pixels = np.count_nonzero(thresh)
    total_pixels = thresh.shape[0] * thresh.shape[1]
    change_ratio = motion_pixels / total_pixels
    
    if change_ratio > global_rejection_ratio:
        # אם יותר מחצי פריים השתנה - זה רעש תאורה. נחזיר מסכה ריקה.
        return np.zeros_like(thresh)
    # --------------------------------------------

    # 5. ניקוי מורפולוגי (המשך הקוד המקורי שלך)
    kernel_small = np.ones((3,3), np.uint8)
    thresh = cv2.erode(thresh, kernel_small, iterations=1)
    
    kernel_large = np.ones((5,5), np.uint8)
    thresh = cv2.dilate(thresh, kernel_large, iterations=4)
    
    # 6. הרחבה סופית
    thresh = cv2.dilate(thresh, None, iterations=2)

    return thresh