# pipeline/viewer.py

import cv2
from .motion_detector import detect_motion_between_frames
from .geometry_utils import extract_contours_and_boxes, draw_boxes # ייבוא החדשים

def run_interactive_viewer(sequences):
    if not sequences:
        return

    idx = 1
    
    while True:
        curr_data = sequences[idx]
        prev_data = sequences[idx - 1]

        # 1. זיהוי תנועה
        motion_mask = detect_motion_between_frames(prev_data["image"], curr_data["image"])

        # 2. חילוץ קופסאות (הצעד החדש שלנו!)
        boxes = extract_contours_and_boxes(motion_mask, min_area=800)

        # 3. ציור הקופסאות על הפריים המקורי
        frame_with_boxes = draw_boxes(curr_data["image"], boxes)

        # תצוגה
        title = f"Frame {idx} | Boxes: {len(boxes)}"
        cv2.imshow("Review Mode - Original", frame_with_boxes)
        cv2.imshow("Review Mode - Motion Mask", motion_mask)
        cv2.setWindowTitle("Review Mode - Original", title)

        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d') and idx < len(sequences) - 1:
            idx += 1
        elif key == ord('a') and idx > 1:
            idx -= 1

    cv2.destroyAllWindows()