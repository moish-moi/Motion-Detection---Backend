# pipeline/visualization_manager.py
import cv2
import numpy as np
import time

def overlay_heatmap(background_image, heatmap_colored, alpha=0.6):
    return cv2.addWeighted(background_image, 1 - alpha, heatmap_colored, alpha, 0)

def overlay_mask_on_image(image, mask, color=(0, 0, 255), alpha=0.4):
    colored_mask = np.zeros_like(image)
    colored_mask[:] = color
    mask_boolean = mask > 0
    overlay = image.copy()
    overlay[mask_boolean] = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)[mask_boolean]
    return overlay

def run_judges_presentation(first_frame, heatmap_overlay, heat_mask, mask_overlay):
    """
    מנהלת את שלבי התצוגה לשופטים אחד אחרי השני בחלון יחיד
    """
    def draw_text(img, text):
        res = img.copy()
        cv2.putText(res, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return res

    window_name = "System Logic Presentation"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 9000, 600)

    # שלב 1: מקור
    cv2.imshow(window_name, draw_text(first_frame, "Step 1: Input Scene"))
    cv2.waitKey(50500)

    # שלב 2: מפת חום
    cv2.imshow(window_name, draw_text(heatmap_overlay, "Step 2: Learning Routine Hotspots"))
    cv2.waitKey(50500)
    # שלב 3: מסכה בינארית
    mask_3d = cv2.cvtColor(heat_mask, cv2.COLOR_GRAY2BGR)
    cv2.imshow(window_name, draw_text(mask_3d, "Step 3: Binary Decision Mask"))
    cv2.waitKey(50500)

    # שלב 4: תוצאה משולבת
    cv2.imshow(window_name, draw_text(mask_overlay, "Step 4: Routine Filter Active"))
    print("Presentation finished. Press any key to exit.")
    cv2.waitKey(0)