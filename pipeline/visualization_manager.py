# pipeline/visualization_manager.py

import cv2

def overlay_heatmap(background_image, heatmap_colored, alpha=0.6):
    """
    מניחה את מפת החום הצבעונית על התמונה המקורית בשקיפות.
    :param background_image: התמונה המקורית (BGR)
    :param heatmap_colored: מפת החום הצבעונית (BGR)
    :param alpha: מידת השקיפות של המפה (0 עד 1)
    :return: תמונה משולבת
    """
    # חיבור שתי התמונות: background * (1-alpha) + heatmap * alpha
    overlay = cv2.addWeighted(background_image, 1 - alpha, heatmap_colored, alpha, 0)
    return overlay