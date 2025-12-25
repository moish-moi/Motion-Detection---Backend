# main.py
from pipeline.data_loader import load_image_sequences
from pipeline.event_manager import extract_all_events
from pipeline.heatmap_generator import build_heatmap
from pipeline.mask_generator import heatmap_to_mask
from pipeline.visualization_manager import overlay_heatmap, overlay_mask_on_image, run_judges_presentation
import cv2
import numpy as np

def main():
    data_root = "./frames/Factory"
    
    # 1. טעינה
    print("Loading data...")
    sequences = load_image_sequences(data_root)
    if not sequences: return

    # 2. למידה (חילוץ אירועים)
    print("Processing events...")
    all_events = extract_all_events(sequences)
    
    # 3. הכנת תשתיות (תמונה ראשונה ומידות)
    first_frame = cv2.imread(sequences[0]["path"])
    image_shape = first_frame.shape

    # 4. יצירת מפת חום ומסכת סינון
    print("Building intelligence...")
    heatmap = build_heatmap(all_events, image_shape)
    heat_mask = heatmap_to_mask(heatmap)

    # 5. הכנת תוצרי ויזואליזציה
    normalized_heatmap = (heatmap / (np.max(heatmap) + 1e-6) * 255).astype(np.uint8)
    colored_heatmap = cv2.applyColorMap(normalized_heatmap, cv2.COLORMAP_JET)
    
    h_overlay = overlay_heatmap(first_frame, colored_heatmap)
    m_overlay = overlay_mask_on_image(first_frame, heat_mask)

    # 6. הרצת המצגת לשופטים
    run_judges_presentation(first_frame, h_overlay, heat_mask, m_overlay)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()