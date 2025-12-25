# main.py

from pipeline.data_loader import load_image_sequences
from pipeline.event_manager import extract_all_events
from pipeline.heatmap_generator import build_heatmap
from pipeline.mask_generator import heatmap_to_mask
from pipeline.visualization_manager import overlay_heatmap  # הייבוא החדש
import cv2
import numpy as np

def main():
    data_root = "./frames"
    print("Loading image sequences...")
    sequences = load_image_sequences(data_root)
    
    if not sequences:
        return

    # 1. עיבוד ואיסוף אירועים
    print("Processing events...")
    all_events = extract_all_events(sequences)
    
    first_frame = cv2.imread(sequences[0]["path"])
    if first_frame is None:
        print("Could not read first frame.")
        return
    # 2. בניית מפת החום
    image_shape = first_frame.shape # עכשיו אנחנו לוקחים את ה-shape מהתמונה שטענו    heatmap = build_heatmap(all_events, image_shape)
    # 2. בניית מפת החום
    heatmap = build_heatmap(all_events, image_shape)
    heat_mask = heatmap_to_mask(heatmap)

    # 3. הכנת התצוגה הויזואלית
    if np.max(heatmap) > 0:
        # יצירת המפה הצבעונית
        normalized_heatmap = (heatmap / np.max(heatmap) * 255).astype(np.uint8)
        colored_heatmap = cv2.applyColorMap(normalized_heatmap, cv2.COLORMAP_JET)
        
        # יצירת השכבה המשולבת (תמונה מקורית + מפת חום)
        first_frame = cv2.imread(sequences[0]["path"])
        heatmap_on_image = overlay_heatmap(first_frame, colored_heatmap)

        # --- הצגת חלונות ---
        
        # חלון 1: מפת חום על גבי המציאות
        cv2.imshow("1. Heatmap Overlay (Reality vs Motion)", heatmap_on_image)
        
        # חלון 2: המסכה הסופית (מה המחשב יחסום)
        cv2.imshow("2. Final Heat Mask (Ignore Areas)", heat_mask)
        
        # חלון 3: התמונה המקורית (נקייה)
        cv2.imshow("3. Original First Frame", first_frame)

        print("\nReviewing results:")
        print("- Window 1: Shows where motion occurs on the actual scene.")
        print("- Window 2: Shows the binary mask that will be used for filtering.")
        print("\nPress any key to finish.")
        cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()