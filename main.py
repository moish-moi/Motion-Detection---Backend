# main.py
from pipeline.data_loader import load_image_sequences
from pipeline.event_manager import extract_all_events
from pipeline.heatmap_generator import build_heatmap
from pipeline.mask_generator import heatmap_to_mask
from pipeline.visualization_manager import overlay_heatmap, overlay_mask_on_image, run_judges_presentation

import cv2
import numpy as np

# ONLINE
from app.pipeline import filter_event

# ADAPTER (OFFLINE -> ONLINE)
from offline_to_online_adapter import (
    normalize_mask_255_to_01,
    offline_events_to_online_events,
)


def main():
    # --- הגדרות נתיבים ---
    learning_data_root = "./frames/Trees"
    inference_data_root = "./frames/Inference" # התיקייה החדשה לזיהוי
    # alert_endpoint = "https://webhook.site/your-unique-id" # כתובת ה-API שלך

    # 1+2+3+4) שלב הלמידה (ללא שינוי)
    print("STEP 1: Learning Routine from Factory...")
    sequences_learning = load_image_sequences(learning_data_root)
    all_learning_events = extract_all_events(sequences_learning)
    
    first_frame = cv2.imread(sequences_learning[0]["path"])
    image_shape = first_frame.shape
    H, W = image_shape[0], image_shape[1]

    heatmap = build_heatmap(all_learning_events, image_shape)
    heat_mask_255 = heatmap_to_mask(heatmap, threshold_factor=0.4)
    heat_mask_01 = normalize_mask_255_to_01(heat_mask_255)
    # 5) ויזואליזציה (לשופטים)
    normalized_heatmap = (heatmap / (np.max(heatmap) + 1e-6) * 255).astype(np.uint8)
    colored_heatmap = cv2.applyColorMap(normalized_heatmap, cv2.COLORMAP_JET)

    h_overlay = overlay_heatmap(first_frame, colored_heatmap)
    m_overlay = overlay_mask_on_image(first_frame, heat_mask_255)

    run_judges_presentation(first_frame, h_overlay, heat_mask_255, m_overlay)

    # =========================
    # 6) ONLINE: חיבור מלא Offline -> Online
    # =========================
    print(f"\nSTEP 2: Monitoring new frames in {inference_data_root}...")
    sequences_inference = load_image_sequences(inference_data_root)
    raw_inference_events = extract_all_events(sequences_inference)
    online_events = offline_events_to_online_events(raw_inference_events)

    config = {
        "min_box_area": int(0.001 * W * H),
        "per_box_threshold": 0.60,
        "T_mean": 0.60,
        "T_ratio": 0.70,
        "enable_min_overlap_safety": True,
        "safety_only_when_multiple_boxes": True,
        "min_overlap_any_box_forward": 0.10,
    }

    forwarded_count = 0
    
    # משתמשים ב-enumerate כדי לקבל את המיקום (i) של האירוע
    for i, ev in enumerate(online_events):
        decision = filter_event(
            ev, 
            heat_mask_01, 
            config, 
            time_score=None, 
            auto_push=True
        )

        if decision["label"] == "FORWARD":
            forwarded_count += 1
            print(f"🚨 [SUSPICIOUS] Event {ev.event_id} at {ev.timestamp}")
            print(f"   Reason: {decision['reason']}")
            
            # --- התיקון כאן ---
            # אנחנו לוקחים את הנתיב ישירות מהמילון המקורי שלך לפי אותו אינדקס
            image_path_from_raw = raw_inference_events[i]["source_path"]
            
            img = cv2.imread(image_path_from_raw)
            if img is not None:
                # מציירים את המלבנים שהשותף סינן (הם בפורמט xyxy)
                for box in ev.boxes: 
                    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
                
                cv2.imshow("Suspicious Activity", img)
                cv2.waitKey(50000) 
        else:
            print(f"✅ [ROUTINE] Event {ev.event_id} ignored.")

if __name__ == "__main__":
    main()