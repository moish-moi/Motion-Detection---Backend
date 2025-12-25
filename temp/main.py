# main.py
from app import config
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
    heat_mask_255 = heatmap_to_mask(heatmap, threshold_factor=0.7)
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

    # --- שלב ב': התאמה ל-EVENT של השותף ---
    # זה הופך את הרשימה שלך לרשימה של אובייקטי Event
    online_events = offline_events_to_online_events(raw_inference_events)

    # --- שלב ג': קריאה ל-filter_event לזיהוי ---
    for ev in online_events:
        # כאן קורה הקסם: השותף בודק את האירוע שלך מול המסיכה שיצרת
        decision = filter_event(
            event=ev,
            heat_mask=heat_mask_01, # המסיכה שלך (0 ו-1)
            config=config,
            auto_push=True          # זה כבר ישלח את ההתראה אם זה FORWARD
        )

        # הדפסת התוצאה למסך
        print(f"Event {ev.event_id}: {decision['label']} - {decision['reason']}")
    # raw_inference_events = extract_all_events(sequences_inference)
    # online_events = offline_events_to_online_events(raw_inference_events)
    
    # for i, ev in enumerate(online_events):
    #     # שולחים לזיהוי - זה יעבוד מעולה על סמך המסיכה
    #     decision = filter_event(ev, heat_mask_01, config)
    
    #     if decision["label"] == "FORWARD":
    #         # כאן אנחנו צריכים את התמונה כדי להראות אותה!
    #         # אנחנו לוקחים את הנתיב מהרשימה המקורית שלנו לפי האינדקס i
    #         original_path = raw_inference_events[i]["path"]
            
    #         print(f"🚨 תנועה חשודה בפריים: {original_path}")
            
    #         # עכשיו אפשר לפתוח את התמונה ולהראות אותה
    #         img = cv2.imread(original_path)
    #         cv2.imshow("Suspicious", img)
    #         cv2.waitKey(500)
    
    
    # # # הגדרות ה-Config שלך
    # # config = {
    # #     "min_box_area": int(0.001 * W * H),
    # #     "per_box_threshold": 0.60,
    # #     "T_mean": 0.60,
    # #     "T_ratio": 0.70,
    # #     "enable_min_overlap_safety": True,
    # #     "safety_only_when_multiple_boxes": True,
    # #     "min_overlap_any_box_forward": 0.10,
    # # }

    # # # מעבר על התמונות החדשות וחיפוש תנועה מול המסכה שלמדנו
    # # # אנחנו נשתמש ב-extract_all_events על התיקייה החדשה כדי למצוא תנועה גולמית
    # # forwarded_count = 0
    # # for ev in online_events:
    # #     # כאן קורה החיבור לשותף:
    # #     decision = filter_event(
    # #         ev, 
    # #         heat_mask_01, 
    # #         config, 
    # #         time_score=None, 
    # #         auto_push=True
    # #     )

    # #     # בדיקת ההחלטה
    # #     if decision["label"] == "FORWARD":
    # #         forwarded_count += 1
    # #         print(f"🚨 [SUSPICIOUS] Event {ev.event_id} at {ev.timestamp}")
    # #         print(f"   Reason: {decision['reason']}")
            
    # #         # הצגת התמונה החשודה לצרכי הדגמה
    # #         img = cv2.imread(ev.source_path)
    # #         if img is not None:
    # #             for box in ev.boxes: # קואורדינטות xyxy
    # #                 cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
    # #             cv2.imshow("Suspicious Activity", img)
    # #             cv2.waitKey(500) # השהייה של חצי שנייה לראות את התוצאה
    # #     else:
    # #         print(f"✅ [ROUTINE] Event {ev.event_id} ignored.")

    # # print(f"\nFinal Report: {forwarded_count} suspicious events detected.")
    # # cv2.waitKey(0)
    # # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()