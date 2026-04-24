from src.modules.ai_camera import IMX500Detector
import time

camera = IMX500Detector()
camera.start(show_preview=True)

while True:
    detections = camera.get_detections()
    labels = camera.get_labels()

    if detections:
        print(f"\n--- {len(detections)} object(s) detected ---")
        for detection in detections:
            label = labels[int(detection.category)]
            confidence = detection.conf

            # Get bounding box coordinates
            # IMX500 typically returns box as (x, y, width, height) or (x1, y1, x2, y2)
            box = detection.box  # or detection.bbox depending on your SDK version

            # If box is (x, y, w, h) — center calculation:
            x = box[0] + box[2] / 2
            y = box[1] + box[3] / 2

            # If box is (x1, y1, x2, y2) — use this instead:
            # x = (box[0] + box[2]) / 2
            # y = (box[1] + box[3]) / 2

            print(f"  {label:<20} conf: {confidence:.2f}  center: ({x:.1f}, {y:.1f})  box: {box}")
    else:
        print("No detections")

    time.sleep(0.1)