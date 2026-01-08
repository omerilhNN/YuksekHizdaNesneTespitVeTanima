import cv2
import os
import torch
from ultralytics import YOLO

# Check for CUDA device
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU (Slow!)")

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# 1) Modeli yükle
model_path = os.path.join(script_dir, "best.pt")
model = YOLO(model_path)   # senin .pt dosyan

video_path = os.path.join(script_dir, "input.mp4")
output_path = os.path.join(script_dir, "output.mp4")

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 2) Her frame üzerinde tahmin
    results = model(frame, verbose=False)[0]

    # 3) Araç kutularını çiz
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]

        # Sadece araç sınıflarını istiyorsan burada filtreleyebilirsin
        # if cls_name not in ["car", "truck", "bus"]: continue

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{cls_name} {conf:.2f}"
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("detections", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()
