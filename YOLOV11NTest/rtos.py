import cv2
from ultralytics import YOLO
import torch

# =============================
#  AYARLAR
# =============================
MODEL_PATH = "best.pt"   # kendi .pt modelinin yolu
CAMERA_INDEX = 0          # varsayılan webcam = 0
CONF_THRESHOLD = 0.4      # minimum güven skoru

# =============================
#  MODELİ YÜKLE
# =============================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = YOLO(MODEL_PATH)
model.to(device)

# Sadece araç sınıflarını istiyorsan burada listele (yoksa None bırak)
# Örnek: COCO için ['car', 'truck', 'bus', 'motorbike']
VEHICLE_CLASSES = ['car', 'truck', 'bus', 'motorbike']  # None yaparsan bütün sınıfları gösterir

# =============================
#  KAMERA AÇ
# =============================
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise RuntimeError("Kamera açılamadı, CAMERA_INDEX değerini kontrol et.")

# =============================
#  MAIN LOOP
# =============================
while True:
    ret, frame = cap.read()
    if not ret:
        print("Kameradan frame okunamadı, çıkılıyor.")
        break

    # -------------------------
    #  MODEL ÇALIŞTIR
    # -------------------------
    # not: Ultralytics içinde otomatik olarak GPU/CPU seçiyor,
    #      model.to(device) ile zaten device'a taşımış olduk.
    results = model(frame, verbose=False)[0]

    # -------------------------
    #  SONUÇLARI ÇİZ
    # -------------------------
    for box in results.boxes:
        conf = float(box.conf[0])
        if conf < CONF_THRESHOLD:
            continue

        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]

        # Sadece araç sınıfları
        if VEHICLE_CLASSES is not None and cls_name not in VEHICLE_CLASSES:
            continue

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{cls_name} {conf:.2f}"
        cv2.putText(
            frame, label, (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
        )

    # -------------------------
    #  GÖRÜNTÜYÜ GÖSTER
    # -------------------------
    cv2.imshow("Real-time Vehicle Detection", frame)

    # 'q' tuşuna basınca çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =============================
#  TEMİZLİK
# =============================
cap.release()
cv2.destroyAllWindows()
