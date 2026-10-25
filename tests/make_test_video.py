"""Buat video uji: crop kendaraan dari bus.jpg, gerakkan melewati garis."""

import sys
from pathlib import Path

import cv2
import numpy as np
import ultralytics
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
bus_path = Path(ultralytics.__file__).parent / "assets" / "bus.jpg"
img = cv2.imread(str(bus_path))

model = YOLO("yolov8n.pt")
res = model.predict(img, conf=0.4, verbose=False)[0]
boxes = sorted(res.boxes.xyxy.tolist(), key=lambda b: b[2] - b[0], reverse=True)
print("kandidat objek:", [(int(b[2] - b[0]), int(b[3] - b[1])) for b in boxes[:3]])

x1, y1, x2, y2 = (int(v) for v in boxes[0])
crop = img[y1:y2, x1:x2]
ch, cw = crop.shape[:2]
print("crop kendaraan:", cw, "x", ch)

W, H = 960, 540
bg = cv2.resize(img, (W, H)).copy()
bg = cv2.GaussianBlur(bg, (0, 0), 25)
bg = (bg.astype(np.float32) * 0.35 + 40).astype(np.uint8)  # redupkan latar

scale = 380 / cw
crop_s = cv2.resize(crop, (380, max(1, int(ch * scale))))
ch_s, cw_s = crop_s.shape[:2]

frames = []
n = 70
for i in range(n):
    t = i / (n - 1)
    # dari atas (y=-ch) ke bawah (y=H), melewati garis y=0.62*H=334
    y = int(-ch_s + t * (H + ch_s))
    x = 300
    frame = bg.copy()
    # tempel dengan clipping agar aman saat keluar bidang
    x0, y0 = max(0, x), max(0, y)
    x1c, y1c = min(W, x + cw_s), min(H, y + ch_s)
    if x1c > x0 and y1c > y0:
        sy, sx = y0 - y, x0 - x
        frame[y0:y1c, x0:x1c] = crop_s[sy:sy + (y1c - y0), sx:sx + (x1c - x0)]
    frames.append(frame)

out = "moving_traffic.mp4"
vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 20, (W, H))
for f in frames:
    vw.write(f)
vw.release()

# verifikasi YOLO masih mendeteksi objek yang digeser
det = model.track(np.array(frames[35]), conf=0.3, persist=True, verbose=False,
                  classes=[2, 3, 5, 7])
ids = det[0].boxes.id
print("cek frame-35 track id:", None if ids is None else ids.tolist())
print("file:", out)
