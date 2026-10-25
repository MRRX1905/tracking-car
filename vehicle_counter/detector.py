"""Deteksi + pelacakan kendaraan dengan YOLOv8 (ultralytics, open source)."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import COCO_VEHICLE_CLASSES, AppConfig  # noqa: E402

VEHICLE_COLORS = {
    "car": (0, 200, 255),
    "motorcycle": (255, 120, 0),
    "bus": (0, 255, 120),
    "truck": (255, 0, 160),
}


@dataclass
class Detection:
    track_id: int
    xyxy: tuple[float, float, float, float]
    centroid: tuple[float, float]  # ternormalisasi 0..1
    name: str
    conf: float


class VehicleDetector:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.model = YOLO(cfg.model_path)
        self._names: dict[int, str] = dict(self.model.names)
        self._device = cfg.device or ""

    def track(self, frame: np.ndarray) -> list[Detection]:
        """Deteksi + ByteTrack. persist=True menjaga ID antar frame."""
        kwargs = dict(
            persist=True,
            verbose=False,
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            imgsz=self.cfg.imgsz,
            tracker=self.cfg.tracker,
            classes=self.cfg.classes,
        )
        if self._device:
            kwargs["device"] = self._device

        results = self.model.track(frame, **kwargs)
        h, w = frame.shape[:2]
        out: list[Detection] = []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0 or r.boxes.id is None:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            conf = r.boxes.conf.cpu().numpy()
            ids = r.boxes.id.cpu().numpy().astype(int)
            for box, c, cf, tid in zip(xyxy, cls, conf, ids):
                x1, y1, x2, y2 = (float(v) for v in box)
                name = self._names.get(int(c), COCO_VEHICLE_CLASSES.get(int(c), str(c)))
                out.append(
                    Detection(
                        track_id=int(tid),
                        xyxy=(x1, y1, x2, y2),
                        centroid=((x1 + x2) / 2.0 / w, (y1 + y2) / 2.0 / h),
                        name=name,
                        conf=float(cf),
                    )
                )
        return out


def draw_overlay(
    frame: np.ndarray,
    detections: list[Detection],
    lines,
    selected_line_id: int | None = None,
    counts: dict[str, int] | None = None,
    fps: float = 0.0,
    status: str = "",
) -> np.ndarray:
    """Gambar garis, kotak deteksi, dan HUD ke frame."""
    h, w = frame.shape[:2]

    for line in lines:
        p1 = (int(line.p1[0] * w), int(line.p1[1] * h))
        p2 = (int(line.p2[0] * w), int(line.p2[1] * h))
        color = (0, 220, 255) if line.id == selected_line_id else (0, 255, 0)
        cv2.line(frame, p1, p2, color, 2, cv2.LINE_AA)
        for p in (p1, p2):
            cv2.circle(frame, p, 5, color, -1, cv2.LINE_AA)
        mx, my = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
        cv2.putText(frame, line.label_or_id, (mx + 6, my - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det.xyxy)
        color = VEHICLE_COLORS.get(det.name, (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        px, py = int(det.centroid[0] * w), int(det.centroid[1] * h)
        cv2.circle(frame, (px, py), 3, (0, 0, 255), -1, cv2.LINE_AA)
        label = f"#{det.track_id} {det.name} {det.conf:.2f}"
        cv2.putText(frame, label, (x1, max(14, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    hud_y = 26
    if counts:
        cv2.putText(frame, f"TOTAL: {sum(counts.values())}", (10, hud_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        hud_y += 24
        for name, n in sorted(counts.items()):
            cv2.putText(frame, f"{name}: {n}", (10, hud_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
            hud_y += 20
    if fps:
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 130, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
    if status:
        cv2.putText(frame, status, (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    return frame
