"""Konfigurasi penghitung kendaraan. Disimpan ke/dimuat dari config.json."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

CONFIG_PATH = Path(__file__).with_name("config.json")

# Kelas COCO yang dihitung sebagai kendaraan.
COCO_VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


@dataclass
class AppConfig:
    source: str = "https://cctv-stream.bandaacehkota.info/memfs/0b8d9f6d-5b2c-4c6d-a645-438cdf8cdd18.m3u8"
    model_path: str = "yolov8n.pt"
    conf: float = 0.35
    iou: float = 0.45
    imgsz: int = 640
    device: str = ""
    classes: list[int] = field(default_factory=lambda: sorted(COCO_VEHICLE_CLASSES))
    tracker: str = "bytetrack.yaml"
    max_display_width: int = 960
    reconnect_delay: float = 2.0

    def load(self) -> "AppConfig":
        if CONFIG_PATH.exists():
            try:
                data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
                for key, value in data.items():
                    if hasattr(self, key):
                        setattr(self, key, value)
            except (json.JSONDecodeError, OSError):
                pass
        return self

    def save(self) -> None:
        CONFIG_PATH.write_text(
            json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8"
        )
