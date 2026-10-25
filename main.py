#!/usr/bin/env python3
"""Penghitung kendaraan dari stream video (.m3u8 / RTSP / file / webcam).

Contoh:
    python main.py                                  # GUI
    python main.py --url https://host/live.m3u8      # GUI + URL siap
    python main.py --headless --url https://host/live.m3u8 --duration 30
"""

from __future__ import annotations

import argparse
import csv
import sys
import time

from config import AppConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Penghitung kendaraan YOLOv8 + OpenCV")
    p.add_argument("--url", default="", help="URL .m3u8, RTSP, path file, atau angka webcam")
    p.add_argument("--headless", action="store_true", help="Tanpa GUI, cetak ringkasan")
    p.add_argument("--duration", type=int, default=30, help="Detik (mode headless)")
    p.add_argument("--out", default="count_report.csv", help="CSV laporan mode headless")
    p.add_argument("--model", default="", help="Path model YOLO (default yolov8n.pt)")
    p.add_argument("--conf", type=float, default=-1, help="Confidence threshold")
    return p.parse_args()


def run_headless(cfg: AppConfig, duration: int, out_path: str) -> int:
    from vehicle_counter.counter import LineCrossingCounter
    from vehicle_counter.detector import VehicleDetector, draw_overlay
    from vehicle_counter.stream import open_source

    import cv2

    print(f"Membuka: {cfg.source}")
    src = open_source(cfg.source, cfg.reconnect_delay)
    print(f"Backend: {src.backend}, ukuran: {src.size}")

    det = VehicleDetector(cfg)
    counter = LineCrossingCounter()
    if not counter.lines:
        counter.add_line((0.05, 0.62), (0.95, 0.62), "Utama")

    deadline = time.time() + duration
    frames = 0
    t_start = time.perf_counter()
    last = None
    while time.time() < deadline:
        ok, frame = src.read()
        if not ok:
            if not src.ensure_open():
                time.sleep(0.2)
            continue
        frames += 1
        dets = det.track(frame)
        events = counter.update(dets)
        for ev in events:
            print(f"  +1 {ev.class_name:<11} {ev.direction:<5} {ev.line_label} #{ev.track_id}")
        last = draw_overlay(frame, dets, counter.lines, counts=counter.counts)
        if frames % 30 == 0:
            print(f"  frame={frames} fps={frames / max(1e-6, time.perf_counter() - t_start):.1f} "
                  f"total={counter.total} deteksi_aktif={len(dets)}")

    src.release()
    elapsed = time.perf_counter() - t_start
    with open(out_path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(counter.HEADERS)
        for ev in counter.events:
            w.writerow(ev.as_row())

    print("\n=== RINGKASAN ===")
    print(f"Frame diproses : {frames}")
    print(f"Durasi         : {elapsed:.1f} dtk ({frames / max(1e-6, elapsed):.1f} fps)")
    for name, n in sorted(counter.counts.items()):
        print(f"{name:<14} : {n}")
    print(f"{'TOTAL':<14} : {counter.total}")
    print(f"Log            : {out_path}")

    if last is not None:
        cv2.imwrite("last_frame.jpg", last)
        print("Snapshot       : last_frame.jpg")
    return 0


def main() -> int:
    args = parse_args()
    cfg = AppConfig().load()
    if args.url:
        cfg.source = args.url
    if args.model:
        cfg.model_path = args.model
    if args.conf >= 0:
        cfg.conf = args.conf

    if args.headless:
        if not cfg.source:
            print("Error: --url wajib pada mode headless.", file=sys.stderr)
            return 2
        return run_headless(cfg, args.duration, args.out)

    if not cfg.source:
        print("Info: belum ada URL. Masukkan lewat GUI atau --url.", file=sys.stderr)
    from vehicle_counter.gui import run_app
    run_app(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
