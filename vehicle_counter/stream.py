"""Pembaca sumber video: .m3u8 (HLS), RTSP, file, webcam.

Dua backend:
  cv2     - OpenCV VideoCapture (ffmpeg build), untuk .m3u8/RTSP/file/webcam
  ffmpeg  - fallback subprocess pipe rawvideo, untuk stream yang gagal dibuka cv2
"""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np


def is_hls(url: str) -> bool:
    return ".m3u8" in url.split("?")[0].lower()


def _probe_size(url: str) -> tuple[int, int] | None:
    """Ambil lebar/tinggi via ffprobe. None bila ffprobe tidak ada/gagal."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    cmd = [
        ffprobe, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "csv=p=0", url,
    ]
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=20, creationflags=0x08000000
        ).stdout
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2 and all(parts[:2]):
                return int(parts[0]), int(parts[1])
        return None
    except (ValueError, OSError, subprocess.SubprocessError):
        return None


class StreamSource:
    """Buka stream, baca frame, dan reconnect otomatis bila putus."""

    def __init__(self, url: str, reconnect_delay: float = 2.0) -> None:
        self.url = url
        self.reconnect_delay = reconnect_delay
        self.backend = "cv2"
        self.size: tuple[int, int] | None = None
        self._cap: cv2.VideoCapture | None = None
        self._proc: subprocess.Popen | None = None
        self._frame_bytes = 0
        self._last_fail = 0.0
        self._consecutive_fail = 0

    # ---------------------------------------------------------------- buka
    def open(self) -> bool:
        self.close()
        if self._try_cv2():
            return True
        if is_hls(self.url) and self._try_ffmpeg():
            return True
        return False

    def _try_cv2(self) -> bool:
        try:
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        except cv2.error:
            return False
        if not cap.isOpened():
            cap.release()
            return False
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except cv2.error:
            pass
        ok, _ = cap.read()
        if not ok:
            cap.release()
            return False
        self._cap = cap
        self.backend = "cv2"
        self.size = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None,
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None,
        )
        return True

    def _try_ffmpeg(self) -> bool:
        if not shutil.which("ffmpeg"):
            return False
        size = _probe_size(self.url)
        if not size:
            return False
        w, h = size
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-fflags", "+genpts", "-i", self.url,
            "-an", "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
        ]
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8
            )
        except OSError:
            return False
        frame_len = w * h * 3
        chunk = proc.stdout.read(frame_len)
        if not chunk or len(chunk) != frame_len:
            proc.terminate()
            return False
        self._proc, self._frame_bytes, self.size = proc, frame_len, (w, h)
        self.backend = "ffmpeg"
        return True

    # --------------------------------------------------------------- baca
    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.backend == "cv2" and self._cap is not None:
            ok, frame = self._cap.read()
            if ok and frame is not None:
                self._consecutive_fail = 0
                return True, frame
        elif self.backend == "ffmpeg" and self._proc is not None:
            raw = self._proc.stdout.read(self._frame_bytes)
            if raw and len(raw) == self._frame_bytes:
                w, h = self.size
                self._consecutive_fail = 0
                return True, np.frombuffer(raw, np.uint8).reshape(h, w, 3).copy()
            self._proc = None

        self._consecutive_fail += 1
        self._last_fail = time.time()
        return False, None

    def ensure_open(self) -> bool:
        """Reconnect bila koneksi putus."""
        if self.backend == "cv2" and self._cap is not None and self._cap.isOpened():
            return True
        if self.backend == "ffmpeg" and self._proc is not None:
            if self._proc.poll() is None:
                return True
            self._proc = None
        if self._consecutive_fail == 0:
            return False
        if time.time() - self._last_fail < self.reconnect_delay:
            return False
        self._last_fail = time.time()
        return self.open()

    def release(self) -> None:
        self.close()

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=3)
            except (OSError, subprocess.SubprocessError):
                self._proc.kill()
            self._proc = None
        self._consecutive_fail = 0


def open_source(url: str, reconnect_delay: float = 2.0) -> StreamSource:
    """Buka URL atau path file. Bila path lokal relatif, konversi ke absolut."""
    if url and not url.lower().startswith(("http://", "https://", "rtsp://", "rtmp://")):
        p = Path(url)
        if p.exists():
            url = str(p.resolve())
    src = StreamSource(url, reconnect_delay)
    if not src.open():
        raise RuntimeError(
            f"Gagal membuka sumber: {url}\n"
            "Periksa URL/akses jaringan, atau pastikan ffmpeg ada di PATH."
        )
    return src
