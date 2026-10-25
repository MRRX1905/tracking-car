"""Antarmuka Tkinter: preview stream, pengaturan garis, panel hitungan, log."""

from __future__ import annotations

import csv
import json
import queue
import threading
import time
from pathlib import Path

import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from config import AppConfig
from vehicle_counter.counter import LineCrossingCounter
from vehicle_counter.detector import VehicleDetector, draw_overlay
from vehicle_counter.stream import StreamSource, open_source

LINES_PATH = Path(__file__).with_name("lines.json")
SQUARE = 9  # ukuran pegangan endpoint garis


class App(tk.Tk):
    def __init__(self, cfg: AppConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.title("Penghitung Kendaraan — YOLOv8 + OpenCV")
        self.geometry("1280x800")
        self.minsize(1000, 640)

        self.counter = LineCrossingCounter()
        self.detector: VehicleDetector | None = None
        self.source: StreamSource | None = None
        self.connected = False

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._frame_q: queue.Queue = queue.Queue(maxsize=1)
        self._event_q: queue.Queue = queue.Queue()
        self._fps = 0.0
        self._status_text = "Siap. Masukkan URL lalu tekan Buka Stream."
        self._conn_state = "idle"  # idle | loading | on  (di-set semua thread)
        self._ui_state = ""
        self._pending_error: str | None = None
        self._raw_frame = None  # frame asli tanpa overlay (untuk refresh statis)
        self._display_w = 0
        self._display_h = 0
        self._photo = None
        self._pending_pt: tuple[float, float] | None = None
        self._cx = self._cy = 0
        self._drag: tuple[int, bool] | None = None  # (line_id, is_p2)
        self._selected_id: int | None = None
        self.mode = tk.StringVar(value="select")
        self._last_counts: dict[str, int] = {}

        self._build()
        self._load_lines()
        self._refresh_static()
        self.after(60, self._poll)

    # ------------------------------------------------------------- layout
    def _build(self) -> None:
        top = ttk.Frame(self, padding=6)
        top.pack(fill="x")
        ttk.Label(top, text="URL stream (.m3u8 / RTSP / file):").pack(side="left")
        self.url_var = tk.StringVar(value=self.cfg.source)
        ttk.Entry(top, textvariable=self.url_var, width=60).pack(
            side="left", padx=(4, 8), fill="x", expand=True
        )
        self.btn_open = ttk.Button(top, text="Buka Stream", command=self.open_stream)
        self.btn_open.pack(side="left", padx=2)
        self.btn_stop = ttk.Button(top, text="Berhenti", command=self.close_stream,
                                   state="disabled")
        self.btn_stop.pack(side="left", padx=2)
        ttk.Button(top, text="Reset Hitung", command=self.reset_counts).pack(
            side="left", padx=(10, 2)
        )
        ttk.Button(top, text="Simpan Gambar", command=self.save_shot).pack(side="left", padx=2)
        ttk.Button(top, text="Ekspor CSV", command=self.export_csv).pack(side="left", padx=2)

        body = ttk.Frame(self, padding=6)
        body.pack(fill="both", expand=True)

        # Preview
        left = ttk.Frame(body)
        left.pack(side="left", fill="both", expand=True)
        self.canvas = tk.Canvas(left, bg="#101010", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Motion>", self._on_move)

        # Panel kanan
        right = ttk.Frame(body, width=300)
        right.pack(side="right", fill="y", padx=(8, 0))
        right.pack_propagate(False)

        box = ttk.LabelFrame(right, text="Hitungan", padding=8)
        box.pack(fill="x")
        self.lbl_total = ttk.Label(box, text="0", font=("Segoe UI", 40, "bold"),
                                   anchor="center")
        self.lbl_total.pack(fill="x")
        ttk.Label(box, text="TOTAL KENDARAAN", anchor="center").pack(fill="x")
        self.frm_classes = ttk.Frame(box)
        self.frm_classes.pack(fill="x", pady=(8, 0))
        self._class_labels: dict[str, ttk.Label] = {}

        tools = ttk.LabelFrame(right, text="Garis Penghitung", padding=8)
        tools.pack(fill="x", pady=(8, 0))
        ttk.Radiobutton(tools, text="Pilih / geser garis", value="select",
                        variable=self.mode).pack(anchor="w")
        ttk.Radiobutton(tools, text="Tambah garis baru", value="add",
                        variable=self.mode).pack(anchor="w")
        btns = ttk.Frame(tools)
        btns.pack(fill="x", pady=(6, 0))
        ttk.Button(btns, text="Hapus garis terpilih",
                   command=self._delete_selected).pack(fill="x")
        ttk.Button(btns, text="Hapus semua garis",
                   command=self._clear_lines).pack(fill="x", pady=(4, 0))
        ttk.Button(btns, text="Garis default",
                   command=self._default_lines).pack(fill="x", pady=(4, 0))

        sens = ttk.LabelFrame(right, text="Parameter", padding=8)
        sens.pack(fill="x", pady=(8, 0))
        self.conf_var = tk.DoubleVar(value=self.cfg.conf)
        ttk.Label(sens, text="Confidence threshold").pack(anchor="w")
        ttk.Scale(sens, from_=0.05, to=0.9, variable=self.conf_var,
                  command=self._on_conf).pack(fill="x")

        log_box = ttk.LabelFrame(right, text="Log penyeberangan", padding=4)
        log_box.pack(fill="both", expand=True, pady=(8, 0))
        cols = ("waktu", "garis", "arah", "kelas", "id", "conf")
        self.tree = ttk.Treeview(log_box, columns=cols, show="headings", height=8)
        titles = ("Waktu", "Garis", "Arah", "Kelas", "ID", "Conf")
        widths = (66, 78, 58, 78, 40, 44)
        for c, t, w in zip(cols, titles, widths):
            self.tree.heading(c, text=t)
            self.tree.column(c, width=w, anchor="center")
        vsb = ttk.Scrollbar(log_box, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        status = ttk.Frame(self, padding=(6, 2))
        status.pack(fill="x", side="bottom")
        self.lbl_status = ttk.Label(status, text="Siap. Masukkan URL lalu tekan Buka Stream.")
        self.lbl_status.pack(side="left")

    # -------------------------------------------------------- koneksi
    def open_stream(self) -> None:
        url = self.url_var.get().strip()
        if not url:
            messagebox.showwarning("URL kosong", "Isi URL stream .m3u8 terlebih dahulu.")
            return
        self._conn_state = "loading"
        self._set_status("Membuka stream & memuat model...")
        threading.Thread(target=self._connect, args=(url,), daemon=True).start()

    def _connect(self, url: str) -> None:
        try:
            if self.detector is None:
                self.detector = VehicleDetector(self.cfg)
            src = open_source(url, self.cfg.reconnect_delay)
            self.source, self.connected = src, True
            self.cfg.source = url
            self.cfg.save()
            self.counter.reset()
            self._stop.clear()
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()
            self._status_text = f"Terhubung ({src.backend})"
            self._conn_state = "on"
        except Exception as exc:  # noqa: BLE001 - pesan error ditampilkan ke UI
            self._pending_error = str(exc)
            self._conn_state = "idle"

    def close_stream(self) -> None:
        self._stop.set()
        self.connected = False
        self._conn_state = "idle"
        t, self._thread = self._thread, None
        if t:
            t.join(timeout=1.5)
        if self.source:
            self.source.release()
            self.source = None
        self._set_status("Dihentikan.")
        self._refresh_static()

    def reset_counts(self) -> None:
        self.counter.reset()
        self._last_counts = {}
        self._render_counts()
        for i in self.tree.get_children():
            self.tree.delete(i)
        self._refresh_static()

    def _set_status(self, text: str) -> None:
        """Aman dipanggil dari thread mana pun; label disinkronkan _poll()."""
        self._status_text = text

    # ----------------------------------------------------------- worker
    def _worker(self) -> None:
        while not self._stop.is_set():
            src = self.source
            if src is None:
                break
            ok, frame = src.read()
            if not ok:
                if not src.ensure_open():
                    self._set_status("Koneksi terputus, mencoba ulang...")
                    time.sleep(0.2)
                    continue
                continue

            t0 = time.perf_counter()
            dets = self.detector.track(frame)
            for ev in self.counter.update(dets):
                self._event_q.put(ev)
            dt = time.perf_counter() - t0
            inst = 1.0 / dt if dt > 0 else 0.0
            self._fps = inst if self._fps == 0 else self._fps * 0.85 + inst * 0.15

            drawn = draw_overlay(
                frame, dets, self.counter.lines, self._selected_id,
                dict(self.counter.counts), self._fps, self._status_text,
            )
            small = self._resize(drawn)
            self._raw_frame = frame
            self._put_frame(small)
    def _put_frame(self, img) -> None:
        if self._frame_q.full():
            try:
                self._frame_q.get_nowait()
            except queue.Empty:
                pass
        try:
            self._frame_q.put_nowait(img)
        except queue.Full:
            pass

    def _resize(self, frame):
        h, w = frame.shape[:2]
        max_w = self.cfg.max_display_width
        if w <= max_w:
            return frame
        scale = max_w / w
        return cv2.resize(frame, (max_w, int(h * scale)), interpolation=cv2.INTER_AREA)

    # ------------------------------------------------------------ poll
    def _poll(self) -> None:
        self._sync_ui()
        try:
            while True:
                ev = self._event_q.get_nowait()
                self._insert_event(ev)
        except queue.Empty:
            pass

        latest = None
        while True:
            try:
                latest = self._frame_q.get_nowait()
            except queue.Empty:
                break
        if latest is not None:
            self._show(latest)

        counts = dict(self.counter.counts)
        if counts != self._last_counts:
            self._last_counts = counts
            self._render_counts()
        self.after(50, self._poll)

    def _sync_ui(self) -> None:
        """Sinkronkan widget dengan state yang di-set thread lain."""
        if self.lbl_status.cget("text") != self._status_text:
            self.lbl_status.config(text=self._status_text)

        want = (
            "disabled" if self._conn_state == "loading" else "normal",
            "normal" if self._conn_state == "on" else "disabled",
        )
        if self._ui_state != want:
            self._ui_state = want
            self.btn_open.config(state=want[0])
            self.btn_stop.config(state=want[1])

        if self._pending_error:
            msg, self._pending_error = self._pending_error, None
            messagebox.showerror("Gagal membuka", msg)

    def _insert_event(self, ev) -> None:
        self.tree.insert("", 0, values=ev.as_row())
        kids = self.tree.get_children()
        if len(kids) > 500:
            self.tree.delete(kids[-1])

    def _render_counts(self) -> None:
        counts = self._last_counts
        self.lbl_total.config(text=str(sum(counts.values())))
        names = sorted(set(list(counts) + [n for n in self._class_names()]))
        for name in names:
            lbl = self._class_labels.get(name)
            if lbl is None:
                lbl = ttk.Label(self.frm_classes, text="", font=("Segoe UI", 11))
                lbl.pack(fill="x", anchor="w")
                self._class_labels[name] = lbl
            lbl.config(text=f"{name:<12} {counts.get(name, 0):>4}")

    def _class_names(self) -> list[str]:
        from config import COCO_VEHICLE_CLASSES
        return [COCO_VEHICLE_CLASSES[c] for c in self.cfg.classes
                if c in COCO_VEHICLE_CLASSES]

    def _show(self, img) -> None:
        h, w = img.shape[:2]
        if (w, h) != (self._display_w, self._display_h):
            self._display_w, self._display_h = w, h
            self.canvas.config(width=w, height=h)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self._photo = ImageTk.PhotoImage(image=Image.fromarray(rgb))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo, tags="img")
        if self._pending_pt and self.mode.get() == "add":
            x1, y1 = self._pending_pt[0] * w, self._pending_pt[1] * h
            self.canvas.create_line(x1, y1, self._cx, self._cy,
                                    fill="#00ff00", dash=(4, 3), tags="ghost")

    def _refresh_static(self) -> None:
        """Render ulang preview bila stream berhenti."""
        import numpy as np
        if self._raw_frame is None:
            frame = np.full((540, 960, 3), 28, np.uint8)
        else:
            frame = self._raw_frame.copy()
        drawn = draw_overlay(frame, [], self.counter.lines, self._selected_id,
                             dict(self.counter.counts))
        self._show(self._resize(drawn))

    # -------------------------------------------------------- interaksi
    def _to_norm(self, event) -> tuple[float, float]:
        w = max(1, self.canvas.winfo_width())
        h = max(1, self.canvas.winfo_height())
        return (min(max(event.x / w, 0.0), 1.0), min(max(event.y / h, 0.0), 1.0))

    def _near_endpoint(self, p) -> tuple[int, bool] | None:
        thr = SQUARE * 1.6 / max(1, self.canvas.winfo_width())
        for line in self.counter.lines:
            for is_p2, end in ((False, line.p1), (True, line.p2)):
                if ((end[0] - p[0]) ** 2 + (end[1] - p[1]) ** 2) ** 0.5 < thr:
                    return line.id, is_p2
        return None

    def _on_click(self, event) -> None:
        p = self._to_norm(event)
        if self.mode.get() == "add":
            if self._pending_pt is None:
                self._pending_pt = p
            else:
                line = self.counter.add_line(self._pending_pt, p)
                self._pending_pt = None
                self._selected_id = line.id
                self._save_lines()
                self._set_status(f"Garis {line.label_or_id} ditambahkan.")
            self._refresh_static()
            return
        hit = self._near_endpoint(p)
        if hit:
            self._drag = hit
            self._selected_id = hit[0]
            self._refresh_static()
            return
        near = self.counter.nearest_line(p, 0.04)
        self._selected_id = near.id if near else None
        self._refresh_static()
        if near is None:
            self._set_status("Tidak ada garis terpilih.")

    def _on_drag(self, event) -> None:
        if not self._drag:
            return
        line_id, is_p2 = self._drag
        p = self._to_norm(event)
        for line in self.counter.lines:
            if line.id == line_id:
                if is_p2:
                    line.p2 = p
                else:
                    line.p1 = p
        self._refresh_static()

    def _on_release(self, _event) -> None:
        if self._drag:
            self._drag = None
            self._save_lines()

    def _on_move(self, event) -> None:
        self._cx, self._cy = event.x, event.y
        if self.mode.get() == "add" and self._pending_pt is not None and self._photo:
            w, h = self._display_w, self._display_h
            x1, y1 = self._pending_pt[0] * w, self._pending_pt[1] * h
            self.canvas.delete("ghost")
            self.canvas.create_line(x1, y1, event.x, event.y,
                                    fill="#00ff00", dash=(4, 3), tags="ghost")

    def _on_conf(self, value: str) -> None:
        try:
            self.cfg.conf = round(float(value), 2)
        except ValueError:
            return
        if self.detector is not None:
            self.detector.cfg.conf = self.cfg.conf

    def _delete_selected(self) -> None:
        if self._selected_id is None:
            messagebox.showinfo("Tidak ada garis", "Klik garis terlebih dahulu.")
            return
        self.counter.remove_line(self._selected_id)
        self._selected_id = None
        self._save_lines()
        self._refresh_static()

    def _clear_lines(self) -> None:
        self.counter.clear_lines()
        self._selected_id = None
        self._save_lines()
        self._refresh_static()

    def _default_lines(self) -> None:
        self._clear_lines()
        self.counter.add_line((0.05, 0.62), (0.95, 0.62), "Utama")
        self._save_lines()
        self._refresh_static()

    def _save_lines(self) -> None:
        try:
            LINES_PATH.write_text(
                json.dumps([l.to_dict() for l in self.counter.lines], indent=2),
                encoding="utf-8",
            )
        except OSError:
            pass

    def _load_lines(self) -> None:
        if not LINES_PATH.exists():
            self._default_lines()
            return
        try:
            data = json.loads(LINES_PATH.read_text(encoding="utf-8"))
            from vehicle_counter.counter import Line
            for d in data:
                self.counter.lines.append(Line.from_dict(d))
            if self.counter.lines:
                self.counter._next_line_id = max(l.id for l in self.counter.lines) + 1
            else:
                self._default_lines()
        except (json.JSONDecodeError, OSError, KeyError, ValueError):
            self._default_lines()

    # ------------------------------------------------------------- ekspor
    def save_shot(self) -> None:
        if self._raw_frame is None:
            messagebox.showinfo("Belum ada gambar", "Buka stream terlebih dahulu.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".jpg", filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")]
        )
        if not path:
            return
        frame = draw_overlay(self._raw_frame.copy(), [], self.counter.lines,
                             self._selected_id, dict(self.counter.counts))
        cv2.imwrite(path, frame)
        self._set_status(f"Gambar disimpan: {path}")

    def export_csv(self) -> None:
        if not self.counter.events:
            messagebox.showinfo("Kosong", "Belum ada penyeberangan untuk diekspor.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV", "*.csv")]
        )
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.writer(fh)
            writer.writerow(self.counter.HEADERS)
            for ev in self.counter.events:
                writer.writerow(ev.as_row())
        self._set_status(f"CSV disimpan: {path}")

    def on_close(self) -> None:
        self._stop.set()
        self.connected = False
        if self.source:
            self.source.release()
        self.cfg.save()
        self.destroy()


def run_app(cfg: AppConfig) -> None:
    app = App(cfg)
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()
