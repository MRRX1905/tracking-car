"""Logika penghitungan kendaraan: penyeberangan garis virtual (line crossing)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

Point = tuple[float, float]  # koordinat ternormalisasi 0..1 terhadap frame


@dataclass
class Line:
    """Garis penghitung. Titik dalam koordinat ternormalisasi (0..1)."""

    id: int
    p1: Point
    p2: Point
    label: str = ""

    def side(self, p: Point) -> float:
        """Tanda cross product. >0 di satu sisi, <0 di sisi lain."""
        return (self.p2[0] - self.p1[0]) * (p[1] - self.p1[1]) - (
            self.p2[1] - self.p1[1]
        ) * (p[0] - self.p1[0])

    @property
    def label_or_id(self) -> str:
        return self.label or f"garis-{self.id}"

    def to_dict(self) -> dict:
        return {"id": self.id, "p1": list(self.p1), "p2": list(self.p2), "label": self.label}

    @classmethod
    def from_dict(cls, d: dict) -> "Line":
        return cls(int(d["id"]), tuple(d["p1"]), tuple(d["p2"]), d.get("label", ""))


@dataclass
class CrossingEvent:
    timestamp: float
    line_id: int
    line_label: str
    direction: str  # "A->B" atau "B->A"
    class_name: str
    track_id: int
    confidence: float

    def as_row(self) -> list:
        return [
            time.strftime("%H:%M:%S", time.localtime(self.timestamp)),
            self.line_label,
            self.direction,
            self.class_name,
            self.track_id,
            f"{self.confidence:.2f}",
        ]


@dataclass
class TrackState:
    """Status satu objek terlacak pada satu garis."""

    line_id: int
    track_id: int
    side: float
    class_name: str
    conf: float
    last_seen: float = field(default_factory=time.time)


class LineCrossingCounter:
    """Menghitung objek yang menyeberang tiap garis.

    Satu track dihitung maksimal satu kali per garis (lalu di-reset saat
    objek keluar layar / hilang cukup lama).
    """

    HEADERS = ["Waktu", "Garis", "Arah", "Kelas", "ID", "Conf"]

    def __init__(self, stale_after: float = 3.0) -> None:
        self.lines: list[Line] = []
        self.events: list[CrossingEvent] = []
        self.counts: dict[str, int] = {}
        self.direction_counts: dict[tuple[int, str], int] = {}
        self._state: dict[tuple[int, int], TrackState] = {}
        self._next_line_id = 1
        self.stale_after = stale_after

    # ------------------------------------------------------------- garis
    def add_line(self, p1: Point, p2: Point, label: str = "") -> Line:
        line = Line(self._next_line_id, p1, p2, label)
        self._next_line_id += 1
        self.lines.append(line)
        return line

    def remove_line(self, line_id: int) -> None:
        self.lines = [l for l in self.lines if l.id != line_id]
        self._state = {k: v for k, v in self._state.items() if k[0] != line_id}

    def clear_lines(self) -> None:
        self.lines.clear()
        self._state.clear()

    def nearest_line(self, p: Point, max_dist: float = 0.03) -> Line | None:
        """Garis terdekat dari titik (dalam ruang ternormalisasi)."""
        best, best_d = None, max_dist
        for line in self.lines:
            for end in (line.p1, line.p2):
                d = ((end[0] - p[0]) ** 2 + (end[1] - p[1]) ** 2) ** 0.5
                if d < best_d:
                    best, best_d = line, d
        return best

    # ---------------------------------------------------------- hitung
    def update(
        self,
        detections: list,
        now: float | None = None,
    ) -> list[CrossingEvent]:
        """Protes deteksi frame ini.

        Terima objek Detection (atribut track_id/centroid/name/conf) atau
        tuple (track_id, centroid, class_name, confidence).
        """
        now = now if now is not None else time.time()
        found: list[CrossingEvent] = []

        items: list[tuple[int, Point, str, float]] = []
        for d in detections:
            if hasattr(d, "centroid"):
                items.append((d.track_id, d.centroid, d.name, d.conf))
            else:
                items.append(d)

        for line in self.lines:
            for track_id, centroid, class_name, conf in items:
                side = line.side(centroid)
                if abs(side) < 1e-9:
                    continue
                key = (line.id, track_id)
                prev = self._state.get(key)
                if prev is None:
                    self._state[key] = TrackState(line.id, track_id, side, class_name, conf, now)
                    continue

                if (prev.side < 0 <= side) or (prev.side > 0 >= side):
                    direction = "A->B" if prev.side < 0 else "B->A"
                    # Hanya hitung sekali: setelah melewati, keadaan disimpan
                    # sebagai sisi baru sehingga objek diam di garis tak dobel.
                    event = CrossingEvent(now, line.id, line.label_or_id, direction,
                                          class_name, track_id, conf)
                    found.append(event)
                    self.counts[class_name] = self.counts.get(class_name, 0) + 1
                    k = (line.id, direction)
                    self.direction_counts[k] = self.direction_counts.get(k, 0) + 1
                prev.side = side
                prev.class_name = class_name
                prev.conf = conf
                prev.last_seen = now

        self.events.extend(found)
        self._prune(now)
        return found

    def _prune(self, now: float) -> None:
        """Buang state track yang sudah lama tak terlihat."""
        dead = [k for k, v in self._state.items() if now - v.last_seen > self.stale_after]
        for k in dead:
            del self._state[k]

    # ---------------------------------------------------------- ringkasan
    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def reset(self) -> None:
        self.counts.clear()
        self.direction_counts.clear()
        self.events.clear()
        self._state.clear()
