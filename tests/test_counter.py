"""Uji logika penyeberangan garis tanpa model/GUI."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vehicle_counter.counter import LineCrossingCounter


def make() -> LineCrossingCounter:
    c = LineCrossingCounter()
    c.add_line((0.0, 0.5), (1.0, 0.5), "G1")  # horizontal di tengah
    return c


def test_cross_downward_counts_once():
    c = make()
    # objek #1 turun dari y=0.3 ke y=0.7 melewati garis y=0.5
    c.update([(1, (0.4, 0.30), "car", 0.9)])
    assert c.total == 0, "belum melewati, tidak boleh dihitung"
    c.update([(1, (0.4, 0.49), "car", 0.9)])
    assert c.total == 0
    ev = c.update([(1, (0.4, 0.55), "car", 0.9)])
    assert len(ev) == 1 and c.total == 1, "penyeberangan pertama harus dihitung"
    # tetap di bawah garis: tidak boleh dobel
    for y in (0.60, 0.70, 0.80):
        c.update([(1, (0.4, y), "car", 0.9)])
    assert c.total == 1, "satu objek hanya dihitung sekali"


def test_cross_upward_direction():
    c = make()
    c.update([(2, (0.4, 0.80), "bus", 0.9)])
    ev = c.update([(2, (0.4, 0.40), "bus", 0.9)])
    assert len(ev) == 1
    assert ev[0].direction == "B->A", f"arah salah: {ev[0].direction}"
    assert c.counts["bus"] == 1


def test_multiple_objects_and_classes():
    c = make()
    c.update([(1, (0.2, 0.4), "car", 0.9), (2, (0.8, 0.4), "motorcycle", 0.9)])
    c.update([(1, (0.2, 0.6), "car", 0.9), (2, (0.8, 0.6), "motorcycle", 0.9)])
    assert c.total == 2
    assert c.counts == {"car": 1, "motorcycle": 1}


def test_two_lines_count_independently():
    c = LineCrossingCounter()
    c.add_line((0.0, 0.3), (1.0, 0.3), "L1")
    c.add_line((0.0, 0.7), (1.0, 0.7), "L2")
    c.update([(1, (0.5, 0.10), "truck", 0.9)])
    c.update([(1, (0.5, 0.40), "truck", 0.9)])  # lewati L1
    assert c.total == 1
    c.update([(1, (0.5, 0.80), "truck", 0.9)])  # lewati L2
    assert c.total == 2
    assert len(c.lines) == 2


def test_vertical_line_and_crossing():
    c = LineCrossingCounter()
    c.add_line((0.5, 0.0), (0.5, 1.0), "Vert")
    c.update([(1, (0.2, 0.5), "car", 0.9)])
    ev = c.update([(1, (0.8, 0.5), "car", 0.9)])
    assert len(ev) == 1 and c.total == 1


def test_retrack_after_leaving():
    c = make()
    c.update([(1, (0.4, 0.2), "car", 0.9)])
    c.update([(1, (0.4, 0.6), "car", 0.9)])
    assert c.total == 1
    # track hilang lama -> state dibersihkan; ID sama muncul lagi di bawah garis
    import time as _t
    now = _t.time()
    for _ in range(4):
        c.update([(1, (0.4, 0.6), "car", 0.9)], now=now)
    c._prune(now + 10)  # stale_after default 3s
    c.update([(1, (0.4, 0.2), "car", 0.9)], now=now + 11)  # reset ke atas garis
    assert c.total == 1, "state lama harus terhapus"
    c.update([(1, (0.4, 0.6), "car", 0.9)], now=now + 12)
    assert c.total == 2, "objek baru harus dihitung lagi"


def test_detection_object_input():
    """Menerima objek Detection (dataclass) dari detector, bukan cuma tuple."""

    class D:
        track_id = 7
        centroid = (0.5, 0.4)
        name = "car"
        conf = 0.88

    c = make()
    c.update([D()])
    D.centroid = (0.5, 0.6)
    ev = c.update([D()])
    assert len(ev) == 1 and ev[0].track_id == 7


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} lulus")
    sys.exit(1 if failed else 0)
