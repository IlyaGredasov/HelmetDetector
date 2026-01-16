from utils import xyxy_to_yolo


def test_xyxy_to_yolo_center_and_size_basic():
    cx, cy, bw, bh = xyxy_to_yolo(0, 0, 10, 20, 20, 40)
    assert cx == 0.25
    assert cy == 0.25
    assert bw == 0.5
    assert bh == 0.5


def test_xyxy_to_yolo_full_image_box():
    cx, cy, bw, bh = xyxy_to_yolo(0, 0, 100, 200, 100, 200)
    assert cx == 0.5
    assert cy == 0.5
    assert bw == 1.0
    assert bh == 1.0


def test_xyxy_to_yolo_partial_box():
    cx, cy, bw, bh = xyxy_to_yolo(10, 20, 30, 60, 100, 100)
    assert abs(cx - 0.2) < 1e-6
    assert abs(cy - 0.4) < 1e-6
    assert abs(bw - 0.2) < 1e-6
    assert abs(bh - 0.4) < 1e-6


def test_xyxy_to_yolo_small_box():
    cx, cy, bw, bh = xyxy_to_yolo(5, 5, 6, 7, 10, 10)
    assert bw == 0.1
    assert bh == 0.2


def test_xyxy_to_yolo_handles_non_square_image():
    cx, cy, bw, bh = xyxy_to_yolo(0, 50, 50, 100, 200, 200)
    assert abs(cx - 0.125) < 1e-6
    assert abs(cy - 0.375) < 1e-6
