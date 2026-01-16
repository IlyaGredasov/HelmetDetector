import cv2
import numpy as np
import pytest

from stream.camera_stream import encode_jpg


def test_encode_jpg_color_image():
    img = np.full((50, 50, 3), 200, dtype=np.uint8)
    data = encode_jpg(img)
    assert isinstance(data, bytes)
    assert len(data) > 0


def test_encode_jpg_grayscale_like_image():
    img = np.full((30, 30, 3), 100, dtype=np.uint8)
    data = encode_jpg(img)
    assert isinstance(data, bytes)


def test_encode_jpg_raises_runtime_error_on_failure(monkeypatch):
    def fake_imencode(ext, img, params):
        return False, None

    monkeypatch.setattr(cv2, "imencode", fake_imencode)

    img = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError):
        encode_jpg(img)


def test_encode_jpg_passes_correct_extension(monkeypatch):
    called = {}

    def fake_imencode(ext, img, params):
        called["ext"] = ext
        return True, np.array([1], dtype=np.uint8)

    monkeypatch.setattr(cv2, "imencode", fake_imencode)

    img = np.zeros((10, 10, 3), dtype=np.uint8)
    encode_jpg(img)
    assert called["ext"] == ".jpg"


def test_encode_jpg_multiple_calls():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    data1 = encode_jpg(img)
    data2 = encode_jpg(img)
    assert isinstance(data1, bytes)
    assert isinstance(data2, bytes)
