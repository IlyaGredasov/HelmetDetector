import numpy as np

from utils import letterbox


def test_letterbox_same_shape():
    img = np.full((100, 200, 3), 123, dtype=np.uint8)
    new_img, ratio, paddings = letterbox(img, (100, 200))
    assert ratio == 1.0
    assert paddings == (0, 0)
    assert new_img.shape == img.shape
    assert np.array_equal(new_img, img)


def test_letterbox_scales_down_and_paddings_center():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    new_img, ratio, paddings = letterbox(img, (200, 300))
    assert new_img.shape == (200, 300, 3)
    assert 1.9 <= ratio <= 3.1
    left, top = paddings
    assert left >= 0
    assert top >= 0


def test_letterbox_scales_up():
    img = np.zeros((10, 20, 3), dtype=np.uint8)
    new_img, ratio, paddings = letterbox(img, (40, 40))
    assert new_img.shape == (40, 40, 3)
    assert ratio > 1.0
    left, top = paddings
    assert 0 <= left < 40
    assert 0 <= top < 40


def test_letterbox_padding_keeps_content_centered():
    img = np.zeros((30, 10, 3), dtype=np.uint8)
    img[10:20, 3:7] = 255
    new_img, ratio, paddings = letterbox(img, (60, 60))
    left, top = paddings
    content = new_img[top + int(10 * ratio):top + int(20 * ratio), left + int(3 * ratio):left + int(7 * ratio)]
    assert content.mean() > 200


def test_letterbox_preserves_dtype():
    img = (np.random.rand(50, 50, 3) * 255).astype(np.uint8)
    new_img, ratio, paddings = letterbox(img, (80, 80))
    assert new_img.dtype == img.dtype
    assert new_img.shape == (80, 80, 3)
