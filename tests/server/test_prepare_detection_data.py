import asyncio

import cv2
import numpy as np
import pytest

from server.camera_slot import CameraSlot
from server.helmet_server import HelmetServer


@pytest.mark.asyncio
async def test_prepare_detection_data_success_with_timestamp(monkeypatch):
    server = object.__new__(HelmetServer)
    server.lock = asyncio.Lock()
    frame = np.full((100, 100, 3), 255, dtype=np.uint8)
    server.slots = {1: CameraSlot(frame=frame, timestamp=None)}

    def fake_imencode(ext, img):
        assert img.shape == frame.shape
        return True, np.array([1, 2, 3], dtype=np.uint8)

    monkeypatch.setattr(cv2, "imencode", fake_imencode)

    img_bytes, detection_time = await server.prepare_detection_data(1, "2025-01-01T00:00:00Z")
    assert img_bytes == bytes(np.array([1, 2, 3], dtype=np.uint8))
    assert detection_time == "2025-01-01T00:00:00Z"


@pytest.mark.asyncio
async def test_prepare_detection_data_generates_timestamp_when_empty(monkeypatch):
    server = object.__new__(HelmetServer)
    server.lock = asyncio.Lock()
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    server.slots = {2: CameraSlot(frame=frame, timestamp=None)}

    def fake_imencode(ext, img):
        return True, np.array([10, 20], dtype=np.uint8)

    monkeypatch.setattr(cv2, "imencode", fake_imencode)

    img_bytes, detection_time = await server.prepare_detection_data(2, "")
    assert img_bytes == bytes(np.array([10, 20], dtype=np.uint8))
    assert isinstance(detection_time, str)
    assert detection_time != ""


@pytest.mark.asyncio
async def test_prepare_detection_data_returns_none_on_encode_failure(monkeypatch):
    server = object.__new__(HelmetServer)
    server.lock = asyncio.Lock()
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    server.slots = {3: CameraSlot(frame=frame, timestamp=None)}

    def fake_imencode(ext, img):
        return False, None

    monkeypatch.setattr(cv2, "imencode", fake_imencode)

    img_bytes, detection_time = await server.prepare_detection_data(3, "ts")
    assert img_bytes is None
    assert detection_time is None


@pytest.mark.asyncio
async def test_prepare_detection_data_raises_key_error_for_unknown_camera(monkeypatch):
    server = object.__new__(HelmetServer)
    server.lock = asyncio.Lock()
    server.slots = {}

    with pytest.raises(KeyError):
        await server.prepare_detection_data(999, "ts")


@pytest.mark.asyncio
async def test_prepare_detection_data_uses_latest_frame(monkeypatch):
    server = object.__new__(HelmetServer)
    server.lock = asyncio.Lock()
    frame1 = np.zeros((10, 10, 3), dtype=np.uint8)
    frame2 = np.ones((10, 10, 3), dtype=np.uint8) * 255
    slot = CameraSlot(frame=frame1, timestamp=None)
    server.slots = {1: slot}

    frames = []

    def fake_imencode(ext, img):
        frames.append(img.copy())
        return True, np.array([1], dtype=np.uint8)

    monkeypatch.setattr(cv2, "imencode", fake_imencode)

    await server.prepare_detection_data(1, "ts1")
    slot.frame = frame2
    await server.prepare_detection_data(1, "ts2")
    assert np.array_equal(frames[0], frame1)
    assert np.array_equal(frames[1], frame2)
