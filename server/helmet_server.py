import asyncio
from datetime import datetime
from datetime import timezone
from typing import Dict

import cv2
import grpc
import httpx
import numpy as np
from httpx import AsyncClient

from config import cfg
from inference.helmet_detector import Detection
from inference.helmet_detector import HelmetDetector
from server.camera_slot import CameraSlot
from server.helmet_server_visualizer import HelmetServerVisualizer
from stream import camera_stream_pb2 as pb
from stream import camera_stream_pb2_grpc as api


class HelmetServer(api.CameraStreamServiceServicer):
    def __init__(self, detector: HelmetDetector, cameras_count: int, db_api_url: str, alarm_factor: float = 0.5,
                 alarm_thresh: float = 0.8, is_visualizing: bool = False):
        super().__init__()
        self.server: grpc.aio.Server | None = None

        self.detector = detector
        self.cameras_count = cameras_count

        self.lock = asyncio.Lock()

        self.blank_frame = np.zeros((cfg.IMG_H, cfg.IMG_W, 3), dtype=np.uint8)

        self.slots: Dict[int, CameraSlot] = {
            camera_id: CameraSlot(
                frame=self.blank_frame.copy(),
                timestamp=None,
            )
            for camera_id in range(self.cameras_count)
        }
        self.cameras_ids = sorted(self.slots.keys())
        self.visualizer = HelmetServerVisualizer(self.cameras_count) if is_visualizing else None

        self.alarm_factor = alarm_factor
        self.alarm_thresh = alarm_thresh
        self.alarm_levels: Dict[int, float] = {camera_id: 0.0 for camera_id in self.cameras_ids}

        self.http_client = AsyncClient(base_url=db_api_url, timeout=5.0)

    async def StreamFrames(self, request_iterator, context):
        try:
            async for msg in request_iterator:
                if msg.camera_id not in self.slots:
                    continue

                img = cv2.imdecode(np.frombuffer(msg.frame, np.uint8), cv2.IMREAD_COLOR_BGR)
                if img is None:
                    continue

                img = cv2.resize(img, (cfg.IMG_W, cfg.IMG_H))

                slot = self.slots[msg.camera_id]
                async with self.lock:
                    if slot.frame.shape != img.shape:
                        slot.frame = img.copy()
                    else:
                        slot.frame[...] = img
                    slot.timestamp = msg.timestamp
        except asyncio.CancelledError:
            pass
        return pb.EmptyResponse()

    async def prepare_detection_data(self, camera_id: int, timestamp: str):
        async with self.lock:
            frame = self.slots[camera_id].frame.copy()

        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            print("[DB] JPEG encode failed, skip")
            return None, None

        img_bytes = bytes(buf)
        return img_bytes, timestamp or datetime.now(timezone.utc).isoformat()

    async def send_detection_to_api(self, camera_id: int, detection_time: str, img_bytes: bytes):
        try:
            requests = await self.http_client.post(
                "/detections",
                data={
                    "camera_id": str(camera_id),
                    "detection_time": detection_time,
                },
                files={
                    "image": ("frame.jpg", img_bytes, "image/jpeg"),
                },
            )
            requests.raise_for_status()
            data = requests.json()
            print(f"[DB] detection stored id={data['detection_id']}")
        except httpx.HTTPError as e:
            print(f"[DB] failed to store detection: {e}")

    async def handle_alarm(self, camera_id: int, timestamp: str, detections: list[Detection], level: float):
        print(
            f"[ALARM] camera_id={camera_id}, timestamp={timestamp}, "
            f"level={level:.3f}, detections={len(detections)}"
        )

        img_bytes, detection_time = await self.prepare_detection_data(camera_id, timestamp)
        if img_bytes is None:
            return

        await self.send_detection_to_api(camera_id, detection_time, img_bytes)
    async def start(self, address: str):
        self.server = grpc.aio.server()
        api.add_CameraStreamServiceServicer_to_server(self, self.server)
        self.server.add_insecure_port(address)
        await self.server.start()
        print(f"[HelmetServer] listening on {address}")

    async def inference_loop(self):
        print("[HelmetServer] inference loop started")
        while True:
            async with self.lock:
                images = [self.slots[camera_id].frame.copy() for camera_id in self.cameras_ids]
                timestamps = [self.slots[camera_id].timestamp or "" for camera_id in self.cameras_ids]
            detections_batch = self.detector.detect(images)
            for camera_id, timestamp, detections in zip(self.cameras_ids, timestamps, detections_batch):
                has_head = any(d.class_id == 1 for d in detections)
                level = self.alarm_factor * int(has_head) + (1.0 - self.alarm_factor) * self.alarm_levels[camera_id]
                self.alarm_levels[camera_id] = level

                print(
                    f"[Detection] camera_id={camera_id}, timestamp={timestamp}, "
                    f"detections={len(detections)}, alarm_level={level:.3f}"
                )

                if level >= self.alarm_thresh:
                    await self.handle_alarm(camera_id, timestamp, detections, level)
                    self.alarm_levels[camera_id] = 0.0

            if self.visualizer is not None:
                alarm_levels = [self.alarm_levels[cid] for cid in self.cameras_ids]
                self.visualizer.visualize(self.cameras_ids, images, detections_batch, alarm_levels)
            await asyncio.sleep(cfg.CAMERA_TIMEOUT)

    async def wait(self):
        await self.server.wait_for_termination()

    async def stop(self, grace: float = 0.0):
        if self.server is not None:
            await self.server.stop(grace)
            self.server = None
            print("[HelmetServer] stopped")
