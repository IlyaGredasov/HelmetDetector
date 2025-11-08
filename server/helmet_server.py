import asyncio
from typing import Dict

import cv2
import grpc
import numpy as np

from config import cfg
from inference.helmet_detector import HelmetDetector
from server.camera_slot import CameraSlot
from server.helmet_server_visualizer import HelmetServerVisualizer
from stream import camera_stream_pb2 as pb
from stream import camera_stream_pb2_grpc as api


class HelmetServer(api.CameraStreamServiceServicer):
    def __init__(self, detector: HelmetDetector, cameras_count: int, is_visualizing: bool):
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

    async def inference_loop(self):
        print("[HelmetServer] inference loop started")
        while True:
            async with self.lock:
                images = [self.slots[camera_id].frame.copy() for camera_id in self.cameras_ids]
                timestamps = [self.slots[camera_id].timestamp or "" for camera_id in self.cameras_ids]
            detections_batch = self.detector.detect(images)
            for camera_id, timestamp, detections in zip(self.cameras_ids, timestamps, detections_batch):
                print(f"[Detection] camera_id={camera_id}, timestamp={timestamp}, detections={len(detections)}")

            if self.visualizer is not None:
                self.visualizer.visualize(self.cameras_ids, images, detections_batch)
            await asyncio.sleep(cfg.CAMERA_TIMEOUT)

    async def start(self, address: str):
        self.server = grpc.aio.server()
        api.add_CameraStreamServiceServicer_to_server(self, self.server)
        self.server.add_insecure_port(address)
        await self.server.start()
        print(f"[HelmetServer] listening on {address}")

    async def wait(self):
        await self.server.wait_for_termination()

    async def stop(self, grace: float = 0.0):
        if self.server is not None:
            await self.server.stop(grace)
            self.server = None
            print("[HelmetServer] stopped")
