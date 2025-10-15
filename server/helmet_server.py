import asyncio

import cv2
import grpc
import numpy as np
from typing_extensions import override

from stream import camera_stream_pb2 as pb
from stream import camera_stream_pb2_grpc as api


class HelmetServer(api.CameraStreamServiceServicer):
    def __init__(self):
        super().__init__()
        self.server: grpc.aio.Server | None = None

    @override
    async def StreamFrames(self, request_iterator, context):
        try:
            async for msg in request_iterator:
                img = cv2.imdecode(np.frombuffer(msg.frame, np.uint8), cv2.IMREAD_COLOR)
                print(f"[HelmetServer] camera={msg.camera_id}, frame={img.shape}, timestamp={msg.timestamp}")
        except asyncio.CancelledError:
            pass
        return pb.EmptyResponse()

    async def start(self, address):
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
