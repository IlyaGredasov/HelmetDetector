import asyncio
from datetime import datetime

import cv2
import grpc

from config import cfg
from stream import camera_stream_pb2 as pb
from stream import camera_stream_pb2_grpc as api


def encode_jpg(img) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return bytes(buf)


class CameraStream:
    def __init__(self, camera_id: int, video: str, timeout: float):
        self.camera_id, self.video, self.timeout = camera_id, video, timeout

    async def frames_generator(self):
        cap = cv2.VideoCapture(self.video)
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                frame = cv2.resize(frame, (cfg.IMG_W, cfg.IMG_H), interpolation=cv2.INTER_LINEAR)
                yield pb.CameraFrame(
                    camera_id=self.camera_id,
                    frame=encode_jpg(frame),
                    timestamp=datetime.now().isoformat(timespec="seconds"),
                )
                await asyncio.sleep(self.timeout)
        except Exception as e:
            print(e)
        finally:
            cap.release()

    async def stream_to(self, addr: str):
        async with grpc.aio.insecure_channel(addr) as ch:
            stub = api.CameraStreamServiceStub(ch)
            while True:
                try:
                    await stub.StreamFrames(self.frames_generator())
                except grpc.aio.AioRpcError as e:
                    print(f"[CameraStream] RPC ended: {e.code().name}, retrying...")
                    await asyncio.sleep(1)
                except asyncio.CancelledError:
                    print("[CameraStream] cancelled, closing")
                    break
