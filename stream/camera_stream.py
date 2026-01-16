import asyncio
from datetime import datetime

import cv2
import grpc

from config import cfg
from stream import camera_stream_pb2 as pb
from stream import camera_stream_pb2_grpc as api
from utils import letterbox


def encode_jpg(img) -> bytes:
    """
    Кодирует изображение в JPEG.

    :param img: исходное изображение
    :return: байты JPEG
    """
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return bytes(buf)


class CameraStream:
    """
    Эмуляция видеопотока, отправляющий кадры на gRPC-сервер.

    :param camera_id: ID камеры
    :param video: путь к видеофайлу
    :param timeout: задержка между кадрами
    """

    def __init__(self, camera_id: int, video: str, timeout: float):
        self.camera_id = camera_id
        self.video = video
        self.timeout = timeout

    async def frames_generator(self):
        """
        Генерирует кадры и отправляет их в формате CameraFrame.

        :return: асинхронный генератор кадров
        """
        cap = cv2.VideoCapture(self.video)
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # step - количество кадров, которое нужно пропустить между отправками,
            # чтобы частота отправки была равна timeout
            step = max(1, round(fps * self.timeout))
            frame_index = 0
            while True:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = cap.read()
                if not ok:
                    frame_index = 0  # если дошли до конца - начинаем сначала
                    continue

                frame, _, _ = letterbox(frame, (cfg.IMG_H, cfg.IMG_W))

                yield pb.CameraFrame(
                    camera_id=self.camera_id,
                    frame=encode_jpg(frame),
                    timestamp=datetime.now().isoformat(timespec="seconds"),
                )
                frame_index = (frame_index + step) % frame_count
                await asyncio.sleep(self.timeout)
        except Exception as e:
            print(f"[CameraStream {self.camera_id}] error:", e)
        finally:
            cap.release()

    async def stream_to(self, addr: str):
        """
        Отправляет поток кадров на указанный gRPC-адрес.

        :param addr: адрес сервера
        :return: None
        """
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
