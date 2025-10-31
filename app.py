import asyncio

from config import cfg
from server.helmet_server import HelmetServer
from stream.camera_stream import CameraStream


async def main():
    cameras = [CameraStream(camera_id=i, video="report_data/test_videos/test_hat.mp4", timeout=cfg.CAMERA_TIMEOUT) for i in
               range(cfg.CAMERAS_COUNT)]
    server = HelmetServer()
    await server.start(cfg.SERVER_BIND_ADDR)

    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(server.wait())
            for cam in cameras:
                tg.create_task(cam.stream_to(cfg.CLIENT_TARGET_ADDR))
    finally:
        await server.stop()
        await asyncio.sleep(0)


if __name__ == "__main__":
    asyncio.run(main())
