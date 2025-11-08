import asyncio

from config import cfg
from inference.helmet_detector import HelmetDetector
from server.helmet_server import HelmetServer
from stream.camera_stream import CameraStream


async def run_camera(camera: CameraStream, address: str, delay: float) -> None:
    await asyncio.sleep(delay)
    await camera.stream_to(address)


async def main():
    helmet_detector = HelmetDetector(
        engine_path=cfg.TRT_ENGINE_PATH,
        det_thresh=cfg.TRT_DET_THRESH,
        iou_thresh=cfg.TRT_IOU_THRESH
    )

    helmet_server = HelmetServer(
        detector=helmet_detector,
        cameras_count=cfg.CAMERAS_COUNT,
        is_visualizing=True,
    )
    await helmet_server.start(cfg.SERVER_BIND_ADDR)

    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(helmet_server.wait())
            tg.create_task(helmet_server.inference_loop())

            for i in range(cfg.CAMERAS_COUNT):
                cam = CameraStream(
                    camera_id=i,
                    video="report_data/test_videos/test_helmet.mp4",
                    timeout=cfg.CAMERA_TIMEOUT,
                )
                start_delay = i * cfg.CAMERA_TIMEOUT / cfg.CAMERAS_COUNT
                tg.create_task(
                    run_camera(cam, cfg.CLIENT_TARGET_ADDR, start_delay),
                    name=f"camera_{i}",
                )
    finally:
        await helmet_server.stop()
        await asyncio.sleep(0)


if __name__ == "__main__":
    asyncio.run(main())
