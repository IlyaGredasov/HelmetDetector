import asyncio
import os

import uvicorn

from config import cfg
from inference.helmet_detector import HelmetDetector
from server.helmet_server import HelmetServer
from stream.camera_stream import CameraStream


async def run_camera_stream(camera: CameraStream, address: str, delay: float) -> None:
    """
    Запускает поток камеры с задержкой.

    :param camera: объект CameraStream
    :param address: адрес gRPC-сервера
    :param delay: задержка перед стартом
    :return: None
    """
    await asyncio.sleep(delay)
    await camera.stream_to(address)


async def run_helmet_db_api():
    """
    Запускает HTTP API базы данных (Uvicorn).

    :return: None
    """
    config = uvicorn.Config(
        "db.helmet_db_api:helmet_db_api",
        host=cfg.DB_API_HOST,
        port=cfg.DB_API_PORT,
        reload=False,
    )
    server = uvicorn.Server(config)
    await server.serve()


async def main():
    """
    Главная точка запуска сервера: создаёт детектор, сервер и процессы камер.

    :return: None
    """
    helmet_detector = HelmetDetector(
        engine_path=cfg.TRT_ENGINE_PATH,
        det_thresh=cfg.TRT_DET_THRESH,
        iou_thresh=cfg.TRT_IOU_THRESH,
    )

    helmet_server = HelmetServer(
        detector=helmet_detector,
        cameras_count=cfg.CAMERAS_COUNT,
        db_api_url=f"http://{cfg.DB_API_HOST}:{cfg.DB_API_PORT}",
        alarm_factor=cfg.ALARM_FACTOR,
        alarm_thresh=cfg.ALARM_THRESH,
        is_visualizing=cfg.VISUALIZE,
    )
    await helmet_server.start(cfg.SERVER_BIND_ADDR)

    try:
        async with (
            asyncio.TaskGroup() as tg
        ):  # TaskGroup — чтобы при ошибке завершались все задачи сервера
            tg.create_task(run_helmet_db_api())
            for i in range(cfg.CAMERAS_COUNT):
                cam = CameraStream(
                    camera_id=i,
                    video="report_data/test_videos/test_hat.mp4",
                    timeout=cfg.CAMERA_TIMEOUT,
                )
                # Задержка старта — чтобы камеры не отправляли кадры одновременно
                start_delay = i * cfg.CAMERA_TIMEOUT / cfg.CAMERAS_COUNT
                tg.create_task(
                    run_camera_stream(cam, cfg.CLIENT_TARGET_ADDR, start_delay),
                    name=f"camera_{i}",
                )
            tg.create_task(helmet_server.wait())
            tg.create_task(helmet_server.inference_loop())
    finally:
        await helmet_server.stop()


if __name__ == "__main__":
    # WindowsSelectorEventLoopPolicy — требуется для корректной работы uvicorn под Windows
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
