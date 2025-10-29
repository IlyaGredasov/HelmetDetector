import os
from dataclasses import dataclass

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(usecwd=True))


@dataclass(frozen=True)
class Config:
    IMG_H: int = int(os.getenv('IMG_H', 640))
    IMG_W: int = int(os.getenv('IMG_W', 640))
    MIN_BATCH: int = int(os.getenv("MIN_BATCH", 1))
    OPT_BATCH: int = int(os.getenv("OPT_BATCH", 4))
    MAX_BATCH: int = int(os.getenv("MAX_BATCH", 16))

    CAMERAS_COUNT: int = int(os.getenv("CAMERAS_COUNT", 6))
    CAMERA_TIMEOUT: float = float(os.getenv("CAMERA_TIMEOUT", 0.1))
    CAMERA_VIDEO_PATH: str = str(os.getenv("CAMERA_VIDEO_PATH"))

    TRT_DET_THRESH: float = float(os.getenv("TRT_DET_THRESH", 0.4))
    TRT_IOU_THRESH: float = float(os.getenv("TRT_IOU_THRESH", 0.45))

    SERVER_BIND_ADDR: str = os.getenv("SERVER_BIND_ADDR", "[::]:50051")
    CLIENT_TARGET_ADDR: str = os.getenv("CLIENT_TARGET_ADDR", "localhost:50051")


cfg = Config()
