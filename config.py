import os
from dataclasses import dataclass

from dotenv import find_dotenv
from dotenv import load_dotenv

load_dotenv(find_dotenv(usecwd=True))


@dataclass(frozen=True)
class Config:
    IMG_H: int = int(os.getenv('IMG_H', 640))
    IMG_W: int = int(os.getenv('IMG_W', 640))

    CAMERAS_COUNT: int = int(os.getenv("CAMERAS_COUNT", 6))
    CAMERA_TIMEOUT: float = float(os.getenv("CAMERA_TIMEOUT", 1))

    TRT_ENGINE_PATH: str = os.getenv("TRT_ENGINE_PATH")
    TRT_DET_THRESH: float = float(os.getenv("TRT_DET_THRESH", 0.4))
    TRT_IOU_THRESH: float = float(os.getenv("TRT_IOU_THRESH", 0.4))

    SERVER_BIND_ADDR: str = os.getenv("SERVER_BIND_ADDR", "0.0.0.0:50051")
    CLIENT_TARGET_ADDR: str = os.getenv("CLIENT_TARGET_ADDR", "127.0.0.1:50051")


cfg = Config()
