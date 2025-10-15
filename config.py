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

    SERVER_BIND_ADDR: str = os.getenv("SERVER_BIND_ADDR", "[::]:50051")
    CLIENT_TARGET_ADDR: str = os.getenv("CLIENT_TARGET_ADDR", "localhost:50051")

cfg = Config()
