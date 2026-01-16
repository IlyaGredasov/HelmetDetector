from dataclasses import dataclass

import numpy as np


@dataclass
class CameraSlot:
    """
    "Сокет" камеры - хранит последний кадр и временную метку от камеры.

    :param frame: изображение кадра
    :param timestamp: строковая метка времени или None
    """
    frame: np.ndarray
    timestamp: str | None
