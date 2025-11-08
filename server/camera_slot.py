from dataclasses import dataclass

import numpy as np


@dataclass
class CameraSlot:
    frame: np.ndarray
    timestamp: str | None
