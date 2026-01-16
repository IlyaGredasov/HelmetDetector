import math
from typing import Sequence

import cv2
import numpy as np

from inference.helmet_detector import Detection


class HelmetServerVisualizer:
    def __init__(self, num_cameras: int):
        self.num_cameras = num_cameras
        self.rows, self.cols = self.compute_grid(num_cameras)

        cv2.namedWindow("HelmetServer", cv2.WINDOW_NORMAL)

    @staticmethod
    def compute_grid(n: int) -> tuple[int, int]:
        if n <= 0:
            return 1, 1
        for rows in range(int(math.isqrt(n)), 0, -1):
            if n % rows == 0:
                cols = n // rows
                return rows, cols
        return n, 1

    def visualize(self, camera_ids: Sequence[int], images: Sequence[np.ndarray],
                  detections_batch: Sequence[Sequence[Detection]], alarm_levels: Sequence[float]) -> None:
        if not images:
            return

        h, w = images[0].shape[:2]
        grid_h = self.rows * h
        grid_w = self.cols * w

        grid = np.zeros((grid_h, grid_w, 3), dtype=images[0].dtype)

        for camera_id, img, detections, alarm_level in zip(camera_ids, images, detections_batch, alarm_levels):
            if camera_id >= self.rows * self.cols:
                break

            frame_row = camera_id // self.cols
            frame_column = camera_id % self.cols

            frame_y0 = frame_row * h
            frame_y1 = frame_y0 + h
            frame_x0 = frame_column * w
            frame_x1 = frame_x0 + w

            tile = img.copy()

            cam_label = f"Camera {camera_id}"
            (tw, th), _ = cv2.getTextSize(cam_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(tile, (0, 0), (tw + 8, th + 8), (0, 0, 0), -1)
            cv2.putText(
                tile,
                cam_label,
                (4, th + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            alarm_txt = f"{alarm_level:.2f}"
            color = (0, 255, 0) if alarm_level < 0.5 else (0, 165, 255) if alarm_level < 0.8 else (0, 0, 255)
            cv2.rectangle(tile, (w - tw - 12, 0), (w, th + 8), (0, 0, 0), -1)
            cv2.putText(tile, alarm_txt, (w - tw - 6, th + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

            for det in detections:
                x1, y1, x2, y2 = det.x1, det.y1, det.x2, det.y2
                color = (0, 255, 0) if det.class_id else (0, 0, 255)
                cv2.rectangle(tile, (x1, y1), (x2, y2), color, 2)

                txt = str(det.class_id)
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                ty = max(0, y1 - 4)
                cv2.rectangle(tile, (x1, ty - th - 6), (x1 + tw + 6, ty), color, -1)
                cv2.putText(
                    tile,
                    txt,
                    (x1 + 3, ty - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

            grid[frame_y0:frame_y1, frame_x0:frame_x1] = tile

        cv2.imshow("HelmetServer", grid)
        cv2.waitKey(1)
