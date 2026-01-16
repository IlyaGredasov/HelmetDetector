import argparse
import time
from typing import Iterable

import cv2

from inference.helmet_detector import Detection
from inference.helmet_detector import HelmetDetector

CLASS_NAMES: list[str] = ["helmet", "head"]


def draw(frame_bgr, detections: Iterable[Detection]):
    for d in detections:
        color = (0, 255, 0) if d.class_id == 0 else (0, 128, 255)
        cv2.rectangle(frame_bgr, (d.x1, d.y1), (d.x2, d.y2), color, 2)
        y = max(d.y1 - 6, 0)
        label = f"{CLASS_NAMES[d.class_id] if d.class_id < len(CLASS_NAMES) else d.class_id}: {d.confidence:.2f}"
        cv2.putText(frame_bgr, label, (d.x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engine", type=str)
    ap.add_argument("video", type=str)
    args = ap.parse_args()

    det = HelmetDetector(args.engine, det_thresh=0.4, iou_thresh=0.4)

    try:
        cam_index = int(args.video)
        cap = cv2.VideoCapture(cam_index)
    except ValueError:
        cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    frame_id = 0

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_id += 1
        t0 = time.time()

        detections = det.detect([frame_bgr for _ in range(1)])[0]

        draw(frame_bgr, detections)

        fps = 1.0 / (time.time() - t0)
        cv2.putText(frame_bgr, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("TRT inference", frame_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
