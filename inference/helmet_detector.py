from dataclasses import dataclass
from typing import List
from typing import Tuple

import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

from config import cfg


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int


def letterbox(img: np.ndarray, new_shape: Tuple[int, int]) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    h, w = img.shape[:2]
    ratio = min(new_shape[0] / h, new_shape[1] / w)
    new_height, new_width = int(round(h * ratio)), int(round(w * ratio))
    pad_h, pad_w = new_shape[0] - new_height, new_shape[1] - new_width
    top = pad_h // 2
    left = pad_w // 2
    resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    out = np.full((new_shape[0], new_shape[1], 3), 114, dtype=img.dtype)
    out[top:top + new_height, left:left + new_width] = resized
    return out, ratio, (left, top)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> List[int]:
    if boxes.size == 0:
        return []
    boxes = boxes.astype(np.float32, copy=False)
    scores = scores.astype(np.float32, copy=False)
    x1, y1, x2, y2 = boxes.T
    w = np.maximum(0.0, x2 - x1)
    h = np.maximum(0.0, y2 - y1)
    valid = np.where((w > 0) & (h > 0))[0]
    boxes = boxes[valid]
    scores = scores[valid]
    if boxes.size == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        ww = np.maximum(0.0, xx2 - xx1)
        hh = np.maximum(0.0, yy2 - yy1)
        inter = ww * hh
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-7)
        order = order[1:][iou <= iou_thresh]
    map_back = valid[keep]
    return map_back.tolist()


class HelmetDetector:
    DTYPE_MAP = {
        trt.DataType.FLOAT: np.float32,
        trt.DataType.HALF: np.float16,
        trt.DataType.BF16: np.float16,
    }

    def __init__(self, engine_path: str, det_thresh: float = 0.4, iou_thresh: float = 0.45, img_size: int = 640):
        self.det_thresh = det_thresh
        self.iou_thresh = iou_thresh
        self.img_size = img_size
        with open(engine_path, "rb") as f:
            self.engine: trt.ICudaEngine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine")
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)
        self.input_np_dtype = self.DTYPE_MAP[self.engine.get_tensor_dtype(self.input_name)]
        self.output_np_dtype = self.DTYPE_MAP[self.engine.get_tensor_dtype(self.output_name)]
        self.last_shape = None
        self.h_input = None
        self.h_output = None
        self.d_input = None
        self.d_output = None
        self.event = cuda.Event()

    def preprocess(self, images: List[np.ndarray]) -> Tuple[
        np.ndarray, List[Tuple[int, int]], List[float], List[Tuple[int, int]]]:
        batch = np.empty((len(images), 3, cfg.IMG_H, cfg.IMG_W), dtype=self.input_np_dtype)
        orig_shapes, scales, paddings = [], [], []
        for i, im_bgr in enumerate(images):
            rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
            lb, r, (left, top) = letterbox(rgb, (cfg.IMG_H, cfg.IMG_W))
            arr = lb.transpose(2, 0, 1).astype(self.input_np_dtype)
            arr *= np.array(1 / 255, dtype=self.input_np_dtype)
            batch[i] = arr
            orig_shapes.append((im_bgr.shape[0], im_bgr.shape[1]))
            scales.append(r)
            paddings.append((left, top))
        return batch, orig_shapes, scales, paddings

    def provide_memory(self, input_shape):
        if self.last_shape == input_shape:
            return
        self.context.set_input_shape(self.input_name, input_shape)
        output_shape = tuple(self.context.get_tensor_shape(self.output_name))
        self.h_input = cuda.pagelocked_empty(input_shape, dtype=self.input_np_dtype)
        self.h_output = cuda.pagelocked_empty(output_shape, dtype=self.output_np_dtype)
        if self.d_input:
            self.d_input.free()
        if self.d_output:
            self.d_output.free()
        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)
        self.context.set_tensor_address(self.input_name, int(self.d_input))
        self.context.set_tensor_address(self.output_name, int(self.d_output))
        self.last_shape = input_shape

    def forward(self, batch: np.ndarray) -> np.ndarray:
        self.provide_memory(tuple(batch.shape))
        np.copyto(self.h_input, batch, casting='no')
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
        self.context.execute_async_v3(stream_handle=int(self.stream.handle))
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.event.record(self.stream)
        self.event.synchronize()
        return self.h_output.copy()

    def postprocess(self, output: np.ndarray, original_shapes: List[Tuple[int, int]], scales: List[float],
                    paddings: List[Tuple[int, int]]) -> List[List[Detection]]:
        if output.shape[1] < output.shape[2]:
            output = np.transpose(output, (0, 2, 1))

        batch_size, num_boxes, num_channels = output.shape
        num_classes = num_channels - 4

        xywh = output[..., :4]
        class_logits = output[..., 4:4 + num_classes]
        class_ids = np.argmax(class_logits, axis=-1).astype(np.int32)
        confidences = np.take_along_axis(class_logits, class_ids[..., None], axis=-1)[..., 0]

        x, y, w, h = np.split(xywh, 4, axis=-1)
        x1 = (x - w / 2)[..., 0]
        y1 = (y - h / 2)[..., 0]
        x2 = (x + w / 2)[..., 0]
        y2 = (y + h / 2)[..., 0]

        results = []

        for b in range(batch_size):
            boxes = np.stack([x1[b], y1[b], x2[b], y2[b]], axis=1)
            scores = confidences[b]
            classes = class_ids[b]

            mask = scores >= self.det_thresh
            boxes, scores, classes = boxes[mask], scores[mask], classes[mask]
            if boxes.shape[0] == 0:
                results.append([])
                continue

            pad_left, pad_top = paddings[b]
            scale = scales[b]
            orig_h, orig_w = original_shapes[b]

            boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_left) / scale
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_top) / scale
            boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, orig_w)
            boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, orig_h)

            final_boxes, final_scores, final_classes = [], [], []

            for cls_id in np.unique(classes):
                idx = np.where(classes == cls_id)[0]
                kept_indices = nms(boxes[idx], scores[idx], self.iou_thresh)
                if len(kept_indices):
                    final_boxes.append(boxes[idx][kept_indices])
                    final_scores.append(scores[idx][kept_indices])
                    final_classes.append(np.full(len(kept_indices), cls_id, np.int32))

            if final_boxes:
                boxes = np.concatenate(final_boxes, 0)
                scores = np.concatenate(final_scores, 0)
                classes = np.concatenate(final_classes, 0)
            else:
                boxes = np.empty((0, 4))
                scores = np.empty((0,))
                classes = np.empty((0,), np.int32)

            detections = [
                Detection(int(x1), int(y1), int(x2), int(y2), float(score), int(cls))
                for (x1, y1, x2, y2), score, cls in zip(boxes, scores, classes)
            ]
            results.append(detections)

        return results

    def detect(self, images: List[np.ndarray]) -> List[List[Detection]]:
        batch, orig_shapes, scales, pads = self.preprocess(images)
        out = self.forward(batch)
        return self.postprocess(out, orig_shapes, scales, pads)

    def __del__(self):
        if self.d_input is not None:
            try:
                self.d_input.free()
            except Exception:
                pass
        if self.d_output is not None:
            try:
                self.d_output.free()
            except Exception:
                pass
