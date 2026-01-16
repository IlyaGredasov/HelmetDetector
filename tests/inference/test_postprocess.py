import numpy as np

from inference.helmet_detector import Detection
from inference.helmet_detector import HelmetDetector


def make_detector(det_thresh=0.4, iou_thresh=0.5):
    detector = object.__new__(HelmetDetector)
    detector.det_thresh = det_thresh
    detector.iou_thresh = iou_thresh
    return detector


def build_output_no_transpose():
    out = np.zeros((1, 6, 5), dtype=np.float32)
    out[0, 0, :4] = [50, 50, 20, 20]
    out[0, 0, 4] = 0.9
    out[0, 1, :4] = [52, 52, 20, 20]
    out[0, 1, 4] = 0.5
    out[0, 2, :4] = [200, 200, 30, 30]
    out[0, 2, 4] = 0.8
    out[0, 3, 4] = 0.1
    out[0, 4, 4] = 0.2
    out[0, 5, 4] = 0.39
    return out


def test_postprocess_filters_by_threshold_and_nms():
    detector = make_detector()
    output = build_output_no_transpose()
    original_shapes = [(480, 640)]
    scales = [1.0]
    paddings = [(0, 0)]
    result = detector.postprocess(output, original_shapes, scales, paddings)
    assert len(result) == 1
    detections = result[0]
    assert len(detections) == 2
    assert isinstance(detections[0], Detection)
    xs = [d.x1 for d in detections]
    ys = [d.y1 for d in detections]
    assert min(xs) >= 0
    assert min(ys) >= 0


def test_postprocess_handles_transposed_output():
    detector = make_detector()
    output = np.transpose(build_output_no_transpose(), (0, 2, 1))
    original_shapes = [(480, 640)]
    scales = [1.0]
    paddings = [(0, 0)]
    result = detector.postprocess(output, original_shapes, scales, paddings)
    detections = result[0]
    assert len(detections) == 2


def test_postprocess_all_scores_below_threshold():
    detector = make_detector()
    output = build_output_no_transpose()
    output[..., 4] = 0.1
    original_shapes = [(480, 640)]
    scales = [1.0]
    paddings = [(0, 0)]
    result = detector.postprocess(output, original_shapes, scales, paddings)
    assert result == [[]]


def test_postprocess_applies_per_class_nms():
    detector = make_detector()
    out = np.zeros((1, 6, 6), dtype=np.float32)
    out[0, 0, :4] = [50, 50, 20, 20]
    out[0, 0, 4] = 0.9
    out[0, 0, 5] = 0.1
    out[0, 1, :4] = [50, 50, 20, 20]
    out[0, 1, 4] = 0.1
    out[0, 1, 5] = 0.9
    original_shapes = [(480, 640)]
    scales = [1.0]
    paddings = [(0, 0)]
    result = detector.postprocess(out, original_shapes, scales, paddings)
    detections = result[0]
    assert len(detections) == 2
    class_ids = {d.class_id for d in detections}
    assert class_ids == {0, 1}


def test_postprocess_clips_boxes_to_image_bounds():
    detector = make_detector()
    out = np.zeros((1, 5, 5), dtype=np.float32)
    out[0, 0, :4] = [500, 500, 1000, 1000]
    out[0, 0, 4] = 0.9
    original_shapes = [(480, 640)]
    scales = [1.0]
    paddings = [(0, 0)]
    result = detector.postprocess(out, original_shapes, scales, paddings)
    detections = result[0]
    assert len(detections) == 1
    det = detections[0]
    assert 0 <= det.x1 <= 640
    assert 0 <= det.y1 <= 480
    assert 0 <= det.x2 <= 640
    assert 0 <= det.y2 <= 480
