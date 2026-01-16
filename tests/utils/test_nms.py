import numpy as np

from utils import nms


def test_nms_empty_boxes_returns_empty_list():
    boxes = np.zeros((0, 4), dtype=np.float32)
    scores = np.zeros((0,), dtype=np.float32)
    keep = nms(boxes, scores, iou_thresh=0.5)
    assert keep == []


def test_nms_single_box_returns_index_zero():
    boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
    scores = np.array([0.9], dtype=np.float32)
    keep = nms(boxes, scores, iou_thresh=0.5)
    assert keep == [0]


def test_nms_non_overlapping_boxes_keeps_all():
    boxes = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    keep = nms(boxes, scores, iou_thresh=0.5)
    assert set(keep) == {0, 1}


def test_nms_overlapping_boxes_keeps_highest_score():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 9, 9], [20, 20, 30, 30]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    keep = nms(boxes, scores, iou_thresh=0.5)
    assert 0 in keep
    assert 1 not in keep
    assert 2 in keep


def test_nms_filters_zero_size_boxes():
    boxes = np.array([[0, 0, 0, 10], [0, 0, 10, 0], [0, 0, 10, 10]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    keep = nms(boxes, scores, iou_thresh=0.5)
    assert keep == [2]
