import os
import shutil
from pathlib import Path
from typing import List
from typing import Tuple

import cv2
import numpy as np
from ultralytics.models import YOLO

global_counter = 0

DATASET = Path("dataset")
TRAIN_IMAGES = DATASET / "train" / "images"
TRAIN_LABELS = DATASET / "train" / "labels"
VAL_IMAGES = DATASET / "valid" / "images"
VAL_LABELS = DATASET / "valid" / "labels"
TEST_IMAGES = DATASET / "test" / "images"
TEST_LABELS = DATASET / "test" / "labels"


def rename_and_merge(image_path: Path, label_path: Path, dest_images: Path, dest_labels: Path):
    """
    Переименовывает пару (image + label) в общий формат.

    :param image_path: путь к изображению
    :param label_path: путь к меткам
    :param dest_images: директория назначения для изображений
    :param dest_labels: директория назначения для меток
    :return: None
    """
    global global_counter

    new_image_name = f"{global_counter}{image_path.suffix.lower()}"
    new_label_name = f"{global_counter}.txt"

    new_image_path = dest_images / new_image_name
    new_label_path = dest_labels / new_label_name

    shutil.move(str(image_path), str(new_image_path))
    shutil.move(str(label_path), str(new_label_path))

    global_counter += 1


def rename_files():
    """
    Переименовывает все данные в train/valid/test наборах.

    :return: None
    """
    global global_counter
    for file_name in os.listdir(TRAIN_IMAGES):
        img = TRAIN_IMAGES / file_name
        lbl = TRAIN_LABELS / f"{Path(file_name).stem}.txt"
        rename_and_merge(img, lbl, TRAIN_IMAGES, TRAIN_LABELS)

    for file_name in os.listdir(VAL_IMAGES):
        img = VAL_IMAGES / file_name
        lbl = VAL_LABELS / f"{Path(file_name).stem}.txt"
        rename_and_merge(img, lbl, VAL_IMAGES, VAL_LABELS)

    for file_name in os.listdir(TEST_IMAGES):
        img = TEST_IMAGES / file_name
        lbl = TEST_LABELS / f"{Path(file_name).stem}.txt"
        rename_and_merge(img, lbl, TEST_IMAGES, TEST_LABELS)


def visualize(img_path, save_path=None):
    """
    Визуализирует разметку YOLO на изображении.

    :param img_path: путь к изображению
    :param save_path: если указан — сохранит результат
    :return: изображение с отрисованными метками
    """
    p = Path(img_path)
    lbl_path = p.parent.parent / "labels" / f"{p.stem}.txt"
    img = cv2.imread(str(p))
    h, w = img.shape[:2]
    if lbl_path.exists():
        with open(lbl_path, "r", encoding="utf-8") as f:
            for line in f:
                c, cx, cy, bw, bh = map(float, line.split()[:5])
                x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
                x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
                color = (0, 255, 0) if c else (0, 0, 255)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                txt = str(int(c))
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                ty = max(0, y1 - 4)
                cv2.rectangle(img, (x1, ty - th - 6), (x1 + tw + 6, ty), color, -1)
                cv2.putText(img, txt, (x1 + 3, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), img)
    else:
        cv2.imshow("vis", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return img


def visualize_dataset():
    """
    Визуализирует весь датасет (train/valid/test).

    :return: None
    """
    for split in ['train', 'valid', 'test']:
        img_dir = DATASET / split / "images"
        if not img_dir.exists():
            continue
        for file_name in os.listdir(img_dir):
            src = img_dir / file_name
            dst = Path("visualized_dataset") / split / file_name
            visualize(src, save_path=dst)


def process_labels(dir_path, unwanted_ids=None, replace_map=None):
    """
    Массово изменяет метки YOLO.

    :param dir_path: директория с .txt файлами
    :param unwanted_ids: классы для удаления
    :param replace_map: таблица замен классов
    :return: None
    """
    p = Path(dir_path)
    unwanted_ids = set(unwanted_ids or [])
    replace_map = replace_map or {}

    for txt in p.glob("*.txt"):
        new_lines = []
        with open(txt, "r", encoding="utf-8") as f:
            for line in f:
                parts = list(line.strip().split())
                if not parts:
                    continue
                try:
                    cid = int(parts[0])
                except:
                    continue
                if cid in unwanted_ids:
                    continue
                if cid in replace_map:
                    parts[0] = str(replace_map[cid])
                new_lines.append(" ".join(parts) + "\n")
        with open(txt, "w", encoding="utf-8") as f:
            f.writelines(new_lines)


def clear_labels(dir_path, unwanted_ids):
    """
    Удаляет выбранные классы из всех меток.

    :param dir_path: директория
    :param unwanted_ids: классы для удаления
    :return: None
    """
    process_labels(dir_path, unwanted_ids=unwanted_ids)


def change_label(dir_path, old_id, new_id):
    """
    Заменяет один класс на другой.

    :param dir_path: директория
    :param old_id: исходный класс
    :param new_id: новый класс
    :return: None
    """
    process_labels(dir_path, replace_map={old_id: new_id})


def xyxy_to_yolo(x1, y1, x2, y2, w, h):
    """
    Переводит координаты из одного формата, в другой (xyxy -> xywh)

    :param x1: левый X
    :param y1: верхний Y
    :param x2: правый X
    :param y2: нижний Y
    :param w: ширина изображения
    :param h: высота изображения
    :return: координаты YOLO
    """
    bw, bh = (x2 - x1) / w, (y2 - y1) / h
    cx, cy = (x1 + x2) / (2 * w), (y1 + y2) / (2 * h)
    return cx, cy, bw, bh


def add_person_labels(image_path: Path, label_path: Path, model: YOLO, target_class_id: int = 1,
                      source_class_id: int = 0, conf: float = 0.4, iou: float = 0.45):
    """
    Добавляет разметку класса "person" к изображению (с использованием YOLO).

    :param image_path: путь к изображению
    :param label_path: путь к .txt меткам
    :param model: модель YOLO
    :param target_class_id: ID класса назначения
    :param source_class_id: какой класс берем из модели
    :param conf: порог уверенности
    :param iou: порог NMS
    :return: None
    """
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]
    res = model.predict(source=str(image_path), conf=conf, iou=iou, verbose=False)[0]
    with open(str(label_path), "a", encoding="utf-8") as f:
        for b in res.boxes:
            if int(b.cls.item()) != source_class_id:
                continue
            x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
            cx, cy, bw, bh = xyxy_to_yolo(x1, y1, x2, y2, w, h)
            f.write(f"{target_class_id} {cx} {cy} {bw} {bh}\n")


def reannotate_persons(images_path, labels_path):
    """
    Переаннотирует весь датасет классом "person".

    :param images_path: директория с изображениями
    :param labels_path: директория с метками
    :return: None
    """
    test_size = len(os.listdir(images_path))
    for i, file_name in enumerate(os.listdir(images_path)):
        img_path = images_path / file_name
        lbl_path = labels_path / f"{Path(file_name).stem}.txt"
        add_person_labels(img_path, lbl_path, YOLO("models/yolov8l.pt"))
        print(f"{images_path}: {i + 1}/{test_size}")


def letterbox(img: np.ndarray, new_shape: Tuple[int, int]) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Выполняет letterbox-resize изображения.

    :param img: входное изображение
    :param new_shape: требуемый размер (h, w)
    :return: измененное изображение, масштаб, паддинги
    """
    h, w = img.shape[:2]
    if (h, w) == new_shape:
        return img, 1.0, (0, 0)
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
    """
    Выполняет классическое NMS (Non-Maximum Suppression).

    :param boxes: массив боксов
    :param scores: уверенности
    :param iou_thresh: порог IoU
    :return: индексы выбранных боксов
    """
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


if __name__ == '__main__':
    rename_files()
    print("Files have been renamed")
    visualize_dataset()
    print("Dataset has been visualized")
