import os
import shutil
from pathlib import Path

import cv2
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
    global global_counter

    new_image_name = f"{global_counter}{image_path.suffix.lower()}"
    new_label_name = f"{global_counter}.txt"

    new_image_path = dest_images / new_image_name
    new_label_path = dest_labels / new_label_name

    shutil.move(str(image_path), str(new_image_path))
    shutil.move(str(label_path), str(new_label_path))

    global_counter += 1


def rename_files():
    global global_counter
    for fname in os.listdir(TRAIN_IMAGES):
        img = TRAIN_IMAGES / fname
        lbl = TRAIN_LABELS / f"{Path(fname).stem}.txt"
        rename_and_merge(img, lbl, TRAIN_IMAGES, TRAIN_LABELS)

    for fname in os.listdir(VAL_IMAGES):
        img = VAL_IMAGES / fname
        lbl = VAL_LABELS / f"{Path(fname).stem}.txt"
        rename_and_merge(img, lbl, VAL_IMAGES, VAL_LABELS)

    for fname in os.listdir(TEST_IMAGES):
        img = TEST_IMAGES / fname
        lbl = TEST_LABELS / f"{Path(fname).stem}.txt"
        rename_and_merge(img, lbl, TEST_IMAGES, TEST_LABELS)


def visualize(img_path):
    p = Path(img_path)
    lbl_path = p.parent.parent / "labels" / f"{p.stem}.txt"
    print(p)
    img = cv2.imread(str(p))
    h, w = img.shape[:2]
    with open(lbl_path, "r") as f:
        for line in f:
            c, cx, cy, bw, bh = map(float, line.split()[:5])
            x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
            x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow("vis", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_labels(dir_path, unwanted_ids=None, replace_map=None):
    p = Path(dir_path)
    unwanted_ids = set(unwanted_ids or [])
    replace_map = replace_map or {}

    for txt in p.glob("*.txt"):
        new_lines = []
        with open(txt, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    cid = int(float(parts[0]))
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
    process_labels(dir_path, unwanted_ids=unwanted_ids)


def change_label(dir_path, old_id, new_id):
    process_labels(dir_path, replace_map={old_id: new_id})


def xyxy_to_yolo(x1, y1, x2, y2, w, h):
    bw, bh = (x2 - x1) / w, (y2 - y1) / h
    cx, cy = (x1 + x2) / (2 * w), (y1 + y2) / (2 * h)
    return cx, cy, bw, bh


def add_person_labels(image_path: Path, label_path: Path, model: YOLO, target_class_id: int = 1,
                      source_class_id: int = 0, conf: float = 0.4, iou: float = 0.45):
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
    test_size = len(os.listdir(images_path))
    for i, fname in enumerate(os.listdir(images_path)):
        img_path = images_path / fname
        lbl_path = labels_path / f"{Path(fname).stem}.txt"
        add_person_labels(img_path, lbl_path, model)
        print(f"{images_path}: {i + 1}/{test_size}")


if __name__ == '__main__':
    rename_files()
    print("Files have been renamed")
    clear_labels(TRAIN_LABELS, [0, 2])
    clear_labels(VAL_LABELS, [0, 2])
    clear_labels(TEST_LABELS, [0, 2])
    change_label(TRAIN_LABELS, 1, 0)
    change_label(VAL_LABELS, 1, 0)
    change_label(TEST_LABELS, 1, 0)
    print("Labels have been changed")
    model = YOLO("yolov8l.pt")

    reannotate_persons(TRAIN_IMAGES, TRAIN_LABELS)
    reannotate_persons(VAL_IMAGES, VAL_LABELS)
    reannotate_persons(TEST_IMAGES, TEST_LABELS)
