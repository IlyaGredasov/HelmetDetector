import os
import shutil
from pathlib import Path
import cv2

global_counter = 0

DATASET = Path("dataset")
TRAIN_IMAGES = DATASET / "train" / "images"
TRAIN_LABELS = DATASET / "train" / "labels"
VAL_IMAGES   = DATASET / "valid"   / "images"
VAL_LABELS   = DATASET / "valid"   / "labels"
TEST_IMAGES  = DATASET / "test"  / "images"
TEST_LABELS  = DATASET / "test"  / "labels"

def rename_and_merge(image_path: Path, label_path: Path, dest_images: Path, dest_labels: Path):
    global global_counter
    
    new_image_name = f"{global_counter}{image_path.suffix.lower()}"
    new_label_name = f"{global_counter}.txt"

    new_image_path = dest_images / new_image_name
    new_label_path = dest_labels / new_label_name

    shutil.move(str(image_path), str(new_image_path))
    shutil.move(str(label_path), str(new_label_path))

    global_counter += 1

def reannotate_dataset():
    global global_counter
    for fname in os.listdir(TRAIN_IMAGES):
        img = TRAIN_IMAGES / fname
        lbl = TRAIN_LABELS / f"{Path(fname).stem}.txt"
        rename_and_merge(img, lbl, TRAIN_IMAGES, TRAIN_LABELS)

    for fname in os.listdir(VAL_IMAGES):
        img = VAL_IMAGES / fname
        lbl = VAL_LABELS / f"{Path(fname).stem}.txt"
        rename_and_merge(img, lbl, TRAIN_IMAGES, TRAIN_LABELS)

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
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
    cv2.imshow("vis", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def clear_labels(dir_path, ids_to_remove):
    p = Path(dir_path)
    for txt in p.glob("*.txt"):
        with open(txt, "r", encoding="utf-8") as f:
            lines = f.readlines()
        kept = []
        for line in lines:
            parts = line.strip().split()
            if not parts: 
                continue
            try:
                cid = int(float(parts[0]))
            except:
                continue
            if cid not in ids_to_remove:
                kept.append(line)
        with open(txt, "w", encoding="utf-8") as f:
            f.writelines(kept)

def change_label(dir_path, old_id, new_id):
    p = Path(dir_path)
    for txt in p.glob("*.txt"):
        out = []
        with open(txt, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    cid = int(float(parts[0]))
                except:
                    continue
                if cid == old_id:
                    parts[0] = str(new_id)
                    line = " ".join(parts) + "\n"
                out.append(line)
        with open(txt, "w", encoding="utf-8") as f:
            f.writelines(out)