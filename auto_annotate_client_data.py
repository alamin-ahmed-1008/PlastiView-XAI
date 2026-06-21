"""
Auto-Annotator for Client Microplastic Images
==============================================
Step 1 of 3: Use your trained Faster R-CNN to generate
pseudo-labels (bounding boxes) for the client's unannotated images.

Output: client_data/_annotations.csv  (same format as training CSV)

Usage in Colab:
    !python auto_annotate_client_data.py

    OR in a cell:
    exec(open('auto_annotate_client_data.py').read())
    annotate_client_folder(CLIENT_IMAGE_DIR)
"""

import os
import csv
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  ← adjust these paths to your Colab environment
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR         = "/content/drive/MyDrive/Colab Notebooks/Dataset/dataset"
MODEL_PATH       = os.path.join(BASE_DIR, "microplastic_fasterrcnn.pth")

# Folder that contains client's raw .jpg images (no annotations yet)
CLIENT_IMAGE_DIR = os.path.join(BASE_DIR, "personal-data")

# Output CSV will be written here (same folder, matching training format)
CLIENT_CSV_OUT   = os.path.join(CLIENT_IMAGE_DIR, "_annotations.csv")

# Confidence threshold for keeping a detection as a pseudo-label.
# Lower  → more boxes kept (noisier labels, but more coverage)
# Higher → fewer boxes kept (cleaner labels, but might miss small particles)
SCORE_THRESH     = 0.35

NUM_CLASSES      = 2
CLASS_NAME       = "Microplastic"
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────


def load_model(model_path: str, num_classes: int = 2):
    """Load the saved Faster R-CNN weights."""
    model = fasterrcnn_resnet50_fpn(
        weights=None, weights_backbone=None, num_classes=num_classes
    )
    state = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    print(f"[INFO] Model loaded from {model_path}")
    return model


def get_image_files(folder: str) -> list[str]:
    """Return sorted list of image filenames in the folder."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    files = sorted(
        f for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in exts
    )
    return files


def predict_image(model, image_path: str, score_thresh: float):
    """
    Run inference on a single image.
    Returns: (width, height, boxes_list, scores_list)
        boxes_list: list of [xmin, ymin, xmax, ymax] as ints
    """
    img    = Image.open(image_path).convert("RGB")
    w, h   = img.size
    tensor = ToTensor()(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)[0]

    mask   = outputs["scores"] > score_thresh
    boxes  = outputs["boxes"][mask].cpu().numpy().astype(int)
    scores = outputs["scores"][mask].cpu().numpy()

    return w, h, boxes.tolist(), scores.tolist()


def annotate_client_folder(
    image_dir: str  = CLIENT_IMAGE_DIR,
    output_csv: str = CLIENT_CSV_OUT,
    score_thresh: float = SCORE_THRESH,
):
    """
    Main function: iterate over all images in image_dir, run inference,
    and save bounding boxes in CSV format matching the training annotation style.
    """
    print(f"\n{'='*60}")
    print(f"[INFO] Auto-annotating folder: {image_dir}")
    print(f"[INFO] Score threshold:        {score_thresh}")
    print(f"[INFO] Output CSV:             {output_csv}")
    print(f"{'='*60}\n")

    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"[ERROR] Image folder not found: {image_dir}")

    model      = load_model(MODEL_PATH)
    image_files = get_image_files(image_dir)

    if not image_files:
        raise RuntimeError(f"[ERROR] No image files found in {image_dir}")

    print(f"[INFO] Found {len(image_files)} image(s)\n")

    rows              = []
    total_detections  = 0
    skipped_images    = 0   # images with zero detections

    for i, fname in enumerate(image_files, 1):
        img_path = os.path.join(image_dir, fname)
        try:
            w, h, boxes, scores = predict_image(model, img_path, score_thresh)
        except Exception as e:
            print(f"  [WARN] Skipping {fname}: {e}")
            skipped_images += 1
            continue

        if len(boxes) == 0:
            # No detection → write one placeholder row with zeroed-out box
            # This keeps the image "in the dataset" but has no annotation.
            # Remove or keep as needed — comment the next 3 lines to exclude
            # zero-detection images from the CSV entirely.
            print(f"  [{i:03d}] {fname}  → 0 detections (image excluded from CSV)")
            skipped_images += 1
            continue

        for box, score in zip(boxes, scores):
            xmin, ymin, xmax, ymax = box
            # Clamp to image bounds (model sometimes predicts slightly outside)
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(w, xmax)
            ymax = min(h, ymax)
            rows.append({
                "filename": fname,
                "width":    w,
                "height":   h,
                "class":    CLASS_NAME,
                "xmin":     xmin,
                "ymin":     ymin,
                "xmax":     xmax,
                "ymax":     ymax,
                "score":    round(float(score), 4),   # extra column for review
            })

        total_detections += len(boxes)
        print(f"  [{i:03d}] {fname}  → {len(boxes)} detection(s)  "
              f"scores={[round(s,2) for s in scores]}")

    # ── Write CSV ──────────────────────────────────────────────────────────
    fieldnames = ["filename", "width", "height", "class",
                  "xmin", "ymin", "xmax", "ymax", "score"]

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'='*60}")
    print(f"[DONE] Auto-annotation complete")
    print(f"  Images processed : {len(image_files)}")
    print(f"  Images with boxes: {len(image_files) - skipped_images}")
    print(f"  Images skipped   : {skipped_images} (zero detections)")
    print(f"  Total rows       : {len(rows)}")
    print(f"  Total detections : {total_detections}")
    print(f"  CSV saved to     : {output_csv}")
    print(f"{'='*60}\n")

    return output_csv, rows


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    annotate_client_folder()
