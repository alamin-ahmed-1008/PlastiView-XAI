"""
Fine-Tune Faster R-CNN on Combined Dataset
============================================
Step 2 of 3: Fine-tune your existing model using:
    • Original annotated training data  (Kaggle dataset)
    • Client pseudo-labeled data        (from auto_annotate_client_data.py)

Technique: Transfer learning with differential learning rates.
    - Backbone (ResNet-50): frozen for first N epochs, then unfrozen with
      a lower LR so existing feature representations are preserved.
    - Head (RPN + ROI):    always trained, higher LR for fast adaptation.

Usage in Colab:
    exec(open('finetune_with_client_data.py').read())
    finetune()
"""

import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision.transforms import ToTensor, Compose, ColorJitter, RandomHorizontalFlip
import torchvision.transforms.functional as TF
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.ops import box_iou
import random

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR         = "/content/drive/MyDrive/Colab Notebooks/Dataset/dataset"
TRAIN_DIR        = os.path.join(BASE_DIR, "train")
VALID_DIR        = os.path.join(BASE_DIR, "valid")
CLIENT_IMAGE_DIR = os.path.join(BASE_DIR, "personal-data")

TRAIN_CSV        = os.path.join(TRAIN_DIR,        "_annotations.csv")
VALID_CSV        = os.path.join(VALID_DIR,         "_annotations.csv")
CLIENT_CSV       = os.path.join(CLIENT_IMAGE_DIR,  "_annotations.csv")  # from Step 1

# Pretrained model from your original training
PRETRAINED_MODEL = os.path.join(BASE_DIR, "microplastic_fasterrcnn.pth")
# Output: fine-tuned model
FINETUNED_MODEL  = os.path.join(BASE_DIR, "microplastic_fasterrcnn_finetuned.pth")

FREEZE_EPOCHS    = 5     # freeze backbone for first N epochs
TOTAL_EPOCHS     = 25    # total fine-tuning epochs
BATCH_SIZE       = 8
LR_HEAD          = 0.002  # higher LR for the detection head
LR_BACKBONE      = 0.0002 # lower LR when backbone is unfrozen
SCORE_THRESH     = 0.5
NUM_CLASSES      = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device: {DEVICE}")


# ─────────────────────────────────────────────────────────────────────────────
# AUGMENTATION (extra augmentation for client data to compensate small size)
# ─────────────────────────────────────────────────────────────────────────────
class AugmentedMicroplasticDataset(Dataset):
    """
    Dataset that supports simple box-safe augmentations:
        - Random horizontal flip
        - Random vertical flip
        - Color jitter (brightness, contrast, saturation)
    Augmentation is applied with augment=True (for client data)
    or disabled (augment=False, for original data).
    """
    CLASS_MAP = {"Microplastic": 1}

    def __init__(self, img_dir, csv_path, augment=False):
        self.img_dir = img_dir
        self.augment = augment

        df = pd.read_csv(csv_path)

        # Drop rows with invalid boxes (can appear in pseudo-labels)
        df = df[(df["xmax"] > df["xmin"]) & (df["ymax"] > df["ymin"])].copy()

        self.df        = df
        self.filenames = df["filename"].unique()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname    = self.filenames[idx]
        img_path = os.path.join(self.img_dir, fname)
        img      = Image.open(img_path).convert("RGB")
        w, h     = img.size

        rows   = self.df[self.df["filename"] == fname]
        boxes  = []
        labels = []

        for _, row in rows.iterrows():
            boxes.append([
                float(row["xmin"]), float(row["ymin"]),
                float(row["xmax"]), float(row["ymax"])
            ])
            labels.append(self.CLASS_MAP.get(str(row["class"]).strip(), 1))

        boxes = torch.tensor(boxes, dtype=torch.float32)

        # ── Augmentations (box-safe) ─────────────────────────────────────
        if self.augment:
            # Horizontal flip
            if random.random() > 0.5:
                img    = TF.hflip(img)
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]

            # Vertical flip
            if random.random() > 0.5:
                img    = TF.vflip(img)
                boxes[:, [1, 3]] = h - boxes[:, [3, 1]]

            # Color jitter (image only, boxes unchanged)
            if random.random() > 0.3:
                img = TF.adjust_brightness(img, 0.7 + random.random() * 0.6)
            if random.random() > 0.3:
                img = TF.adjust_contrast(img, 0.7 + random.random() * 0.6)
            if random.random() > 0.3:
                img = TF.adjust_saturation(img, 0.7 + random.random() * 0.6)

        # Clamp boxes to image bounds
        boxes[:, 0].clamp_(0, w)
        boxes[:, 1].clamp_(0, h)
        boxes[:, 2].clamp_(0, w)
        boxes[:, 3].clamp_(0, h)

        # Remove degenerate boxes after augmentation
        keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes  = boxes[keep]
        labels = [l for l, k in zip(labels, keep.tolist()) if k]

        if len(boxes) == 0:
            # Fallback: whole image as dummy box (shouldn't happen often)
            boxes  = torch.tensor([[0, 0, float(w), float(h)]], dtype=torch.float32)
            labels = [1]

        target = {
            "boxes":    boxes,
            "labels":   torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
        }

        img_tensor = ToTensor()(img)
        return img_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))


# ─────────────────────────────────────────────────────────────────────────────
# METRICS (same as your original training script)
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(pred_boxes, gt_boxes, iou_thresh=0.5):
    if len(pred_boxes) == 0 and len(gt_boxes) == 0:
        return 1.0, 1.0, 1.0, 1.0
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return 0.0, 0.0, 0.0, 0.0
    ious    = box_iou(pred_boxes, gt_boxes)
    matched = set()
    tp      = 0
    for i in range(len(pred_boxes)):
        best_iou, best_j = 0, -1
        for j in range(len(gt_boxes)):
            if j in matched:
                continue
            if ious[i, j] > best_iou:
                best_iou, best_j = ious[i, j], j
        if best_iou >= iou_thresh:
            tp += 1
            matched.add(best_j)
    fp  = len(pred_boxes) - tp
    fn  = len(gt_boxes)   - tp
    pre = tp / (tp + fp + 1e-6)
    rec = tp / (tp + fn + 1e-6)
    f1  = 2 * pre * rec / (pre + rec + 1e-6)
    miou = ious.mean().item() if ious.numel() > 0 else 0.0
    return pre, rec, f1, miou


def validate(model, loader, device, score_thresh=0.5):
    model.eval()
    all_pre, all_rec, all_f1, all_iou = [], [], [], []
    with torch.no_grad():
        for images, targets in loader:
            images  = [img.to(device) for img in images]
            outputs = model(images)
            for out, tgt in zip(outputs, targets):
                pred_boxes = out["boxes"][out["scores"] > score_thresh].cpu()
                gt_boxes   = tgt["boxes"].cpu()
                pre, rec, f1, iou = compute_metrics(pred_boxes, gt_boxes)
                all_pre.append(pre)
                all_rec.append(rec)
                all_f1.append(f1)
                all_iou.append(iou)
    model.train()
    return (
        float(np.mean(all_pre)), float(np.mean(all_rec)),
        float(np.mean(all_f1)), float(np.mean(all_iou)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# FINE-TUNING
# ─────────────────────────────────────────────────────────────────────────────
def set_backbone_trainable(model, trainable: bool):
    """Freeze or unfreeze the ResNet-50 backbone."""
    for name, param in model.named_parameters():
        if "backbone" in name:
            param.requires_grad = trainable
    status = "UNFROZEN (trainable)" if trainable else "FROZEN"
    print(f"  [INFO] Backbone: {status}")


def finetune():
    print("\n" + "="*60)
    print(" FINE-TUNING: Combined Dataset")
    print("="*60)

    # ── Check client CSV exists ────────────────────────────────────────────
    if not os.path.isfile(CLIENT_CSV):
        raise FileNotFoundError(
            f"\n[ERROR] Client annotation CSV not found: {CLIENT_CSV}\n"
            f"Run auto_annotate_client_data.py first (Step 1)."
        )

    # ── Datasets ───────────────────────────────────────────────────────────
    original_ds = AugmentedMicroplasticDataset(TRAIN_DIR,        TRAIN_CSV,  augment=False)
    client_ds   = AugmentedMicroplasticDataset(CLIENT_IMAGE_DIR, CLIENT_CSV, augment=True)
    valid_ds    = AugmentedMicroplasticDataset(VALID_DIR,        VALID_CSV,  augment=False)

    # Combine: original + client (client is repeated 3× to balance dataset size)
    client_repeated = ConcatDataset([client_ds] * 3)
    combined_ds     = ConcatDataset([original_ds, client_repeated])

    train_loader = DataLoader(combined_ds, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_fn,
                              num_workers=2, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=1,
                              shuffle=False, collate_fn=collate_fn)

    print(f"[INFO] Original train samples : {len(original_ds)}")
    print(f"[INFO] Client samples (raw)   : {len(client_ds)}")
    print(f"[INFO] Client samples (×3)    : {len(client_repeated)}")
    print(f"[INFO] Combined train samples : {len(combined_ds)}")
    print(f"[INFO] Validation samples     : {len(valid_ds)}")

    # ── Load pretrained model ──────────────────────────────────────────────
    model = fasterrcnn_resnet50_fpn(
        weights=None, weights_backbone=None, num_classes=NUM_CLASSES
    )
    state = torch.load(PRETRAINED_MODEL, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state)
    model.to(DEVICE)
    print(f"\n[INFO] Loaded pretrained weights from {PRETRAINED_MODEL}")

    # Phase 1: Freeze backbone, train head only
    set_backbone_trainable(model, trainable=False)
    head_params = [p for p in model.parameters() if p.requires_grad]
    optimizer   = torch.optim.SGD(head_params, lr=LR_HEAD,
                                  momentum=0.9, weight_decay=5e-4)
    scheduler   = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

    best_f1 = 0.0
    history = []

    print(f"\n[PHASE 1] Epochs 1–{FREEZE_EPOCHS}: backbone frozen, training head only")
    print("="*60)

    for epoch in range(1, TOTAL_EPOCHS + 1):

        # Switch to Phase 2: unfreeze backbone
        if epoch == FREEZE_EPOCHS + 1:
            print(f"\n[PHASE 2] Epoch {epoch}: unfreezing backbone with lower LR")
            set_backbone_trainable(model, trainable=True)
            optimizer = torch.optim.SGD([
                {"params": [p for n, p in model.named_parameters() if "backbone" in n],
                 "lr": LR_BACKBONE},
                {"params": [p for n, p in model.named_parameters() if "backbone" not in n],
                 "lr": LR_HEAD},
            ], momentum=0.9, weight_decay=5e-4)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)
            print("="*60)

        model.train()
        epoch_loss = 0.0

        for images, targets in train_loader:
            images  = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss      = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)

        if epoch % 5 == 0 or epoch == TOTAL_EPOCHS or epoch == FREEZE_EPOCHS:
            pre, rec, f1, miou = validate(model, valid_loader, DEVICE, SCORE_THRESH)
            history.append({
                "epoch": epoch, "loss": avg_loss,
                "precision": pre, "recall": rec, "f1": f1, "mean_iou": miou
            })
            print(
                f"Epoch [{epoch:03d}/{TOTAL_EPOCHS}] "
                f"Loss: {avg_loss:.4f} | "
                f"P: {pre:.4f} R: {rec:.4f} F1: {f1:.4f} mIoU: {miou:.4f}"
            )
            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), FINETUNED_MODEL)
                print(f"  ✅ Best model saved → {FINETUNED_MODEL}  (F1={best_f1:.4f})")
        else:
            print(f"Epoch [{epoch:03d}/{TOTAL_EPOCHS}] Loss: {avg_loss:.4f}")

    # Save history
    hist_path = os.path.join(BASE_DIR, "finetune_history.csv")
    pd.DataFrame(history).to_csv(hist_path, index=False)

    print("="*60)
    print(f"[DONE] Fine-tuning complete")
    print(f"  Best F1    : {best_f1:.4f}")
    print(f"  Model path : {FINETUNED_MODEL}")
    print(f"  History    : {hist_path}")
    print("="*60)


if __name__ == "__main__":
    finetune()
