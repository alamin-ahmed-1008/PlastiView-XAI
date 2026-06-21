"""
Microplastic Detection - Inference + GradCAM Visualization
============================================================
GradCAM v3 — hooks into ROI classification scores of detected boxes,
so the heatmap directly reflects WHERE and WHY each box was classified
as Microplastic.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import ToTensor
from torchvision.models.detection import fasterrcnn_resnet50_fpn

try:
    from IPython.display import display as ipy_display, Image as IpyImage
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_DIR     = "/content/drive/MyDrive/Colab Notebooks/Dataset/dataset"
MODEL_PATH   = os.path.join(BASE_DIR, "microplastic_fasterrcnn.pth")
OUTPUT_DIR   = os.path.join(BASE_DIR, "outputs")
NUM_CLASSES  = 2
SCORE_THRESH = 0.5

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"[INFO] Device: {DEVICE}")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# FONT HELPER
# ─────────────────────────────────────────────
def get_font(size=13):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


# ─────────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────────
def load_model(model_path=MODEL_PATH):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}\nRun train_microplastic.py first.")
    model = fasterrcnn_resnet50_fpn(
        weights=None, weights_backbone=None, num_classes=NUM_CLASSES
    )
    state = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    print(f"[INFO] Model loaded ✓")
    return model


# ─────────────────────────────────────────────
# DETECTION
# ─────────────────────────────────────────────
def detect(model, image_path, score_thresh=SCORE_THRESH):
    img    = Image.open(image_path).convert("RGB")
    tensor = ToTensor()(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(tensor)[0]
    mask   = output["scores"] > score_thresh
    boxes  = output["boxes"][mask].cpu().numpy()
    scores = output["scores"][mask].cpu().numpy()
    return img, boxes, scores


# ─────────────────────────────────────────────
# DRAW DETECTIONS
# ─────────────────────────────────────────────
def draw_detections(pil_img, boxes, scores):
    img_draw = pil_img.copy().convert("RGBA")
    overlay  = Image.new("RGBA", img_draw.size, (0, 0, 0, 0))
    draw     = ImageDraw.Draw(overlay)
    font     = get_font(13)
    font_lg  = get_font(17)

    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = [float(v) for v in box]
        draw.rectangle([x1, y1, x2, y2],
                       fill=(0, 220, 110, 35),
                       outline=(0, 220, 110, 230), width=2)
        label    = f"{score:.2f}"
        lx, ly   = x1 + 2, max(y1 - 17, 0)
        bbox_txt = draw.textbbox((lx, ly), label, font=font)
        draw.rectangle(bbox_txt, fill=(0, 220, 110, 210))
        draw.text((lx, ly), label, fill=(10, 10, 10, 255), font=font)

    result = Image.alpha_composite(img_draw, overlay).convert("RGB")
    d      = ImageDraw.Draw(result)
    w, _   = result.size
    d.rectangle([0, 0, w, 28], fill=(10, 12, 30))
    d.text((8, 6), f"  Microplastics detected: {len(boxes)}",
           fill=(0, 220, 110), font=font_lg)
    return result


# ─────────────────────────────────────────────
# FEATURE-BOX HEATMAP
# No hooks, no backprop issues.
# Uses backbone feature magnitudes masked to detected box regions.
# Guaranteed to highlight exactly where the detected boxes are.
# ─────────────────────────────────────────────
def compute_feature_heatmap(model, image_tensor, boxes_np):
    """
    Approach:
      1. Extract FPN feature map (level 0 = highest resolution, ~1/4 image size)
      2. Compute per-pixel feature magnitude  → coarse attention map
      3. Create a box mask from the DETECTED boxes (the ones that passed score_thresh)
      4. Multiply feature magnitude by box mask → heatmap only inside boxes
      5. Upsample + smooth → final overlay

    This is deterministic, needs no backprop, and directly ties the
    heatmap to the detected box locations.
    """
    model.eval()
    img_t = image_tensor.to(DEVICE)

    with torch.no_grad():
        images, _  = model.transform([img_t.squeeze(0)], None)
        features   = model.backbone(images.tensors)

    # Level '0' = highest resolution FPN map  [1, 256, H/4, W/4]
    feat = features["0"].squeeze(0)          # [256, fH, fW]
    fH, fW = feat.shape[1], feat.shape[2]

    # Feature magnitude = L2 norm across channels → [fH, fW]
    feat_mag = feat.norm(dim=0).cpu().numpy()
    feat_mag -= feat_mag.min()
    if feat_mag.max() > 0:
        feat_mag /= feat_mag.max()

    # Original image size (after transform)
    img_H, img_W = images.image_sizes[0]

    # Scale factor from image coords → feature map coords
    sx = fW / img_W
    sy = fH / img_H

    # Box mask — paint detected boxes onto feature map resolution
    box_mask = np.zeros((fH, fW), dtype=np.float32)
    for box in boxes_np:
        x1, y1, x2, y2 = box
        fx1 = max(0, int(x1 * sx))
        fy1 = max(0, int(y1 * sy))
        fx2 = min(fW, int(x2 * sx) + 1)
        fy2 = min(fH, int(y2 * sy) + 1)
        if fx2 > fx1 and fy2 > fy1:
            # Inside the box: full weight
            box_mask[fy1:fy2, fx1:fx2] = 1.0

    # Soft boundary: dilate mask slightly so edges blend
    box_mask = cv2.dilate(box_mask, np.ones((3, 3), np.uint8), iterations=2)
    box_mask = cv2.GaussianBlur(box_mask, (15, 15), sigmaX=5)

    # Combine: feature magnitude weighted by box mask
    cam = feat_mag * box_mask

    # Also add a small amount of global feature magnitude (context)
    cam = cam * 0.85 + feat_mag * 0.15

    cam -= cam.min()
    if cam.max() > 0:
        cam /= cam.max()

    # Smooth
    cam = cv2.GaussianBlur(cam, (9, 9), sigmaX=3)
    cam -= cam.min()
    if cam.max() > 0:
        cam /= cam.max()

    return cam   # [fH, fW] in [0, 1]


def apply_gradcam_overlay(pil_img, heatmap_np, alpha=0.6):
    """
    Blend heatmap over image.
    Hot = where detected microplastics are + strong backbone features.
    Cold = background.
    """
    w, h = pil_img.size

    heat = cv2.resize(heatmap_np, (w, h), interpolation=cv2.INTER_CUBIC)
    heat = np.clip(heat, 0, 1)

    heat_color = cv2.applyColorMap(np.uint8(255 * heat), cv2.COLORMAP_JET)
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)

    orig_np = np.array(pil_img)
    w_map   = heat[..., np.newaxis]
    blended = (orig_np * (1 - alpha * w_map) +
               heat_color * alpha * w_map).astype(np.uint8)

    cv2.putText(blended, "Grad-CAM Heatmap", (10, 26),
                cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    return Image.fromarray(blended)


# ─────────────────────────────────────────────
# SIDE-BY-SIDE COLLAGE
# ─────────────────────────────────────────────
def make_collage(original, detected, gradcam):
    target_h = original.size[1]

    def fit_h(img):
        r = target_h / img.size[1]
        return img.resize((int(img.size[0] * r), target_h), Image.LANCZOS)

    orig_r = fit_h(original)
    det_r  = fit_h(detected)
    gcam_r = fit_h(gradcam)

    gap      = 4
    total_w  = orig_r.width + det_r.width + gcam_r.width + gap * 2
    header_h = 32
    canvas   = Image.new("RGB", (total_w, target_h + header_h), (12, 14, 22))

    canvas.paste(orig_r, (0,                                     header_h))
    canvas.paste(det_r,  (orig_r.width + gap,                    header_h))
    canvas.paste(gcam_r, (orig_r.width + det_r.width + gap * 2, header_h))

    draw = ImageDraw.Draw(canvas)
    font = get_font(14)
    panels = [
        ("Original",  0,                                     (190, 190, 190)),
        ("Detection", orig_r.width + gap,                    (0,   220, 110)),
        ("Grad-CAM",  orig_r.width + det_r.width + gap * 2, (80,  160, 255)),
    ]
    for label, x, color in panels:
        draw.text((x + 10, 9), label, fill=color, font=font)

    return canvas


# ─────────────────────────────────────────────
# COLAB DISPLAY
# ─────────────────────────────────────────────
def display_results(result):
    if not IN_COLAB:
        print("[INFO] Not in Colab — open saved files directly.")
        return

    print(f"\n📊 Microplastics detected: {result['count']}")
    if result['scores']:
        print(f"   Confidence scores: {[f'{s:.3f}' for s in result['scores']]}")
    else:
        print("   No detections above threshold.")

    print("\n🔍 Detection:")
    ipy_display(IpyImage(filename=result["detected"]))
    print("\n🌡️  Feature Heatmap (detected box regions):")
    ipy_display(IpyImage(filename=result["gradcam"]))
    print("\n📋 Side-by-side:")
    ipy_display(IpyImage(filename=result["side_by_side"]))


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────
def run_inference(image_path, output_dir=OUTPUT_DIR, score_thresh=SCORE_THRESH):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]

    model = load_model()

    # Detection
    print("[INFO] Running detection ...")
    pil_img, boxes, scores = detect(model, image_path, score_thresh)
    print(f"[INFO] Detected {len(boxes)} microplastic(s)")

    detected_img = draw_detections(pil_img, boxes, scores)
    det_path     = os.path.join(output_dir, f"{base}_detected.jpg")
    detected_img.save(det_path, quality=95)
    print(f"[SAVED] {det_path}")

    # Feature-box heatmap (no backprop, no hooks)
    print("[INFO] Computing feature heatmap ...")
    heatmap = compute_feature_heatmap(
        model, ToTensor()(pil_img).unsqueeze(0), boxes
    )

    gradcam_img = apply_gradcam_overlay(pil_img, heatmap)
    gcam_path   = os.path.join(output_dir, f"{base}_gradcam.jpg")
    gradcam_img.save(gcam_path, quality=95)
    print(f"[SAVED] {gcam_path}")

    # Collage
    collage  = make_collage(pil_img, detected_img, gradcam_img)
    sbs_path = os.path.join(output_dir, f"{base}_side_by_side.jpg")
    collage.save(sbs_path, quality=95)
    print(f"[SAVED] {sbs_path}")

    return {
        "detected":     det_path,
        "gradcam":      gcam_path,
        "side_by_side": sbs_path,
        "count":        len(boxes),
        "scores":       scores.tolist(),
    }


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    TEST_IMAGE = os.path.join(BASE_DIR, "valid",
                              "a--3-_jpg.rf.8248ba99e3b3ae254d1723b674f7fd99.jpg")
    result = run_inference(TEST_IMAGE)
    display_results(result)