"""
Microplastic Detection - Inference + GradCAM Visualization
============================================================
Google Colab version

Outputs saved to:
  /content/drive/MyDrive/Colab Notebooks/Dataset/dataset/outputs/
    ├── <name>_detected.jpg      → bounding boxes on image
    ├── <name>_gradcam.jpg       → Grad-CAM heatmap overlay
    └── <name>_side_by_side.jpg  → Original | Detection | Grad-CAM

Usage (in a Colab cell):
    result = run_inference("/path/to/image.jpg")
    display_results(result)
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import ToTensor
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# ── Colab display helpers ──────────────────────────────────────
try:
    from IPython.display import display as ipy_display, Image as IpyImage
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# ─────────────────────────────────────────────
# CONFIG  ← edit these if your paths differ
# ─────────────────────────────────────────────
BASE_DIR    = "/content/drive/MyDrive/Colab Notebooks/Dataset/dataset"
MODEL_PATH  = os.path.join(BASE_DIR, "microplastic_fasterrcnn.pth")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
NUM_CLASSES  = 2
SCORE_THRESH = 0.5

# Device
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"[INFO] Device: {DEVICE}")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────────
def load_model(model_path=MODEL_PATH):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"\n[ERROR] Model not found: {model_path}\n"
            f"  → Run train_microplastic.py first and make sure the .pth file exists."
        )
    model = fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
        num_classes=NUM_CLASSES,
    )
    state = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    print(f"[INFO] Model loaded ✓  ({model_path})")
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

    # Font — Colab has DejaVu available
    for font_path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]:
        try:
            font       = ImageFont.truetype(font_path, 13)
            font_large = ImageFont.truetype(font_path, 17)
            break
        except Exception:
            font = font_large = ImageFont.load_default()

    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = [float(v) for v in box]
        # Box fill + outline
        draw.rectangle([x1, y1, x2, y2],
                       fill=(0, 220, 110, 35),
                       outline=(0, 220, 110, 230),
                       width=2)
        # Label
        label    = f"{score:.2f}"
        lx, ly   = x1 + 2, max(y1 - 17, 0)
        bbox_txt = draw.textbbox((lx, ly), label, font=font)
        draw.rectangle(bbox_txt, fill=(0, 220, 110, 210))
        draw.text((lx, ly), label, fill=(10, 10, 10, 255), font=font)

    result = Image.alpha_composite(img_draw, overlay).convert("RGB")

    # Top banner
    d      = ImageDraw.Draw(result)
    banner = f"  Microplastics detected: {len(boxes)}"
    w, _   = result.size
    d.rectangle([0, 0, w, 28], fill=(10, 12, 30))
    d.text((8, 6), banner, fill=(0, 220, 110), font=font_large)

    return result


# ─────────────────────────────────────────────
# GRAD-CAM
# ─────────────────────────────────────────────
class GradCAMExtractor:
    """
    Hooks into ResNet-50 layer4's last conv (conv3) inside the FPN backbone.
    Backpropagates through the FPN feature map mean as an objectness proxy.
    """

    def __init__(self, model):
        self.model       = model
        self.gradients   = None
        self.activations = None
        self._hooks      = []
        self._register()

    def _register(self):
        layer = self.model.backbone.body.layer4[-1].conv3

        def fwd(module, inp, out):
            self.activations = out.detach()

        def bwd(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self._hooks.append(layer.register_forward_hook(fwd))
        self._hooks.append(layer.register_full_backward_hook(bwd))

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()

    def compute(self, image_tensor):
        """Returns numpy heatmap [H, W] normalised to [0, 1]."""
        self.model.train()   # gradients need to flow

        img_t = image_tensor.to(DEVICE)

        # Forward through backbone only → get FPN feature maps
        features = self.model.backbone(img_t)

        # '0' = highest-resolution FPN level
        feat  = features["0"]
        score = feat.mean()           # scalar proxy for objectness

        self.model.zero_grad()
        score.backward()

        self.model.eval()

        grads = self.gradients    # [1, C, h, w]
        acts  = self.activations  # [1, C, h, w]

        if grads is None or acts is None:
            raise RuntimeError("[GradCAM] Gradients not captured — check hook registration.")

        weights = grads.mean(dim=(2, 3), keepdim=True)   # [1, C, 1, 1]
        cam     = F.relu((weights * acts).sum(dim=1))    # [1, h, w]
        cam     = cam.squeeze().cpu().numpy()             # [h, w]

        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        return cam   # [H, W] in [0, 1]


def apply_gradcam_overlay(pil_img, heatmap_np, alpha=0.45):
    """Blend JET colormap heatmap over the original image."""
    w, h = pil_img.size

    heat_resized  = cv2.resize(heatmap_np, (w, h))
    heat_color    = cv2.applyColorMap(np.uint8(255 * heat_resized), cv2.COLORMAP_JET)
    heat_color    = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)

    orig_np  = np.array(pil_img)
    blended  = cv2.addWeighted(orig_np, 1 - alpha, heat_color, alpha, 0)

    # Label
    cv2.putText(blended, "Grad-CAM Heatmap", (10, 26),
                cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    return Image.fromarray(blended)


# ─────────────────────────────────────────────
# SIDE-BY-SIDE COLLAGE
# ─────────────────────────────────────────────
def make_collage(original, detected, gradcam):
    """Stitch Original | Detection | Grad-CAM into one image."""
    target_h = original.size[1]   # use original height as reference

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

    # Paste panels
    canvas.paste(orig_r, (0,                                        header_h))
    canvas.paste(det_r,  (orig_r.width + gap,                       header_h))
    canvas.paste(gcam_r, (orig_r.width + det_r.width + gap * 2,    header_h))

    # Header labels
    draw = ImageDraw.Draw(canvas)
    for fpath in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]:
        try:
            font = ImageFont.truetype(fpath, 14); break
        except Exception:
            font = ImageFont.load_default()

    panels = [
        ("Original",  0,                                      (190, 190, 190)),
        ("Detection", orig_r.width + gap,                     (0,   220, 110)),
        ("Grad-CAM",  orig_r.width + det_r.width + gap * 2,   (80,  160, 255)),
    ]
    for label, x, color in panels:
        draw.text((x + 10, 9), label, fill=color, font=font)

    return canvas


# ─────────────────────────────────────────────
# COLAB DISPLAY HELPER
# ─────────────────────────────────────────────
def display_results(result):
    """Show outputs inline in Colab. Call after run_inference()."""
    if not IN_COLAB:
        print("[INFO] Not in Colab — open the saved files directly.")
        return

    print(f"\n📊 Microplastics detected: {result['count']}")
    if result['scores']:
        print(f"   Confidence scores: {[f'{s:.3f}' for s in result['scores']]}")
    else:
        print("   No detections above threshold.")

    print("\n🔍 Detection:")
    ipy_display(IpyImage(filename=result["detected"]))

    print("\n🌡️  Grad-CAM Heatmap:")
    ipy_display(IpyImage(filename=result["gradcam"]))

    print("\n📋 Side-by-side:")
    ipy_display(IpyImage(filename=result["side_by_side"]))


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────
def run_inference(image_path, output_dir=OUTPUT_DIR, score_thresh=SCORE_THRESH):
    """
    Run detection + Grad-CAM on a single image.

    Args:
        image_path  : path to input .jpg / .png
        output_dir  : where to save results (default: BASE_DIR/outputs/)
        score_thresh: confidence threshold 0–1

    Returns:
        dict with keys: detected, gradcam, side_by_side, count, scores
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]

    # ── Load model ───────────────────────────────
    model = load_model()

    # ── Detection ────────────────────────────────
    print("[INFO] Running detection ...")
    pil_img, boxes, scores = detect(model, image_path, score_thresh)
    print(f"[INFO] Detected {len(boxes)} microplastic(s)")

    detected_img = draw_detections(pil_img, boxes, scores)
    det_path     = os.path.join(output_dir, f"{base}_detected.jpg")
    detected_img.save(det_path, quality=95)
    print(f"[SAVED] {det_path}")

    # ── Grad-CAM ─────────────────────────────────
    print("[INFO] Computing Grad-CAM ...")
    extractor   = GradCAMExtractor(model)
    img_tensor  = ToTensor()(pil_img).unsqueeze(0)
    heatmap     = extractor.compute(img_tensor)
    extractor.remove_hooks()

    gradcam_img = apply_gradcam_overlay(pil_img, heatmap)
    gcam_path   = os.path.join(output_dir, f"{base}_gradcam.jpg")
    gradcam_img.save(gcam_path, quality=95)
    print(f"[SAVED] {gcam_path}")

    # ── Collage ──────────────────────────────────
    collage     = make_collage(pil_img, detected_img, gradcam_img)
    sbs_path    = os.path.join(output_dir, f"{base}_side_by_side.jpg")
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
# RUN  (edit the image path below and execute)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # ← Change this to any image in your valid/ folder
    TEST_IMAGE = os.path.join(BASE_DIR, "valid",
                              "a--3-_jpg.rf.8248ba99e3b3ae254d1723b674f7fd99.jpg")

    result = run_inference(TEST_IMAGE)
    display_results(result)