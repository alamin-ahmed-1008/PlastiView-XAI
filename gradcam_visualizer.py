"""
Microplastic GradCAM Visualizer
=================================
Uses pytorch-grad-cam library which correctly handles Faster R-CNN.

Install first:
    pip install grad-cam

Usage in Colab:
    exec(open('gradcam_visualizer.py').read())
    result = generate_gradcam("/path/to/image.jpg")
    show(result)
"""

import os
import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import ToTensor
from torchvision.models.detection import fasterrcnn_resnet50_fpn

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
except ImportError:
    raise ImportError(
        "\n[ERROR] pytorch-grad-cam not installed.\n"
        "Run this first:  !pip install grad-cam\n"
    )

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
SCORE_THRESH = 0.3
NUM_CLASSES  = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"[INFO] Device: {DEVICE}")


# ─────────────────────────────────────────────
# FASTER R-CNN WRAPPER FOR GRAD-CAM
# pytorch-grad-cam needs a model that returns a single tensor,
# not a dict. This wrapper extracts the classification logits.
# ─────────────────────────────────────────────
class FasterRCNNGradCAMWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # x: [1, 3, H, W]
        images, _ = self.model.transform([x[0]], None)
        features  = self.model.backbone(images.tensors)

        # RPN proposals
        proposals, _ = self.model.rpn(images, features, None)

        # ROI pool + head
        box_features = self.model.roi_heads.box_roi_pool(
            features, proposals, images.image_sizes
        )
        box_features = self.model.roi_heads.box_head(box_features)
        cls_logits, _ = self.model.roi_heads.box_predictor(box_features)

        # Return class-1 (Microplastic) scores — shape [N_proposals, 1]
        # GradCAM will backprop through the mean of this
        return cls_logits[:, 1:2]   # [N, 1]


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
def load_model():
    model = fasterrcnn_resnet50_fpn(
        weights=None, weights_backbone=None, num_classes=NUM_CLASSES
    )
    state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    print("[INFO] Model loaded ✓")
    return model


# ─────────────────────────────────────────────
# FONT HELPER
# ─────────────────────────────────────────────
def get_font(size=14):
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
# DETECTION (standard, no grad)
# ─────────────────────────────────────────────
def detect(model, image_path):
    img    = Image.open(image_path).convert("RGB")
    tensor = ToTensor()(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(tensor)[0]
    mask   = out["scores"] > SCORE_THRESH
    boxes  = out["boxes"][mask].cpu().numpy()
    scores = out["scores"][mask].cpu().numpy()
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
        tb        = draw.textbbox((lx, ly), label, font=font)
        draw.rectangle(tb, fill=(0, 220, 110, 210))
        draw.text((lx, ly), label, fill=(10, 10, 10, 255), font=font)

    out  = Image.alpha_composite(img_draw, overlay).convert("RGB")
    d    = ImageDraw.Draw(out)
    w, _ = out.size
    d.rectangle([0, 0, w, 28], fill=(10, 12, 30))
    d.text((8, 6), f"  Microplastics detected: {len(boxes)}",
           fill=(0, 220, 110), font=font_lg)
    return out


# ─────────────────────────────────────────────
# GRAD-CAM HEATMAP
# ─────────────────────────────────────────────
def compute_gradcam(model, pil_img):
    """
    Uses pytorch-grad-cam with the FasterRCNNGradCAMWrapper.
    Target layer: layer4[-1]  (last ResNet block before FPN).
    """
    wrapper     = FasterRCNNGradCAMWrapper(model).to(DEVICE)
    target_layer = [model.backbone.body.layer4[-1]]

    # pytorch-grad-cam expects float32 numpy [H, W, 3] in [0,1] for display
    img_np  = np.array(pil_img).astype(np.float32) / 255.0
    img_t   = ToTensor()(pil_img).unsqueeze(0).to(DEVICE)

    # targets=None → GradCAM uses the highest scoring output automatically
    cam = GradCAM(model=wrapper, target_layers=target_layer)
    grayscale_cam = cam(input_tensor=img_t, targets=None)  # [1, H, W]
    grayscale_cam = grayscale_cam[0]                       # [H, W]

    # Overlay using library helper
    visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

    # Add label
    result = visualization.copy()
    cv2.putText(result, "Grad-CAM Heatmap", (10, 26),
                cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    return Image.fromarray(result), grayscale_cam


# ─────────────────────────────────────────────
# COLLAGE
# ─────────────────────────────────────────────
def make_collage(original, detected, gradcam_img):
    target_h = original.size[1]

    def fit(img):
        r = target_h / img.size[1]
        return img.resize((int(img.size[0] * r), target_h), Image.LANCZOS)

    o = fit(original)
    d = fit(detected)
    g = fit(gradcam_img)

    gap     = 4
    total_w = o.width + d.width + g.width + gap * 2
    hdr     = 32
    canvas  = Image.new("RGB", (total_w, target_h + hdr), (12, 14, 22))
    canvas.paste(o, (0,                        hdr))
    canvas.paste(d, (o.width + gap,            hdr))
    canvas.paste(g, (o.width + d.width + gap*2, hdr))

    draw   = ImageDraw.Draw(canvas)
    font   = get_font(14)
    panels = [
        ("Original",  0,                        (190, 190, 190)),
        ("Detection", o.width + gap,             (0,   220, 110)),
        ("Grad-CAM",  o.width + d.width + gap*2, (80,  160, 255)),
    ]
    for label, x, color in panels:
        draw.text((x + 10, 9), label, fill=color, font=font)
    return canvas


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def generate_gradcam(image_path, output_dir=OUTPUT_DIR):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    os.makedirs(output_dir, exist_ok=True)
    base  = os.path.splitext(os.path.basename(image_path))[0]
    model = load_model()

    # Detection
    print("[INFO] Detecting ...")
    pil_img, boxes, scores = detect(model, image_path)
    print(f"[INFO] {len(boxes)} microplastic(s) detected")

    det_img  = draw_detections(pil_img, boxes, scores)
    det_path = os.path.join(output_dir, f"{base}_detected.jpg")
    det_img.save(det_path, quality=95)
    print(f"[SAVED] {det_path}")

    # GradCAM
    print("[INFO] Computing Grad-CAM ...")
    gcam_img, _  = compute_gradcam(model, pil_img)
    gcam_path    = os.path.join(output_dir, f"{base}_gradcam.jpg")
    gcam_img.save(gcam_path, quality=95)
    print(f"[SAVED] {gcam_path}")

    # Collage
    collage   = make_collage(pil_img, det_img, gcam_img)
    sbs_path  = os.path.join(output_dir, f"{base}_side_by_side.jpg")
    collage.save(sbs_path, quality=95)
    print(f"[SAVED] {sbs_path}")

    return {
        "detected":     det_path,
        "gradcam":      gcam_path,
        "side_by_side": sbs_path,
        "count":        len(boxes),
        "scores":       scores.tolist(),
    }


def show(result):
    if not IN_COLAB:
        print("Open saved files:", result)
        return
    print(f"\n📊 Detected: {result['count']} microplastic(s)")
    print(f"   Scores: {[f'{s:.3f}' for s in result['scores']]}")
    print("\n🔍 Detection:")
    ipy_display(IpyImage(filename=result["detected"]))
    print("\n🌡️  Grad-CAM:")
    ipy_display(IpyImage(filename=result["gradcam"]))
    print("\n📋 Side-by-side:")
    ipy_display(IpyImage(filename=result["side_by_side"]))


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    TEST_IMAGE = os.path.join(BASE_DIR, "valid",
                              "a--3-_jpg.rf.8248ba99e3b3ae254d1723b674f7fd99.jpg")
    result = generate_gradcam(TEST_IMAGE)
    show(result)