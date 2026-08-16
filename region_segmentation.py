"""
Region Segmentation .

This uses CLIPSeg, an open-vocabulary
segmentation model. You give it ANY text prompt ("grass", "the sun", "sky",
"a person running") and it returns a mask of where that thing is in the image.
This is what lets the app respond to arbitrary commands instead of a fixed
list of regions.

Falls back to a simple color-based heuristic for a few common outdoor
elements if the model isn't installed / fails to load (e.g. no internet
to download weights) — so the pipeline still runs in a degraded mode.
"""

import numpy as np
import cv2

_clipseg_processor = None
_clipseg_model = None


def _load_clipseg():
    """Lazy-load CLIPSeg the first time it's needed (weights are ~500MB)."""
    global _clipseg_processor, _clipseg_model
    if _clipseg_model is not None:
        return _clipseg_processor, _clipseg_model

    from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
    _clipseg_processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    _clipseg_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
    _clipseg_model.eval()
    return _clipseg_processor, _clipseg_model


def get_mask_clipseg(image_bgr: np.ndarray, text_prompt: str, threshold: float = 0.4) -> np.ndarray:
    """
    image_bgr: numpy array (BGR, as read by cv2.imread)
    text_prompt: any noun phrase, e.g. "grass", "the sun", "sky", "person"
    returns: uint8 mask, same H x W as image, values 0 or 255
    """
    import torch
    from PIL import Image

    processor, model = _load_clipseg()

    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    inputs = processor(text=[text_prompt], images=[pil_img], return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    # outputs.logits: (1, H', W') low-res heatmap -> resize to original image size
    logits = outputs.logits.unsqueeze(0)  # (1, 1, H', W')
    probs = torch.sigmoid(logits)[0, 0].numpy()
    probs_resized = cv2.resize(probs, (w, h))

    mask = (probs_resized > threshold).astype(np.uint8) * 255

    # smooth small holes/noise
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


# --- Fallback: simple color-range heuristics for common outdoor scene elements ---
# Used only if CLIPSeg can't load. Rough, but keeps the app functional offline.
_COLOR_HEURISTICS = {
    "grass": [(np.array([30, 40, 30]), np.array([90, 255, 255]))],   # green range (HSV)
    "sky": [(np.array([90, 40, 60]), np.array([130, 255, 255]))],    # blue range
    "sun": None,  # handled specially: brightest small blob
}


def get_mask_color_heuristic(image_bgr: np.ndarray, target: str) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    if target == "sun":
        # brightest, most saturated-white blob = crude "sun" proxy
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        if mask.sum() == 0:  # nothing that bright, relax threshold
            _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        return mask

    ranges = _COLOR_HEURISTICS.get(target)
    if ranges is None:
        # unknown target and no model available -> no region found
        return np.zeros((h, w), dtype=np.uint8)

    mask = np.zeros((h, w), dtype=np.uint8)
    for lower, upper in ranges:
        mask |= cv2.inRange(hsv, lower, upper)
    return mask


def get_mask(image_bgr: np.ndarray, target: str, use_model: bool = True) -> np.ndarray:
    """
    Main entry point: get a mask for `target` (e.g. "grass", "the sun", "sky").
    Tries CLIPSeg first (works for ANY target), falls back to color heuristics.
    """
    if use_model:
        try:
            return get_mask_clipseg(image_bgr, target)
        except Exception as e:
            print(f"⚠️  CLIPSeg unavailable ({e}), falling back to color heuristic")

    return get_mask_color_heuristic(image_bgr, target)


def draw_mask_overlay(image_bgr: np.ndarray, mask: np.ndarray, color=(0, 255, 0), alpha=0.4):
    """Debug helper: overlay the mask on the image so you can see what got selected."""
    overlay = image_bgr.copy()
    colored = np.zeros_like(image_bgr)
    colored[:] = color
    mask_bool = mask.astype(bool)
    overlay[mask_bool] = cv2.addWeighted(image_bgr, 1 - alpha, colored, alpha, 0)[mask_bool]
    return overlay


if __name__ == "__main__":
    img = cv2.imread("images/sample.jpg")
    mask = get_mask(img, "grass")
    cv2.imwrite("outputs/mask_preview.jpg", mask)
    cv2.imwrite("outputs/mask_overlay.jpg", draw_mask_overlay(img, mask))