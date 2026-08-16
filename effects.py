"""
Effects Library
Every function takes (image_bgr, mask, **params) and returns an edited image.
`mask` is a uint8 0/255 array the same size as the image, or None for whole-image effects.

Design: edits are computed on the FULL image, then blended back in using the
(feathered) mask so edges look natural instead of hard-cut rectangles.
"""

import cv2
import numpy as np


def _feather(mask: np.ndarray, blur=15) -> np.ndarray:
    """Soften mask edges so edits blend smoothly instead of a hard cutout line."""
    if mask is None:
        return None
    m = cv2.GaussianBlur(mask, (blur, blur), 0)
    return (m.astype(np.float32) / 255.0)[..., None]  # HxWx1, 0..1


def _blend(original: np.ndarray, edited: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Composite `edited` over `original` using a feathered mask. mask=None -> full replace."""
    if mask is None:
        return edited
    alpha = _feather(mask)
    out = original.astype(np.float32) * (1 - alpha) + edited.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------- Basic adjustments ----------

def brightness(image, mask, value=40):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + value, 0, 255)
    edited = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return _blend(image, edited, mask)


def saturation(image, mask, value=-40):
    """Negative value = 'reduce color', positive = more vivid."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] + value, 0, 255)
    edited = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return _blend(image, edited, mask)


def contrast(image, mask, value=1.3):
    """value: multiplier, 1.0 = no change."""
    edited = cv2.convertScaleAbs(image, alpha=value, beta=0)
    return _blend(image, edited, mask)


def hue_shift(image, mask, degrees=20):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[:, :, 0] = (hsv[:, :, 0] + degrees) % 180
    edited = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return _blend(image, edited, mask)


# ---------- Blur / sharpen ----------

def blur_region(image, mask, ksize=25):
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    edited = cv2.GaussianBlur(image, (ksize, ksize), 0)
    return _blend(image, edited, mask)


def sharpen_region(image, mask):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    edited = cv2.filter2D(image, -1, kernel)
    return _blend(image, edited, mask)


def focus_effect(image, mask, blur_strength=31):
    """
    'Focus on X': sharpen the masked region, blur everything else.
    This is the depth-of-field look ("focus the sun" = sun crisp, rest soft).
    """
    ksize = blur_strength if blur_strength % 2 == 1 else blur_strength + 1
    background_blurred = cv2.GaussianBlur(image, (ksize, ksize), 0)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    subject_sharp = cv2.filter2D(image, -1, kernel)

    if mask is None:
        return subject_sharp  # nothing to contrast against

    alpha = _feather(mask, blur=9)
    out = background_blurred.astype(np.float32) * (1 - alpha) + subject_sharp.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------- Whole-image stylistic effects ----------

def motion_blur(image, mask=None, angle=0, kernel_size=25):
    """
    Directional blur to simulate motion/speed ("running effect").
    angle: 0 = horizontal streaks, 90 = vertical.
    Typically applied to the whole image or background (pass a mask to
    restrict it, e.g. blur everything EXCEPT a runner by inverting the mask).
    """
    k = np.zeros((kernel_size, kernel_size))
    k[kernel_size // 2, :] = np.ones(kernel_size)
    rot_mat = cv2.getRotationMatrix2D((kernel_size / 2, kernel_size / 2), angle, 1)
    k = cv2.warpAffine(k, rot_mat, (kernel_size, kernel_size))
    k = k / k.sum()

    edited = cv2.filter2D(image, -1, k)
    return _blend(image, edited, mask)


def vignette(image, mask=None, strength=2.5):
    h, w = image.shape[:2]
    kernel_x = cv2.getGaussianKernel(w, w / strength)
    kernel_y = cv2.getGaussianKernel(h, h / strength)
    kernel = kernel_y * kernel_x.T
    mask_v = kernel / kernel.max()
    edited = image.astype(np.float32) * mask_v[..., None]
    edited = np.clip(edited, 0, 255).astype(np.uint8)
    return edited  # whole-image effect by design, mask ignored


def warmth(image, mask, value=20):
    """Push toward orange (warm) or blue (cool) tones. Negative value = cooler."""
    edited = image.astype(np.int16)
    edited[:, :, 2] = np.clip(edited[:, :, 2] + value, 0, 255)   # R up = warmer
    edited[:, :, 0] = np.clip(edited[:, :, 0] - value, 0, 255)   # B down = warmer
    edited = edited.astype(np.uint8)
    return _blend(image, edited, mask)


# Registry: action name (from the command parser) -> function
ACTIONS = {
    "brightness": brightness,
    "saturation": saturation,
    "contrast": contrast,
    "hue_shift": hue_shift,
    "blur": blur_region,
    "sharpen": sharpen_region,
    "focus": focus_effect,
    "motion_blur": motion_blur,
    "running_effect": motion_blur,  # alias
    "vignette": vignette,
    "warmth": warmth,
}