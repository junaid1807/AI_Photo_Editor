"""
General-Purpose Voice Photo Editor - End to End Pipeline

Handles arbitrary commands like:
    "reduce the grass color"
    "focus the sun"
    "add a running effect"
    "blur the background"
    "make it warmer"

Run:
    uv run main.py --image images/sample.jpg --text "reduce the grass color"
    uv run main.py --image images/sample.jpg              # uses mic
"""

import argparse
import os
import cv2

from dotenv import load_dotenv
load_dotenv()

from voice_to_text import record_audio, transcribe
from command_parser import parse_command
from region_segmentation import get_mask, draw_mask_overlay
from effects import ACTIONS

# Actions that apply to the whole image and don't need a region mask
WHOLE_IMAGE_ACTIONS = {"running_effect", "motion_blur", "vignette"}


def run_pipeline(image_path: str, command_text: str = None, record_seconds: int = 4):
    os.makedirs("outputs", exist_ok=True)  # cv2.imwrite fails silently if this is missing

    # 1. Get the command
    if command_text is None:
        audio_path = record_audio(duration=record_seconds)
        command_text = transcribe(audio_path)
    print(f"🗣️  Command: {command_text}")

    # 2. Parse into structured intent (target / action / value)
    intent = parse_command(command_text)
    print(f"📦 Parsed intent: {intent}")

    action_name = intent.get("action")
    target = intent.get("target")
    value = intent.get("value")

    if action_name not in ACTIONS:
        raise ValueError(f"Unknown action '{action_name}'. Valid: {list(ACTIONS.keys())}")

    # 3. Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # 4. Get region mask (skip for whole-image effects or no target)
    mask = None
    if target and action_name not in WHOLE_IMAGE_ACTIONS:
        mask = get_mask(image, target)
        if mask is None or mask.sum() == 0:
            print(f"⚠️  Could not find region '{target}', applying effect to whole image instead")
            mask = None
        else:
            overlay = draw_mask_overlay(image, mask)
            ok = cv2.imwrite("outputs/mask_overlay.jpg", overlay)
            print(f"📍 Found region for '{target}'" + (", see outputs/mask_overlay.jpg" if ok else " (⚠️ failed to save overlay)"))

    # 5. Apply the effect
    effect_fn = ACTIONS[action_name]
    kwargs = {}
    if value is not None:
        # different effects name their param differently; try common ones
        param_name = {
            "motion_blur": "kernel_size",
            "running_effect": "kernel_size",
            "blur": "ksize",
            "contrast": "value",
        }.get(action_name, "value")
        kwargs[param_name] = value

    edited = effect_fn(image, mask, **kwargs) if mask is not None or action_name in WHOLE_IMAGE_ACTIONS else effect_fn(image, mask)

    # 6. Save
    ok = cv2.imwrite("outputs/edited.jpg", edited)
    if ok:
        print(f"✅ Done. See {os.path.abspath('outputs/edited.jpg')}")
    else:
        print("❌ Failed to save outputs/edited.jpg — check the 'outputs' folder exists and is writable")
    return edited


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--text", default=None, help="Skip mic and pass command text directly")
    parser.add_argument("--seconds", type=int, default=4, help="Mic recording duration")
    args = parser.parse_args()

    run_pipeline(args.image, command_text=args.text, record_seconds=args.seconds)