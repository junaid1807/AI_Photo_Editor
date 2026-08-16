"""
Phase 3 - Streamlit UI

Run:
    uv run streamlit run app.py

Flow:
    1. Upload an image
    2. Either record a voice command (browser mic) or type one
    3. See the detected command, the region mask, and the final edit
"""

import os
import tempfile

import cv2
import numpy as np
import streamlit as st

from voice_to_text import transcribe
from command_parser import parse_command
from region_segmentation import get_mask, draw_mask_overlay
from effects import ACTIONS

WHOLE_IMAGE_ACTIONS = {"running_effect", "motion_blur", "vignette"}

st.set_page_config(page_title="Voice Photo Editor", layout="wide")
st.title("🎙️ Voice-Controlled Photo Editor")
st.caption('Try things like: "reduce the grass color", "focus the lion", "add a running effect"')


def bgr_from_uploaded(uploaded_file) -> np.ndarray:
    """Read an uploaded image file into an OpenCV BGR array."""
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


def get_command_text(audio_value) -> str | None:
    """Transcribe recorded mic audio (st.audio_input) to text."""
    if audio_value is None:
        return None
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_value.getvalue())
        tmp_path = tmp.name
    try:
        return transcribe(tmp_path)
    finally:
        os.remove(tmp_path)


def apply_edit(image: np.ndarray, intent: dict):
    """Runs region detection + effect for a parsed intent. Returns (edited_image, mask_overlay_or_None)."""
    action_name = intent.get("action")
    target = intent.get("target")
    value = intent.get("value")

    if action_name not in ACTIONS:
        raise ValueError(f"Unknown action '{action_name}'. Valid: {list(ACTIONS.keys())}")

    mask = None
    overlay = None
    if target and action_name not in WHOLE_IMAGE_ACTIONS:
        mask = get_mask(image, target)
        if mask is None or mask.sum() == 0:
            mask = None  # region not found -> apply to whole image instead
        else:
            overlay = draw_mask_overlay(image, mask)

    effect_fn = ACTIONS[action_name]
    kwargs = {}
    if value is not None:
        param_name = {
            "motion_blur": "kernel_size",
            "running_effect": "kernel_size",
            "blur": "ksize",
            "contrast": "value",
        }.get(action_name, "value")
        kwargs[param_name] = value

    edited = effect_fn(image, mask, **kwargs) if mask is not None or action_name in WHOLE_IMAGE_ACTIONS else effect_fn(image, mask)
    return edited, overlay


# ---------- UI ----------

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

col_mic, col_text = st.columns(2)
with col_mic:
    st.write("**Record a command**")
    audio_value = st.audio_input("Speak your edit command")
with col_text:
    st.write("**...or type one**")
    typed_command = st.text_input("Command", placeholder='e.g. "focus the lion"')

run_clicked = st.button("Apply Edit", type="primary", disabled=uploaded_image is None)

if run_clicked:
    image = bgr_from_uploaded(uploaded_image)

    with st.spinner("Transcribing / reading command..."):
        command_text = typed_command.strip() if typed_command.strip() else get_command_text(audio_value)

    if not command_text:
        st.error("No command given — record audio or type one before applying an edit.")
    else:
        st.info(f"🗣️ Detected command: **{command_text}**")

        with st.spinner("Understanding the command..."):
            intent = parse_command(command_text)
        st.caption(f"Parsed as: `{intent}`")

        with st.spinner("Finding region and applying edit..."):
            try:
                edited_bgr, overlay_bgr = apply_edit(image, intent)
            except Exception as e:
                st.error(f"Edit failed: {e}")
                edited_bgr, overlay_bgr = None, None

        if edited_bgr is not None:
            cols = st.columns(3 if overlay_bgr is not None else 2)
            cols[0].image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)
            if overlay_bgr is not None:
                cols[1].image(cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB), caption="Detected region", use_container_width=True)
            cols[-1].image(cv2.cvtColor(edited_bgr, cv2.COLOR_BGR2RGB), caption="Edited", use_container_width=True)

            _, buf = cv2.imencode(".jpg", edited_bgr)
            st.download_button("Download edited image", data=buf.tobytes(), file_name="edited.jpg", mime="image/jpeg")
elif uploaded_image is not None:
    st.image(cv2.cvtColor(bgr_from_uploaded(uploaded_image), cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)