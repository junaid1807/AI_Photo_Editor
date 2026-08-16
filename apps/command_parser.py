"""
Command Understanding
Turns free-form speech ("reduce the grass color", "focus the sun",
"add a running effect") into structured JSON:
    {"target": "grass", "action": "saturation", "value": -40}
    {"target": "the sun", "action": "focus", "value": null}
    {"target": null, "action": "running_effect", "value": null}

Uses Groq's API (free tier, fast inference) so it generalizes to phrasing
you didn't hard-code.
Requires: export GROQ_API_KEY=gsk_...  (get one free at console.groq.com)
Falls back to a small keyword parser if no API key is set, so the app still
runs (with much more limited understanding) without one.
"""

import os
import json
import re

from dotenv import load_dotenv
load_dotenv()  # reads GROQ_API_KEY (and anything else) from a .env file in the cwd

from dotenv import load_dotenv
load_dotenv()  # reads .env file in the current directory and loads it into os.environ

VALID_ACTIONS = [
    "brightness", "saturation", "contrast", "hue_shift",
    "blur", "sharpen", "focus", "motion_blur", "running_effect",
    "vignette", "warmth",
]

SYSTEM_PROMPT = f"""You convert a spoken photo-editing command into JSON.

Valid actions: {VALID_ACTIONS}

Rules:
- "target" is the short noun phrase for what to edit (e.g. "grass", "the sun", "sky",
  "face"). If the command applies to the whole photo (e.g. "add a running effect",
  "add a vignette"), set target to null.
- "action" must be one of the valid actions above. Map meaning, not just keywords:
  "reduce the grass color" -> saturation (negative value)
  "focus the sun" -> focus
  "add a running effect" / "make it look fast" -> running_effect
  "blur the background" -> blur (target: "background")
  "make it warmer/cooler" -> warmth
- "value" is a number appropriate to the action (brightness/saturation/warmth: -100..100,
  contrast: 0.5..2.0 multiplier, motion_blur/running_effect kernel_size: 10..40,
  blur ksize: 5..50). Use a sensible default if the user didn't specify a magnitude.
  Use null if the action has no tunable value (e.g. focus).

Respond with ONLY the JSON object, no other text, no markdown fences.
"""


def parse_command_llm(text: str) -> dict:
    from groq import Groq

    client = Groq()  # reads GROQ_API_KEY from env
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # free-tier, strong instruction following
        max_tokens=200,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```json|```$", "", raw, flags=re.MULTILINE).strip()
    return json.loads(raw)


# ---------- Fallback: no API key available ----------

_ACTION_KEYWORDS = {
    "saturation": ["color", "colour", "saturation", "vivid", "reduce", "faded", "desaturate"],
    "focus": ["focus"],
    "running_effect": ["running", "speed", "motion", "fast", "action"],
    "blur": ["blur"],
    "sharpen": ["sharpen", "sharp", "crisp"],
    "brightness": ["bright", "brighten", "lighten"],
    "vignette": ["vignette", "darken edges", "darken corners"],
    "warmth": ["warm", "cool", "cold", "tone"],
    "hue_shift": ["hue", "color shift"],
    "contrast": ["contrast"],
}


def parse_command_fallback(text: str) -> dict:
    text_l = text.lower()

    action = None
    for name, keywords in _ACTION_KEYWORDS.items():
        if any(k in text_l for k in keywords):
            action = name
            break
    action = action or "brightness"

    # crude target extraction: look for "of X" / "the X" patterns, else None
    match = re.search(r"(?:of|the)\s+([a-z\s]+?)(?:\.|$|,)", text_l)
    target = match.group(1).strip() if match else None
    if target:
        # strip trailing generic words that aren't part of the region name
        for stopword in ["color", "colour", "effect", "region", "area"]:
            target = re.sub(rf"\s*{stopword}$", "", target).strip()

    value = -40 if "reduce" in text_l or "less" in text_l else 40
    return {"target": target, "action": action, "value": value}


def parse_command(text: str) -> dict:
    """Main entry point. Tries the LLM parser, falls back to keywords on any failure."""
    if os.environ.get("GROQ_API_KEY"):
        try:
            return parse_command_llm(text)
        except Exception as e:
            print(f"⚠️  LLM parser failed ({e}), using keyword fallback")
    else:
        print("⚠️  No GROQ_API_KEY set, using keyword fallback (limited understanding)")

    return parse_command_fallback(text)


if __name__ == "__main__":
    tests = [
        "reduce the grass color",
        "focus the sun",
        "add a running effect",
        "blur the background",
        "make it warmer",
    ]
    for t in tests:
        print(t, "->", parse_command(t))