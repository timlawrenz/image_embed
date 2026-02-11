from PIL import Image
from typing import Optional, List, Tuple

import re
import torch

from app.core import model_loader
from app.services.image_utils import crop_image_and_get_base64

DEFAULT_MAX_LENGTH = 300

_SYSTEM_PROMPT = (
    "You are generating a caption for a text-to-image training dataset. "
    "Write exactly one dense paragraph in a dry, descriptive tone (no flowery language, no lists). "
    "Describe only what is visible in the image; do not guess or invent details. "
    "If a detail is unclear, say it is unclear/indistinct. "
    "Include (when visible): subject, pose, clothing/accessories, lighting, background, composition/framing, and camera angle."
)

_USER_PROMPT = (
    "Analyze this image for a text-to-image training dataset. "
    "Describe the subject, pose, clothing, lighting, background, and camera angle in extreme detail."
)


def get_image_description(
    pil_image_rgb: Image.Image,
    crop_box: Optional[List[int]] = None,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> Tuple[str, Optional[str], Optional[List[int]]]:
    """Generate a dense, single-paragraph description using Gemma 3 multimodal."""

    device = model_loader.get_device()
    if device != "cuda":
        raise RuntimeError("describe_image requires CUDA")

    image_to_process = pil_image_rgb
    b64_image = None
    if crop_box:
        image_to_process, b64_image = crop_image_and_get_base64(pil_image_rgb, crop_box)

    llm, processor = model_loader.get_gemma_vision_model_and_processor()

    messages = [
        {"role": "system", "content": [{"type": "text", "text": _SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_to_process},
                {"type": "text", "text": _USER_PROMPT},
            ],
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    if hasattr(inputs, "to"):
        inputs = inputs.to(device)
    else:
        inputs = {k: v.to(device) for k, v in inputs.items()}

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = llm.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=False,
        )

    generated_tokens = generation[0][input_len:]
    text = processor.decode(generated_tokens, skip_special_tokens=True)
    text = re.sub(r"\s+", " ", text).strip()

    return text, b64_image, crop_box
