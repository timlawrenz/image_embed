from PIL import Image
from typing import Optional, List, Tuple
from app.core import model_loader
from app.services.image_utils import crop_image_and_get_base64
import re

# The model name for Salesforce's BLIP model.
CAPTION_MODEL_NAME = "Salesforce/blip-image-captioning-large"

def get_image_description(
    pil_image_rgb: Image.Image,
    crop_box: Optional[List[int]] = None,
    max_length: int = 50,
) -> Tuple[str, Optional[str], Optional[List[int]]]:
    """
    Generates a text description for an image or a cropped region of it
    using the Salesforce BLIP model.

    Args:
        pil_image_rgb: The PIL image in RGB format.
        crop_box: Optional bounding box [xmin, ymin, xmax, ymax] to crop from the image.
        max_length: The maximum length of the generated caption.

    Returns:
        A tuple containing:
        - The generated text description (string).
        - A base64 encoded string of the cropped image if crop_box was provided, otherwise None.
        - The bounding box used for cropping if crop_box was provided, otherwise None.
    """
    model, processor = model_loader.get_image_captioning_model_and_processor(CAPTION_MODEL_NAME)
    device = model_loader.get_device()

    image_to_process = pil_image_rgb
    b64_image = None
    
    if crop_box:
        image_to_process, b64_image = crop_image_and_get_base64(pil_image_rgb, crop_box)

    # Process the image.
    inputs = processor(images=image_to_process, return_tensors="pt").to(device)
    
    # Generate the base caption.
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_beams=4,
        early_stopping=True,
    )

    base_caption = processor.decode(outputs[0], skip_special_tokens=True)

    try:
        llm, tokenizer = model_loader.get_gemma_text_model_and_tokenizer()
        prompt = (
            "You are an expert image describer. Write a detailed but concise description "
            "of the image. Use the following base caption as grounding and do not invent "
            "objects not implied by it.\n\n"
            f"Base caption: {base_caption}\n\nDescription:"
        )

        llm_inputs = tokenizer(prompt, return_tensors="pt")
        if hasattr(llm_inputs, "to"):
            llm_inputs = llm_inputs.to(device)
        else:
            llm_inputs = {k: v.to(device) for k, v in llm_inputs.items()}

        generated = llm.generate(
            **llm_inputs,
            max_new_tokens=max(32, max_length),
            do_sample=False,
        )
        final_text = tokenizer.decode(generated[0], skip_special_tokens=True)

        # Best-effort: strip the prompt back out if it was echoed.
        if "Description:" in final_text:
            final_text = final_text.split("Description:", 1)[-1].strip()

        return final_text, b64_image, crop_box
    except Exception as e:
        # Non-fatal fallback to BLIP caption
        model_loader.logger.warning("DescriptionService: Gemma unavailable, falling back to base caption: %s", e)
        return base_caption, b64_image, crop_box
