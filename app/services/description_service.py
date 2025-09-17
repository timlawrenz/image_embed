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
    max_length: int = 50  # Adjusted for typical BLIP caption length
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
    
    # Generate the caption.
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_beams=4,
        early_stopping=True
    )
    
    # Decode the generated tokens.
    caption = processor.decode(outputs[0], skip_special_tokens=True)

    return caption, b64_image, crop_box
