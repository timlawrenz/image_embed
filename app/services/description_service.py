from PIL import Image
from typing import Optional, List, Tuple
from app.core import model_loader
from app.services.image_utils import crop_image_and_get_base64

# A specific model name can be chosen, but we'll use a default for now.
CAPTION_MODEL_NAME = "Salesforce/blip-image-captioning-large"

def get_image_description(
    pil_image_rgb: Image.Image,
    crop_box: Optional[List[int]] = None
) -> Tuple[str, Optional[str], Optional[List[int]]]:
    """
    Generates a text description for an image or a cropped region of it.

    Args:
        pil_image_rgb: The PIL image in RGB format.
        crop_box: Optional bounding box [xmin, ymin, xmax, ymax] to crop from the image.

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
        # This utility function crops the image, and also returns the base64 representation
        image_to_process, b64_image = crop_image_and_get_base64(pil_image_rgb, crop_box)

    # Process image for the captioning model
    # Note: The "large" model may benefit from a different image size, but the processor handles resizing.
    inputs = processor(images=image_to_process, return_tensors="pt").to(device)
    
    # Generate caption using the model
    outputs = model.generate(**inputs, max_length=50) # Set max_length for concise captions
    
    # Decode the generated tokens to a string
    caption = processor.decode(outputs[0], skip_special_tokens=True)

    # The first word is often "a" or "an", which we can capitalize for better readability.
    if caption:
        caption = caption[0].upper() + caption[1:]

    return caption, b64_image, crop_box
