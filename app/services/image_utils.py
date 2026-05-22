import io
import logging
import requests
from PIL import Image
from fastapi import HTTPException
import base64
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)

def download_image(image_url_str: str) -> Image.Image:
    logger.info(f"Downloading image from {image_url_str}")
    try:
        headers = {'User-Agent': 'ImageAnalysisAPI/0.2.1'}
        response = requests.get(image_url_str, stream=True, timeout=10, headers=headers)
        response.raise_for_status()
        image_bytes = response.content
        pil_image = Image.open(io.BytesIO(image_bytes))
        if pil_image.mode != 'RGB':
            logger.info(f"Converting image from mode {pil_image.mode} to RGB.")
            pil_image = pil_image.convert('RGB')
        logger.info(f"Successfully downloaded and opened image from {image_url_str}")
        return pil_image
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download image from {image_url_str}: {e}")
        raise HTTPException(status_code=400, detail=f"Could not download image from URL: {e}")
    except Exception as e:
        logger.error(f"Failed to process image after download from {image_url_str}: {e}")
        raise HTTPException(status_code=400, detail=f"Could not process image: {e}")


def process_uploaded_image(image_bytes: bytes) -> Image.Image:
    """
    Processes image bytes from an upload into a PIL Image object.
    """
    logger.info("Processing uploaded image bytes.")
    try:
        pil_image = Image.open(io.BytesIO(image_bytes))
        if pil_image.mode != 'RGB':
            logger.info(f"Converting image from mode {pil_image.mode} to RGB.")
            pil_image = pil_image.convert('RGB')
        logger.info("Successfully processed uploaded image.")
        return pil_image
    except Exception as e:
        logger.error(f"Failed to process uploaded image bytes: {e}")
        raise HTTPException(status_code=400, detail=f"Could not process uploaded image: {e}")

def crop_image_and_get_base64(
    pil_image: Image.Image, 
    bbox: List[int]
) -> Tuple[Image.Image, str]:
    """
    Crops a PIL image using a bounding box and returns the cropped image
    along with its base64 encoded PNG string representation.
    """
    cropped_image = pil_image.crop(bbox)
    
    buffer = io.BytesIO()
    cropped_image.save(buffer, format="PNG")
    b64_string = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    return cropped_image, b64_string


def get_cropped_image(
    pil_image: Image.Image,
    bbox: List[int],
    shared_context: Optional[dict] = None,
) -> Tuple[Image.Image, str]:
    """
    Returns (cropped_image, base64_str) for the given bbox.

    If shared_context is provided, caches the result under the key
    f"crop_{tuple(bbox)}" so that subsequent calls with the same bbox
    on the same request reuse the crop without re-executing it.
    """
    if shared_context is not None:
        cache_key = f"crop_{tuple(bbox)}"
        cached = shared_context.get(cache_key)
        if cached is not None:
            logger.info("Reusing cached crop for bbox %s", bbox)
            return cached
        cropped, b64 = crop_image_and_get_base64(pil_image, bbox)
        shared_context[cache_key] = (cropped, b64)
        return cropped, b64
    return crop_image_and_get_base64(pil_image, bbox)
