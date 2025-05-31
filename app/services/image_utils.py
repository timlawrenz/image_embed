import io
import logging
import requests
from PIL import Image
from fastapi import HTTPException

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
