import io
import logging
import base64
from typing import List, Optional, Tuple
from PIL import Image
from app.core import model_loader

logger = logging.getLogger(__name__)

def get_clip_embedding(
    pil_image_rgb: Image.Image, 
    clip_model_name: str, # Added to specify which CLIP model to use
    crop_bbox: Optional[List[int]] = None
) -> Tuple[List[float], Optional[str], Optional[List[int]]]:
    
    clip_model_instance, clip_preprocess_instance = model_loader.get_clip_model_and_preprocess(clip_model_name)
    current_device = model_loader.get_device()

    image_to_embed = pil_image_rgb
    base64_cropped_image = None
    actual_crop_bbox = None

    if crop_bbox:
        logger.info(f"Cropping image for CLIP embedding with bbox: {crop_bbox}")
        xmin, ymin, xmax, ymax = crop_bbox
        if xmin >= xmax or ymin >= ymax:
            logger.error(f"Invalid crop_bbox for embedding: {crop_bbox}")
            raise ValueError("Invalid bounding box for cropping.")
        image_to_embed = pil_image_rgb.crop(crop_bbox)
        if image_to_embed.width == 0 or image_to_embed.height == 0:
            logger.error(f"Cropped image for embedding is empty using bbox: {crop_bbox}")
            raise ValueError("Cropped image for embedding is empty.")
        
        actual_crop_bbox = crop_bbox 
        buffered = io.BytesIO()
        image_to_embed.save(buffered, format="PNG")
        base64_cropped_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
        logger.info(f"Cropped image for embedding (size: {image_to_embed.size}) encoded to base64.")

    logger.info(f"Generating CLIP embedding for image (size: {image_to_embed.size}) using {clip_model_name}.")
    image_input = clip_preprocess_instance(image_to_embed).unsqueeze(0).to(current_device)
    with model_loader.torch.no_grad(): # Use torch from model_loader
        embedding_tensor = clip_model_instance.encode_image(image_input)
    embedding_list = embedding_tensor[0].cpu().numpy().tolist()
    logger.info("CLIP embedding generated successfully.")
    return embedding_list, base64_cropped_image, actual_crop_bbox


def get_dino_embedding(
    pil_image_rgb: Image.Image,
    dino_model_name: str = "vit_base_patch14_dinov2.lvd142m",
    crop_bbox: Optional[List[int]] = None
) -> Tuple[List[float], Optional[str], Optional[List[int]]]:
    
    dino_model_instance, dino_processor_instance = model_loader.get_dino_model_and_processor(dino_model_name)
    current_device = model_loader.get_device()

    image_to_embed = pil_image_rgb
    base64_cropped_image = None
    actual_crop_bbox = None

    if crop_bbox:
        logger.info(f"Cropping image for DINO embedding with bbox: {crop_bbox}")
        xmin, ymin, xmax, ymax = crop_bbox
        if xmin >= xmax or ymin >= ymax:
            logger.error(f"Invalid crop_bbox for embedding: {crop_bbox}")
            raise ValueError("Invalid bounding box for cropping.")
        image_to_embed = pil_image_rgb.crop(crop_bbox)
        if image_to_embed.width == 0 or image_to_embed.height == 0:
            logger.error(f"Cropped image for embedding is empty using bbox: {crop_bbox}")
            raise ValueError("Cropped image for embedding is empty.")
        
        actual_crop_bbox = crop_bbox
        buffered = io.BytesIO()
        image_to_embed.save(buffered, format="PNG")
        base64_cropped_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
        logger.info(f"Cropped image for embedding (size: {image_to_embed.size}) encoded to base64.")

    logger.info(f"Generating DINO embedding for image (size: {image_to_embed.size}) using {dino_model_name}.")
    image_input = dino_processor_instance(image_to_embed).unsqueeze(0).to(current_device)
    with model_loader.torch.no_grad():
        embedding_tensor = dino_model_instance(image_input)
    embedding_list = embedding_tensor[0].cpu().numpy().tolist()
    logger.info("DINO embedding generated successfully.")
    return embedding_list, base64_cropped_image, actual_crop_bbox
