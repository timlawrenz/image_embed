import logging
from typing import List, Optional, Tuple
from PIL import Image
from app.core import model_loader

logger = logging.getLogger(__name__)

def get_clip_embedding(
    pil_image_rgb: Image.Image, 
    clip_model_name: str,
    crop_bbox: Optional[List[int]] = None,
    shared_context: Optional[dict] = None,
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
        from app.services.image_utils import get_cropped_image
        image_to_embed, base64_cropped_image = get_cropped_image(
            pil_image_rgb, crop_bbox, shared_context
        )
        if image_to_embed.width == 0 or image_to_embed.height == 0:
            logger.error(f"Cropped image for embedding is empty using bbox: {crop_bbox}")
            raise ValueError("Cropped image for embedding is empty.")
        
        actual_crop_bbox = crop_bbox 
        logger.info(f"Cropped image for embedding (size: {image_to_embed.size}) encoded to base64.")

    logger.info(f"Generating CLIP embedding for image (size: {image_to_embed.size}) using {clip_model_name}.")
    image_input = clip_preprocess_instance(image_to_embed).unsqueeze(0).to(current_device)
    with model_loader.torch.no_grad():
        embedding_tensor = clip_model_instance.encode_image(image_input)
    embedding_list = embedding_tensor[0].cpu().numpy().tolist()
    logger.info("CLIP embedding generated successfully.")
    return embedding_list, base64_cropped_image, actual_crop_bbox


def get_dino_embedding(
    pil_image_rgb: Image.Image,
    dino_model_name: str = "vit_base_patch14_dinov2.lvd142m",
    crop_bbox: Optional[List[int]] = None,
    shared_context: Optional[dict] = None,
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
        from app.services.image_utils import get_cropped_image
        image_to_embed, base64_cropped_image = get_cropped_image(
            pil_image_rgb, crop_bbox, shared_context
        )
        if image_to_embed.width == 0 or image_to_embed.height == 0:
            logger.error(f"Cropped image for embedding is empty using bbox: {crop_bbox}")
            raise ValueError("Cropped image for embedding is empty.")

        actual_crop_bbox = crop_bbox
        logger.info(f"Cropped image for embedding (size: {image_to_embed.size}) encoded to base64.")

    logger.info(f"Generating DINO embedding for image (size: {image_to_embed.size}) using {dino_model_name}.")
    image_input = dino_processor_instance(image_to_embed).unsqueeze(0).to(current_device)
    with model_loader.torch.no_grad():
        embedding_tensor = dino_model_instance(image_input)
    embedding_list = embedding_tensor[0].cpu().numpy().tolist()
    logger.info("DINO embedding generated successfully.")
    return embedding_list, base64_cropped_image, actual_crop_bbox


def get_dino_v3_embedding(
    pil_image_rgb: Image.Image,
    dino_v3_model_name: str = "facebook/dinov3-vitl16-pretrain-lvd1689m",
    crop_bbox: Optional[List[int]] = None,
    shared_context: Optional[dict] = None,
) -> Tuple[List[float], Optional[str], Optional[List[int]]]:

    dino_model_instance, dino_processor_instance = model_loader.get_dino_v3_model_and_processor(dino_v3_model_name)
    current_device = model_loader.get_device()

    image_to_embed = pil_image_rgb
    base64_cropped_image = None
    actual_crop_bbox = None

    if crop_bbox:
        logger.info(f"Cropping image for DINOv3 embedding with bbox: {crop_bbox}")
        xmin, ymin, xmax, ymax = crop_bbox
        if xmin >= xmax or ymin >= ymax:
            logger.error(f"Invalid crop_bbox for embedding: {crop_bbox}")
            raise ValueError("Invalid bounding box for cropping.")
        from app.services.image_utils import get_cropped_image
        image_to_embed, base64_cropped_image = get_cropped_image(
            pil_image_rgb, crop_bbox, shared_context
        )
        if image_to_embed.width == 0 or image_to_embed.height == 0:
            logger.error(f"Cropped image for embedding is empty using bbox: {crop_bbox}")
            raise ValueError("Cropped image for embedding is empty.")

        actual_crop_bbox = crop_bbox
        logger.info(f"Cropped image for embedding (size: {image_to_embed.size}) encoded to base64.")

    logger.info(
        f"Generating DINOv3 embedding for image (size: {image_to_embed.size}) using {dino_v3_model_name}."
    )

    inputs = dino_processor_instance(images=image_to_embed, return_tensors="pt")
    inputs = {k: (v.to(current_device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    with model_loader.torch.no_grad():
        outputs = dino_model_instance(**inputs)

    embedding_tensor = getattr(outputs, "image_embeds", None)
    if embedding_tensor is None:
        embedding_tensor = getattr(outputs, "pooler_output", None)
    if embedding_tensor is None and hasattr(outputs, "last_hidden_state"):
        embedding_tensor = outputs.last_hidden_state[:, 0]

    if embedding_tensor is None:
        raise RuntimeError("Unexpected DINOv3 output structure: missing embedding tensor")

    embedding_list = embedding_tensor[0].detach().cpu().numpy().tolist()
    logger.info("DINOv3 embedding generated successfully.")
    return embedding_list, base64_cropped_image, actual_crop_bbox


def get_dino_v3_patch_embedding(
    pil_image_rgb: Image.Image,
    crop_bbox: Optional[List[int]] = None,
    shared_context: Optional[dict] = None,
):
    """
    Generate a mean-pooled patch token embedding from DINOv3 ViT-L.
    Returns 1024-dim vector (mean of all patch tokens, excluding CLS).
    """
    dino_v3_model_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    dino_model_instance, dino_processor_instance = model_loader.get_dino_v3_model_and_processor(dino_v3_model_name)
    current_device = model_loader.get_device()

    image_to_embed = pil_image_rgb
    actual_crop_bbox = None
    base64_cropped_image = None

    if crop_bbox:
        logger.info(f"Cropping image for DINOv3 patch embedding with bbox: {crop_bbox}")
        xmin, ymin, xmax, ymax = crop_bbox
        if xmin >= xmax or ymin >= ymax:
            raise ValueError("Invalid bounding box for cropping.")
        from app.services.image_utils import get_cropped_image
        image_to_embed, base64_cropped_image = get_cropped_image(
            pil_image_rgb, crop_bbox, shared_context
        )
        if image_to_embed.width == 0 or image_to_embed.height == 0:
            raise ValueError("Cropped image for embedding is empty.")

        actual_crop_bbox = crop_bbox

    logger.info(
        f"Generating DINOv3 patch embedding for image (size: {image_to_embed.size}) using {dino_v3_model_name}."
    )

    inputs = dino_processor_instance(images=image_to_embed, return_tensors="pt")
    inputs = {k: (v.to(current_device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    with model_loader.torch.no_grad():
        outputs = dino_model_instance(**inputs)

    if not hasattr(outputs, "last_hidden_state"):
        raise RuntimeError("DINOv3 output missing last_hidden_state for patch tokens")

    # Patch tokens are all tokens after the CLS token (index 0)
    patch_tokens = outputs.last_hidden_state[:, 1:]  # [1, num_patches, 1024]
    mean_pooled = patch_tokens.mean(dim=1)  # [1, 1024]

    embedding_list = mean_pooled[0].detach().cpu().numpy().tolist()
    logger.info(f"DINOv3 patch embedding generated successfully (dim={len(embedding_list)}).")
    return embedding_list, base64_cropped_image, actual_crop_bbox
