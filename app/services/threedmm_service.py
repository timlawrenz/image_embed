import io
import logging
import base64
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image
from app.core import model_loader

logger = logging.getLogger(__name__)

def fit_3dmm_on_face(pil_image_rgb: Image.Image, face_bbox: List[int]) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[List[int]]]:
    threedmm_model_instance = model_loader.get_threedmm_model()
    if not threedmm_model_instance:
        logger.warning("3DMM model not available (via loader). Skipping 3DMM fitting.")
        return None, None, None 
    if not face_bbox:
        logger.warning("No face bounding box provided for 3DMM fitting.")
        return None, None, None
        
    logger.info(f"Performing 3DMM fitting on face with bbox: {face_bbox}")
    xmin, ymin, xmax, ymax = face_bbox
    if xmin >= xmax or ymin >= ymax:
        logger.error(f"Invalid face_bbox for 3DMM: {face_bbox}")
        return None, None, face_bbox 
    
    cropped_face = pil_image_rgb.crop(face_bbox)
    if cropped_face.width == 0 or cropped_face.height == 0:
        logger.error(f"Cropped face for 3DMM is empty using bbox: {face_bbox}")
        return None, None, face_bbox

    buffered = io.BytesIO()
    cropped_face.save(buffered, format="PNG")
    base64_cropped_face = base64.b64encode(buffered.getvalue()).decode("utf-8")
    logger.info(f"Cropped face for 3DMM (size: {cropped_face.size}) encoded to base64.")

    mock_3dmm_params = {"shape_coeffs": [0.1]*50, "expression_coeffs": [0.2]*20, "pose": [0,0,0,1,0,0]}
    logger.info(f"Mock 3DMM fitting completed for face in bbox: {face_bbox}")
    return mock_3dmm_params, base64_cropped_face, face_bbox
