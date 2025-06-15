import logging
from typing import List, Optional
from PIL import Image
import torchvision.transforms as T
from app.core import model_loader

logger = logging.getLogger(__name__)

def get_prominent_person_bbox(pil_image_rgb: Image.Image) -> Optional[List[int]]:
    logger.info("Attempting to detect prominent person.")
    person_detection_model_instance = model_loader.get_person_detection_model()
    current_device = model_loader.get_device()

    transform_detection = T.Compose([T.ToTensor()])
    img_tensor_detection = transform_detection(pil_image_rgb).to(current_device)
    with model_loader.torch.no_grad(): # Use torch from model_loader to ensure consistency if it were to be aliased/managed there
        predictions = person_detection_model_instance([img_tensor_detection])
    scores = predictions[0]['scores']
    labels = predictions[0]['labels']
    boxes = predictions[0]['boxes']
    person_indices = (labels == 1).nonzero(as_tuple=True)[0]
    if len(person_indices) > 0:
        person_scores = scores[person_indices]
        highest_score_person_local_idx = person_scores.argmax()
        highest_score_person_global_idx = person_indices[highest_score_person_local_idx]
        bbox = boxes[highest_score_person_global_idx].cpu().numpy().astype(int)
        xmin, ymin, xmax, ymax = bbox
        if xmin >= xmax or ymin >= ymax or pil_image_rgb.width == 0 or pil_image_rgb.height == 0:
            logger.error(f"Invalid bounding box [{xmin},{ymin},{xmax},{ymax}] or image dimensions for person detection.")
            return None
        if xmin < 0: xmin = 0
        if ymin < 0: ymin = 0
        if xmax > pil_image_rgb.width: xmax = pil_image_rgb.width
        if ymax > pil_image_rgb.height: ymax = pil_image_rgb.height
        if xmin >= xmax or ymin >= ymax:
             logger.error(f"Clamped bounding box [{xmin},{ymin},{xmax},{ymax}] is still invalid.")
             return None
        confidence = scores[highest_score_person_global_idx].item()
        logger.info(f"Prominent person detected with bbox: [{xmin}, {ymin}, {xmax}, {ymax}], confidence: {confidence:.2f}")
        # Convert numpy types to standard python int for JSON serialization
        return [int(xmin), int(ymin), int(xmax), int(ymax)]
    else:
        logger.info("No person detected.")
        return None

def get_prominent_face_bbox_in_region(pil_image_rgb: Image.Image, person_bbox: Optional[List[int]]) -> Optional[List[int]]:
    face_detection_model_instance = model_loader.get_face_detection_model()
    if not face_detection_model_instance:
        logger.warning("Face detection model not available (via loader). Skipping face detection.")
        return None
    target_image_for_face_detection = pil_image_rgb
    offset_x, offset_y = 0, 0
    if person_bbox:
        logger.info(f"Performing face detection within person bbox: {person_bbox}")
        xmin_p, ymin_p, xmax_p, ymax_p = person_bbox
        if xmin_p >= xmax_p or ymin_p >= ymax_p:
            logger.error(f"Invalid person_bbox for cropping: {person_bbox}")
            return None
        target_image_for_face_detection = pil_image_rgb.crop(person_bbox)
        offset_x, offset_y = xmin_p, ymin_p
        if target_image_for_face_detection.width == 0 or target_image_for_face_detection.height == 0:
            logger.error(f"Cropped person region for face detection is empty using bbox: {person_bbox}")
            return None
    else:
        logger.info("Performing face detection on whole image (no person_bbox provided).")
    # Mock detection for demonstration
    if person_bbox: # This mock logic implies face detection is only attempted if a person_bbox is available
        pw = target_image_for_face_detection.width
        ph = target_image_for_face_detection.height
        if pw > 10 and ph > 10 :
            face_bbox_relative = [int(pw*0.1), int(ph*0.1), int(pw*0.4), int(ph*0.4)] # Example relative coordinates
            logger.info(f"Mock face detected in region with relative bbox: {face_bbox_relative}")
        else:
            logger.info(f"Person region too small for mock face detection: w={pw}, h={ph}")
            return None
    else: # No person_bbox means we are checking the whole image
        logger.info("Mock face detection: No person_bbox, so no face detected on whole image for this mock.")
        return None

    xmin_f, ymin_f, xmax_f, ymax_f = face_bbox_relative
    # Convert to standard python int for JSON serialization
    final_face_bbox = [int(xmin_f + offset_x), int(ymin_f + offset_y), int(xmax_f + offset_x), int(ymax_f + offset_y)]
    logger.info(f"Prominent face detected with final bbox: {final_face_bbox}")
    return final_face_bbox
