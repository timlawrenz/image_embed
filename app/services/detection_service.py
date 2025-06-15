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

    # Use the MTCNN model to detect faces
    try:
        boxes, probs = face_detection_model_instance.detect(target_image_for_face_detection)
    except Exception as e:
        logger.exception(f"Face detection model failed during inference: {e}")
        return None

    # MTCNN returns None if no faces are found
    if boxes is None or len(boxes) == 0:
        logger.info("No faces detected in the target region.")
        return None

    # Find the face with the highest probability
    best_prob_idx = probs.argmax()
    best_box = boxes[best_prob_idx]
    best_prob = probs[best_prob_idx]

    xmin_f, ymin_f, xmax_f, ymax_f = best_box
    
    # Add offset to convert to absolute coordinates and ensure values are JSON-serializable Python ints
    final_face_bbox = [int(xmin_f + offset_x), int(ymin_f + offset_y), int(xmax_f + offset_x), int(ymax_f + offset_y)]
    logger.info(f"Prominent face detected with final bbox: {final_face_bbox}, confidence: {best_prob:.2f}")

    return final_face_bbox
