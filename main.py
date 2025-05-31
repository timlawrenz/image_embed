import io
import logging
from typing import Any, Dict, List, Optional, Tuple # Added Tuple

import clip
import requests
import torch
import torchvision # Added for detection model
import torchvision.transforms as T # Added for detection model
from fastapi import FastAPI, HTTPException
from PIL import Image
# Pydantic models moved to app.pydantic_models
import base64 # Added
from app.pydantic_models import AnalysisTask, ImageAnalysisRequest, OperationResult, ImageAnalysisResponse

from app.core import model_loader # Added

# --- Configuration & Initialization ---

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Advanced Image Analysis API",
    description="Performs various analyses on an image from a URL, including embeddings, object detection, and more.",
    version="0.2.1",
)

# --- Model Loading ---
# Models are now loaded on-demand by app.core.model_loader
# DEVICE = model_loader.get_device() # Device is managed by model_loader

# Default CLIP model name (can be made configurable or part of request later)
MODEL_NAME_CLIP = "ViT-B/32" 

# Placeholders for models that are not yet fully implemented in the loader
# but whose getter functions exist in model_loader.
# face_detection_model = model_loader.get_face_detection_model()
# threedmm_model = model_loader.get_threedmm_model()

# --- Pydantic Models have been moved to app/pydantic_models.py ---

# --- Helper Functions for Image Processing ---

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


def get_prominent_person_bbox(pil_image_rgb: Image.Image) -> Optional[List[int]]:
    logger.info("Attempting to detect prominent person.")
    person_detection_model_instance = model_loader.get_person_detection_model()
    current_device = model_loader.get_device()

    transform_detection = T.Compose([T.ToTensor()])
    img_tensor_detection = transform_detection(pil_image_rgb).to(current_device)
    with torch.no_grad():
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
        return [xmin, ymin, xmax, ymax]
    else:
        logger.info("No person detected.")
        return None

def get_prominent_face_bbox_in_region(pil_image_rgb: Image.Image, person_bbox: Optional[List[int]]) -> Optional[List[int]]:
    # (Implementation from previous version, including placeholder logic)
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
        # For this mock, let's assume no face is found on the whole image if no person was specified
        logger.info("Mock face detection: No person_bbox, so no face detected on whole image for this mock.")
        return None # Or, you could define a mock whole-image face detection here

    xmin_f, ymin_f, xmax_f, ymax_f = face_bbox_relative
    final_face_bbox = [xmin_f + offset_x, ymin_f + offset_y, xmax_f + offset_x, ymax_f + offset_y]
    logger.info(f"Prominent face detected with final bbox: {final_face_bbox}")
    return final_face_bbox


def get_clip_embedding(pil_image_rgb: Image.Image, crop_bbox: Optional[List[int]] = None) -> Tuple[List[float], Optional[str], Optional[List[int]]]:
    # For now, get_clip_embedding uses the globally defined MODEL_NAME_CLIP.
    # This will be parameterized in Stage 2.
    clip_model_instance, clip_preprocess_instance = model_loader.get_clip_model_and_preprocess(MODEL_NAME_CLIP)
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
        
        actual_crop_bbox = crop_bbox # Store the bbox used
        buffered = io.BytesIO()
        image_to_embed.save(buffered, format="PNG")
        base64_cropped_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
        logger.info(f"Cropped image for embedding (size: {image_to_embed.size}) encoded to base64.")

    logger.info(f"Generating CLIP embedding for image (size: {image_to_embed.size}) using {MODEL_NAME_CLIP}.")
    image_input = clip_preprocess_instance(image_to_embed).unsqueeze(0).to(current_device)
    with torch.no_grad():
        embedding_tensor = clip_model_instance.encode_image(image_input)
    embedding_list = embedding_tensor[0].cpu().numpy().tolist()
    logger.info("CLIP embedding generated successfully.")
    return embedding_list, base64_cropped_image, actual_crop_bbox

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
        return None, None, face_bbox # Return bbox even if invalid for context
    
    cropped_face = pil_image_rgb.crop(face_bbox)
    if cropped_face.width == 0 or cropped_face.height == 0:
        logger.error(f"Cropped face for 3DMM is empty using bbox: {face_bbox}")
        return None, None, face_bbox # Return bbox even if crop is empty

    buffered = io.BytesIO()
    cropped_face.save(buffered, format="PNG")
    base64_cropped_face = base64.b64encode(buffered.getvalue()).decode("utf-8")
    logger.info(f"Cropped face for 3DMM (size: {cropped_face.size}) encoded to base64.")

    mock_3dmm_params = {"shape_coeffs": [0.1]*50, "expression_coeffs": [0.2]*20, "pose": [0,0,0,1,0,0]}
    logger.info(f"Mock 3DMM fitting completed for face in bbox: {face_bbox}")
    return mock_3dmm_params, base64_cropped_face, face_bbox

# --- API Endpoint ---

@app.post("/analyze_image/", response_model=ImageAnalysisResponse)
async def analyze_image(request: ImageAnalysisRequest):
    image_url_str = str(request.image_url)
    logger.info(f"Received analysis request for URL: {image_url_str} with {len(request.tasks)} tasks.")

    try:
        pil_image_rgb = download_image(image_url_str)
    except HTTPException as e:
        results = {}
        for task_def in request.tasks:
            results[task_def.operation_id] = OperationResult(status="error", error_message=f"Failed to download or process base image: {e.detail}")
        return ImageAnalysisResponse(image_url=image_url_str, results=results)

    analysis_results: Dict[str, OperationResult] = {}
    shared_context: Dict[str, Any] = {
        "pil_image_rgb": pil_image_rgb,
        "prominent_person_bbox": None,
        "prominent_face_bbox": None,
    }
    person_detection_done = False
    face_detection_done = False

    for task_def in request.tasks:
        op_id = task_def.operation_id
        op_type = task_def.type
        # If task_def.params is None, default to an empty dict
        op_params = task_def.params if task_def.params is not None else {}
        
        # Get target from params, default to None if not specified
        target = op_params.get("target")
        # Get face_context from params, default to 'prominent_person' if not specified
        face_context = op_params.get("face_context", "prominent_person")


        logger.info(f"Processing task: id='{op_id}', type='{op_type}', params='{op_params}' (derived target='{target}', face_context='{face_context}')")

        try:
            current_result_data: Optional[Any] = None
            current_cropped_image_base64: Optional[str] = None
            current_cropped_image_bbox: Optional[List[int]] = None
            
            if op_type == "detect_bounding_box":
                # Default target for detect_bounding_box if not specified
                if target is None:
                    logger.info(f"Task {op_id}: 'target' not specified for detect_bounding_box, defaulting to 'prominent_person'.")
                    target = "prominent_person"

                if target == "prominent_person":
                    if not person_detection_done:
                        shared_context["prominent_person_bbox"] = get_prominent_person_bbox(shared_context["pil_image_rgb"])
                        person_detection_done = True
                    current_result_data = shared_context["prominent_person_bbox"]
                elif target == "prominent_face":
                    if not person_detection_done and face_context == "prominent_person":
                        shared_context["prominent_person_bbox"] = get_prominent_person_bbox(shared_context["pil_image_rgb"])
                        person_detection_done = True
                    if not face_detection_done:
                        person_bbox_for_face = shared_context["prominent_person_bbox"] if face_context == "prominent_person" else None
                        shared_context["prominent_face_bbox"] = get_prominent_face_bbox_in_region(shared_context["pil_image_rgb"], person_bbox_for_face)
                        face_detection_done = True
                    current_result_data = shared_context["prominent_face_bbox"]
                else:
                    raise ValueError(f"Unsupported target '{target}' for detect_bounding_box.")
            
            elif op_type == "embed_clip_vit_b_32":
                # Default target for embedding if not specified
                if target is None:
                    logger.info(f"Task {op_id}: 'target' not specified for embed_clip_vit_b_32, defaulting to 'whole_image'.")
                    target = "whole_image" 

                crop_for_embedding = None
                if target == "whole_image":
                    pass # No crop needed
                elif target == "prominent_person":
                    if not person_detection_done:
                        shared_context["prominent_person_bbox"] = get_prominent_person_bbox(shared_context["pil_image_rgb"])
                        person_detection_done = True
                    if shared_context["prominent_person_bbox"]:
                        crop_for_embedding = shared_context["prominent_person_bbox"]
                    else: # No person found, embed whole image as fallback
                        logger.warning(f"Task {op_id}: prominent_person target for embedding, but no person found. Embedding whole image as fallback.")
                        # crop_for_embedding remains None, so whole image is used by get_clip_embedding
                elif target == "prominent_face":
                    if not face_detection_done: # Ensure face detection has run if needed
                         if not person_detection_done and face_context == "prominent_person":
                            shared_context["prominent_person_bbox"] = get_prominent_person_bbox(shared_context["pil_image_rgb"])
                            person_detection_done = True
                         person_bbox_for_face = shared_context["prominent_person_bbox"] if face_context == "prominent_person" else None
                         shared_context["prominent_face_bbox"] = get_prominent_face_bbox_in_region(shared_context["pil_image_rgb"], person_bbox_for_face)
                         face_detection_done = True
                    if shared_context["prominent_face_bbox"]:
                        crop_for_embedding = shared_context["prominent_face_bbox"]
                    else: # No face found
                        raise ValueError("No prominent face found for embedding.")
                else:
                    raise ValueError(f"Unsupported target '{target}' for embed_clip_vit_b_32.")
                
                embedding_list, b64_img, bbox_used = get_clip_embedding(shared_context["pil_image_rgb"], crop_bbox=crop_for_embedding)
                current_result_data = embedding_list
                current_cropped_image_base64 = b64_img
                current_cropped_image_bbox = bbox_used


            elif op_type == "fit_3dmm":
                if target is None:
                    logger.info(f"Task {op_id}: 'target' not specified for fit_3dmm, defaulting to 'prominent_face'.")
                    target = "prominent_face"

                if target == "prominent_face":
                    if not face_detection_done: # Ensure face detection has run if needed
                        if not person_detection_done and face_context == "prominent_person":
                           shared_context["prominent_person_bbox"] = get_prominent_person_bbox(shared_context["pil_image_rgb"])
                           person_detection_done = True
                        person_bbox_for_face = shared_context["prominent_person_bbox"] if face_context == "prominent_person" else None
                        shared_context["prominent_face_bbox"] = get_prominent_face_bbox_in_region(shared_context["pil_image_rgb"], person_bbox_for_face)
                        face_detection_done = True
                    
                    if shared_context["prominent_face_bbox"]:
                         # The check for model availability is now inside fit_3dmm_on_face using the loader
                         params, b64_img, bbox_used = fit_3dmm_on_face(shared_context["pil_image_rgb"], shared_context["prominent_face_bbox"])
                         current_result_data = params
                         current_cropped_image_base64 = b64_img
                         current_cropped_image_bbox = bbox_used
                         # If params and b64_img are None, fit_3dmm_on_face already logged warnings or errors.
                         # We might raise a more generic error if the operation was expected to succeed but returned None.
                         if params is None and model_loader.get_threedmm_model() is not None: # Check if model was supposed to be available
                             raise ValueError("3DMM fitting failed for the prominent face despite model being configured (mocked).")
                    else: # No face found
                        raise ValueError("No prominent face found for 3DMM fitting.")
                else:
                    raise ValueError(f"Unsupported target '{target}' for fit_3dmm.")
            else:
                raise ValueError(f"Unsupported operation type: {op_type}")

            analysis_results[op_id] = OperationResult(
                status="success", 
                data=current_result_data,
                cropped_image_base64=current_cropped_image_base64,
                cropped_image_bbox=current_cropped_image_bbox
            )
            logger.info(f"Successfully processed task: id='{op_id}'")

        except ValueError as ve: # For expected issues like "unsupported target", "no face found"
            logger.warning(f"Skipping task {op_id} due to ValueError: {ve}")
            analysis_results[op_id] = OperationResult(status="skipped", error_message=str(ve))
        except HTTPException: # Re-raise HTTPExceptions (e.g. from download_image)
            raise
        except Exception as e: # For unexpected internal errors
            logger.exception(f"Failed to process task {op_id} ('{op_type}' on target '{target}'): {e}")
            analysis_results[op_id] = OperationResult(status="error", error_message=f"Internal error processing task: {e}")

    return ImageAnalysisResponse(image_url=image_url_str, results=analysis_results)

# --- How to Run (for development) ---
# uvicorn main:app --reload
#
# Example POST request to http://localhost:8000/analyze_image/
# {
#   "image_url": "YOUR_IMAGE_URL_HERE",
#   "tasks": [
#     // Embedding the whole image (target specified)
#     {"operation_id": "whole_image_emb", "type": "embed_clip_vit_b_32", "params": {"target": "whole_image"}},
#     
#     // Detecting prominent person (target can be omitted, defaults to prominent_person)
#     {"operation_id": "person_box_default", "type": "detect_bounding_box"}, 
#     // Explicit version:
#     // {"operation_id": "person_box_explicit", "type": "detect_bounding_box", "params": {"target": "prominent_person"}},
#     
#     // Embedding prominent person (target can be specified, or if omitted, could default - here we default embed to whole_image unless specified)
#     // To get prominent person embedding, you MUST specify target.
#     {"operation_id": "person_emb", "type": "embed_clip_vit_b_32", "params": {"target": "prominent_person"}},
#
#     // Detecting prominent face (target can be omitted, defaults to prominent_face; face_context defaults to prominent_person)
#     {"operation_id": "face_box_default", "type": "detect_bounding_box", "params": {"target":"prominent_face"}}, // target prominent_face needed here
#     // Explicit version:
#     // {"operation_id": "face_box_explicit", "type": "detect_bounding_box", "params": {"target": "prominent_face", "face_context": "prominent_person"}},
#
#     // Fitting 3DMM on prominent face (target can be omitted for fit_3dmm, defaults to prominent_face; face_context defaults to prominent_person)
#     {"operation_id": "face_3dmm_default", "type": "fit_3dmm"}
#     // Explicit version:
#     // {"operation_id": "face_3dmm_explicit", "type": "fit_3dmm", "params": {"target": "prominent_face", "face_context": "prominent_person"}}
#   ]
# }

if __name__ == "__main__":
    import uvicorn
    # This part is for direct execution (python main.py),
    # but typically you'd use uvicorn CLI for more control.
    uvicorn.run(app, host="0.0.0.0", port=8000)
