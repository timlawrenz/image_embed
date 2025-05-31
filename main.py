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
# Services will be imported below, model_loader is used by services
# from app.core import model_loader # No longer directly used in main.py, but by services

# Import service functions
from app.services.image_utils import download_image
from app.services.detection_service import get_prominent_person_bbox, get_prominent_face_bbox_in_region
from app.services.embedding_service import get_clip_embedding
from app.services.threedmm_service import fit_3dmm_on_face


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
# threedmm_model = model_loader.get_threedmm_model() # This line is commented out

# --- Pydantic Models have been moved to app/pydantic_models.py ---

# --- Helper Functions for Image Processing have been moved to app/services/ ---

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


        log_details = f"derived target='{target}'"
        # face_context is relevant if the target is a prominent_face,
        # or if the operation is fit_3dmm (which defaults to prominent_face).
        if target == "prominent_face" or op_type == "fit_3dmm":
            log_details += f", face_context='{face_context}'"
        
        logger.info(f"Processing task: id='{op_id}', type='{op_type}', params='{op_params}' ({log_details})")

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
                
                # Pass MODEL_NAME_CLIP to the service function
                embedding_list, b64_img, bbox_used = get_clip_embedding(
                    shared_context["pil_image_rgb"], 
                    clip_model_name=MODEL_NAME_CLIP, # Pass the default model name
                    crop_bbox=crop_for_embedding
                )
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
                         # Accessing model_loader directly here is less ideal after refactoring,
                         # but for this specific error check, it's a small point.
                         # The core logic of whether the 3dmm model is available is handled within fit_3dmm_on_face.
                         if params is None and app.core.model_loader.get_threedmm_model() is not None: # Check if model was supposed to be available
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
