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
from app.pydantic_models import AnalysisTask, ImageAnalysisRequest, OperationResult, ImageAnalysisResponse, AvailableOperationsResponse
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
    version="0.3.0",
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

# --- Centralized Operation Definitions ---
AVAILABLE_OPERATIONS = {
    "detect_bounding_box": {
        "description": "Detects a bounding box for a specified target.",
        "allowed_targets": ["prominent_person", "prominent_face"],
        "default_target": "prominent_person",
    },
    "embed_clip_vit_b_32": {
        "description": f"Generates an embedding using the CLIP {MODEL_NAME_CLIP} model.",
        "allowed_targets": ["whole_image", "prominent_person", "prominent_face"],
        "default_target": "whole_image",
    },
    "fit_3dmm": {
        "description": "Fits a 3D Morphable Model (mock implementation).",
        "allowed_targets": ["prominent_face"],
        "default_target": "prominent_face",
    },
}

# --- API Endpoints ---

@app.get("/available_operations/", response_model=AvailableOperationsResponse, tags=["Configuration"])
async def get_available_operations():
    """
    Provides a list of available analysis operations, their allowed targets,
    and default targets.
    """
    return {"operations": AVAILABLE_OPERATIONS}


@app.post("/analyze_image/", response_model=ImageAnalysisResponse, tags=["Analysis"])
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
        op_params = task_def.params if task_def.params is not None else {}
        
        # Define target here, it may be None initially.
        # This ensures it's available in the final exception handler's log message.
        target = op_params.get("target")

        try:
            # --- Centralized Validation and Defaulting ---
            op_info = AVAILABLE_OPERATIONS.get(op_type)
            if not op_info:
                raise ValueError(f"Unsupported operation type: '{op_type}'")

            # If target was not specified in params, get the default.
            if target is None:
                target = op_info["default_target"]
                logger.info(f"Task {op_id}: 'target' not specified for '{op_type}', defaulting to '{target}'.")
            
            # Validate the final target against the allowed list for the operation.
            if target not in op_info["allowed_targets"]:
                raise ValueError(f"Unsupported target '{target}' for operation '{op_type}'. Allowed targets are: {op_info['allowed_targets']}")

            face_context = op_params.get("face_context", "prominent_person")
            # --- End Validation ---

            log_message = f"Processing task: id='{op_id}', type='{op_type}', target='{target}'"
            if "face" in target:
                 log_message += f", face_context='{face_context}'"
            logger.info(log_message)

            current_result_data: Optional[Any] = None
            current_cropped_image_base64: Optional[str] = None
            current_cropped_image_bbox: Optional[List[int]] = None
            
            if op_type == "detect_bounding_box":
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
            
            elif op_type == "embed_clip_vit_b_32":
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
                
                embedding_list, b64_img, bbox_used = get_clip_embedding(
                    shared_context["pil_image_rgb"], 
                    clip_model_name=MODEL_NAME_CLIP,
                    crop_bbox=crop_for_embedding
                )
                current_result_data = embedding_list
                current_cropped_image_base64 = b64_img
                current_cropped_image_bbox = bbox_used

            elif op_type == "fit_3dmm":
                if target == "prominent_face":
                    if not face_detection_done: # Ensure face detection has run if needed
                        if not person_detection_done and face_context == "prominent_person":
                           shared_context["prominent_person_bbox"] = get_prominent_person_bbox(shared_context["pil_image_rgb"])
                           person_detection_done = True
                        person_bbox_for_face = shared_context["prominent_person_bbox"] if face_context == "prominent_person" else None
                        shared_context["prominent_face_bbox"] = get_prominent_face_bbox_in_region(shared_context["pil_image_rgb"], person_bbox_for_face)
                        face_detection_done = True
                    
                    if shared_context["prominent_face_bbox"]:
                         params, b64_img, bbox_used = fit_3dmm_on_face(shared_context["pil_image_rgb"], shared_context["prominent_face_bbox"])
                         current_result_data = params
                         current_cropped_image_base64 = b64_img
                         current_cropped_image_bbox = bbox_used
                         if params is None and app.core.model_loader.get_threedmm_model() is not None:
                             raise ValueError("3DMM fitting failed for the prominent face despite model being configured (mocked).")
                    else: # No face found
                        raise ValueError("No prominent face found for 3DMM fitting.")

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


if __name__ == "__main__":
    import uvicorn
    # This part is for direct execution (python main.py),
    # but typically you'd use uvicorn CLI for more control.
    uvicorn.run(app, host="0.0.0.0", port=8000)
