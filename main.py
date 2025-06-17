import io
import logging
from contextlib import asynccontextmanager
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
from app.core import model_loader

# Import service functions
from app.services.image_utils import download_image
from app.services.detection_service import get_prominent_person_bbox, get_prominent_face_bbox_in_region
from app.services.embedding_service import get_clip_embedding
from app.services.classification_service import classify_embedding


# --- Configuration & Initialization ---

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events for the application.
    Loads all models on startup by calling the centralized preloader.
    """
    logger.info("Application startup: Triggering model pre-loading...")
    model_loader.preload_all_models(clip_model_name=MODEL_NAME_CLIP)
    yield
    logger.info("Application shutting down.")


# Initialize FastAPI app
app = FastAPI(
    title="Advanced Image Analysis API",
    description="Performs various analyses on an image from a URL, including embeddings, object detection, and more.",
    version="0.3.0",
    lifespan=lifespan,
)

# --- Model Loading ---
# Models are pre-loaded at startup via the lifespan manager.

# Default CLIP model name (can be made configurable or part of request later)
MODEL_NAME_CLIP = "ViT-B/32"

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
    "classify": {
        "description": "Classifies an image region using a pre-trained model for a specific collection.",
        "allowed_targets": ["whole_image", "prominent_person", "prominent_face"],
        "default_target": "whole_image",
    }
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
    shared_context: Dict[str, Any] = {"pil_image_rgb": pil_image_rgb}
    person_detection_done = False
    face_detection_done = False

    # --- Helper function to get embeddings on demand, with caching ---
    def get_embedding_for_target(
        target: str, face_context: str, op_id: str
    ) -> Tuple[Optional[List[float]], Optional[str], Optional[List[int]]]:
        nonlocal person_detection_done, face_detection_done

        embedding_cache_key = f"embedding_{target}"
        if "face" in target:
            embedding_cache_key += f"_{face_context}"
        
        if embedding_cache_key in shared_context:
            logger.info(f"Task {op_id}: Using cached embedding from '{embedding_cache_key}'.")
            return shared_context[embedding_cache_key]

        logger.info(f"Task {op_id}: No cached embedding for '{embedding_cache_key}', computing now...")
        crop_box = None
        if target == "whole_image":
            pass
        elif target == "prominent_person":
            if not person_detection_done:
                shared_context["prominent_person_bbox"] = get_prominent_person_bbox(pil_image_rgb)
                person_detection_done = True
            if shared_context.get("prominent_person_bbox"):
                crop_box = shared_context["prominent_person_bbox"]
            else:
                logger.warning(f"Task {op_id}: prominent_person target, but no person found. Using whole image.")
        elif target == "prominent_face":
            if not face_detection_done:
                 if not person_detection_done and face_context == "prominent_person":
                    shared_context["prominent_person_bbox"] = get_prominent_person_bbox(pil_image_rgb)
                    person_detection_done = True
                 person_bbox_for_face = shared_context.get("prominent_person_bbox") if face_context == "prominent_person" else None
                 shared_context["prominent_face_bbox"] = get_prominent_face_bbox_in_region(pil_image_rgb, person_bbox_for_face)
                 face_detection_done = True
            if shared_context.get("prominent_face_bbox"):
                crop_box = shared_context["prominent_face_bbox"]
            else:
                raise ValueError(f"No prominent face found for operation '{op_id}'.")
        
        embedding_list, b64_img, bbox_used = get_clip_embedding(pil_image_rgb, MODEL_NAME_CLIP, crop_box)
        shared_context[embedding_cache_key] = (embedding_list, b64_img, bbox_used)
        return embedding_list, b64_img, bbox_used

    # --- Process Tasks ---
    for task_def in request.tasks:
        op_id = task_def.operation_id
        op_type = task_def.type
        op_params = task_def.params if task_def.params is not None else {}
        target = op_params.get("target")

        try:
            op_info = AVAILABLE_OPERATIONS.get(op_type)
            if not op_info:
                raise ValueError(f"Unsupported operation type: '{op_type}'")

            if target is None:
                target = op_info["default_target"]
                logger.info(f"Task {op_id}: 'target' not specified, defaulting to '{target}'.")
            
            if target not in op_info["allowed_targets"]:
                raise ValueError(f"Unsupported target '{target}' for '{op_type}'. Allowed: {op_info['allowed_targets']}")

            face_context = op_params.get("face_context", "prominent_person")
            
            log_message = f"Processing task: id='{op_id}', type='{op_type}', target='{target}'"
            if "face" in target or op_type == "classify":
                 log_message += f", face_context='{face_context}'"
            logger.info(log_message)

            current_result_data: Optional[Any] = None
            current_cropped_image_base64: Optional[str] = None
            current_cropped_image_bbox: Optional[List[int]] = None
            
            if op_type == "detect_bounding_box":
                if target == "prominent_person":
                    if not person_detection_done:
                        shared_context["prominent_person_bbox"] = get_prominent_person_bbox(pil_image_rgb)
                        person_detection_done = True
                    current_result_data = shared_context.get("prominent_person_bbox")
                elif target == "prominent_face":
                    if not person_detection_done and face_context == "prominent_person":
                        shared_context["prominent_person_bbox"] = get_prominent_person_bbox(pil_image_rgb)
                        person_detection_done = True
                    if not face_detection_done:
                        person_bbox = shared_context.get("prominent_person_bbox") if face_context == "prominent_person" else None
                        shared_context["prominent_face_bbox"] = get_prominent_face_bbox_in_region(pil_image_rgb, person_bbox)
                        face_detection_done = True
                    current_result_data = shared_context.get("prominent_face_bbox")
            
            elif op_type == "embed_clip_vit_b_32":
                embedding_list, b64_img, bbox_used = get_embedding_for_target(target, face_context, op_id)
                current_result_data = embedding_list
                current_cropped_image_base64 = b64_img
                current_cropped_image_bbox = bbox_used
            
            elif op_type == "classify":
                collection_id = op_params.get("collection_id")
                if collection_id is None:
                    raise ValueError("'collection_id' param is required for 'classify' operation.")
                
                embedding, b64_img, bbox_used = get_embedding_for_target(target, face_context, op_id)
                if embedding is None:
                    raise ValueError("Could not generate embedding, classification cannot proceed.")
                
                try:
                    current_result_data = classify_embedding(embedding, int(collection_id))
                    current_cropped_image_base64 = b64_img
                    current_cropped_image_bbox = bbox_used
                except FileNotFoundError as e:
                    raise ValueError(f"Classifier not found for collection_id {collection_id}.") from e

            analysis_results[op_id] = OperationResult(
                status="success", 
                data=current_result_data,
                cropped_image_base64=current_cropped_image_base64,
                cropped_image_bbox=current_cropped_image_bbox
            )
            logger.info(f"Successfully processed task: id='{op_id}'")

        except ValueError as ve:
            logger.warning(f"Skipping task {op_id} due to ValueError: {ve}")
            analysis_results[op_id] = OperationResult(status="skipped", error_message=str(ve))
        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Failed to process task {op_id} ('{op_type}' on target '{target}'): {e}")
            analysis_results[op_id] = OperationResult(status="error", error_message=f"Internal error processing task: {e}")

    return ImageAnalysisResponse(image_url=image_url_str, results=analysis_results)


if __name__ == "__main__":
    import uvicorn
    # This part is for direct execution (python main.py),
    # but typically you'd use uvicorn CLI for more control.
    uvicorn.run(app, host="0.0.0.0", port=8000)
