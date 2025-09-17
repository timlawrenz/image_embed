import io
import logging
import warnings
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Tuple # Added Tuple

# Suppress a specific UserWarning from the clip library regarding pkg_resources.
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API", category=UserWarning)

import clip
import requests
import torch
import torchvision # Added for detection model
import torchvision.transforms as T # Added for detection model
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from PIL import Image
import json # For parsing tasks from form data
# Pydantic models moved to app.pydantic_models
import base64 # Added
from app.pydantic_models import AnalysisTask, ImageAnalysisRequest, OperationResult, ImageAnalysisResponse, AvailableOperationsResponse
# Services will be imported below, model_loader is used by services
from app.core import model_loader

# Import service functions
from app.services.image_utils import download_image, process_uploaded_image
from app.services.detection_service import get_prominent_person_bbox, get_prominent_face_bbox_in_region
from app.services.embedding_service import get_clip_embedding, get_dino_embedding
from app.services.classification_service import classify_embedding
from app.services.description_service import get_image_description

import time
import os
import uuid


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

@app.middleware("http")
async def add_request_timing_and_pid_logging(request: Request, call_next):
    """
    Middleware to log request processing time and the worker process ID.
    """
    request_id = str(uuid.uuid4())
    pid = os.getpid()
    start_time = time.time()
    
    logger.info(f"req_id={request_id} pid={pid} event=request_started path={request.url.path}")

    response = await call_next(request)
    
    process_time = time.time() - start_time
    logger.info(f"req_id={request_id} pid={pid} event=request_finished path={request.url.path} duration={process_time:.4f}s")
    
    return response

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
    "embed_dino_v2": {
        "description": "Generates a visual embedding using the DINOv2 model.",
        "allowed_targets": ["whole_image", "prominent_person", "prominent_face"],
        "default_target": "whole_image",
    },
    "classify": {
        "description": "Classifies an image region using a pre-trained model for a specific collection.",
        "allowed_targets": ["whole_image", "prominent_person", "prominent_face"],
        "default_target": "whole_image",
    },
    "describe_image": {
        "description": "Generates a text description of an image region.",
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


def _perform_analysis(pil_image_rgb: Image.Image, tasks: List[AnalysisTask]) -> Tuple[Dict[str, OperationResult], Dict[str, float]]:
    """
    Core logic to perform a list of analysis tasks on a given image.
    Returns a dictionary of results and a dictionary of timing stats.
    """
    analysis_results: Dict[str, OperationResult] = {}
    shared_context: Dict[str, Any] = {"pil_image_rgb": pil_image_rgb}
    person_detection_done = False
    face_detection_done = False
    timing_stats = {"detection": 0.0, "embedding": 0.0, "classification": 0.0, "description": 0.0}

    # --- Helper function to get embeddings on demand, with caching ---
    def get_embedding_for_target(
        target: str, face_context: str, op_id: str
    ) -> Tuple[Optional[List[float]], Optional[str], Optional[List[int]]]:
        nonlocal person_detection_done, face_detection_done, timing_stats

        embedding_cache_key = f"embedding_{target}"
        if "face" in target:
            embedding_cache_key += f"_{face_context}"
        
        if embedding_cache_key in shared_context:
            return shared_context[embedding_cache_key]

        crop_box = None
        if target == "whole_image":
            pass
        elif target == "prominent_person":
            if not person_detection_done:
                detection_start = time.time()
                shared_context["prominent_person_bbox"] = get_prominent_person_bbox(pil_image_rgb)
                timing_stats["detection"] += time.time() - detection_start
                person_detection_done = True
            if shared_context.get("prominent_person_bbox"):
                crop_box = shared_context["prominent_person_bbox"]
            else:
                logger.warning(f"Task {op_id}: prominent_person target, but no person found. Using whole image.")
        elif target == "prominent_face":
            if not face_detection_done:
                 detection_block_start = time.time()
                 if not person_detection_done and face_context == "prominent_person":
                    shared_context["prominent_person_bbox"] = get_prominent_person_bbox(pil_image_rgb)
                    person_detection_done = True
                 person_bbox_for_face = shared_context.get("prominent_person_bbox") if face_context == "prominent_person" else None
                 shared_context["prominent_face_bbox"] = get_prominent_face_bbox_in_region(pil_image_rgb, person_bbox_for_face)
                 face_detection_done = True
                 timing_stats["detection"] += time.time() - detection_block_start
            if shared_context.get("prominent_face_bbox"):
                crop_box = shared_context["prominent_face_bbox"]
            else:
                raise ValueError(f"No prominent face found for operation '{op_id}'.")
        
        embedding_start = time.time()
        embedding_list, b64_img, bbox_used = get_clip_embedding(pil_image_rgb, MODEL_NAME_CLIP, crop_box)
        timing_stats["embedding"] += time.time() - embedding_start
        shared_context[embedding_cache_key] = (embedding_list, b64_img, bbox_used)
        return embedding_list, b64_img, bbox_used

    # --- Process Tasks ---
    for task_def in tasks:
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
            
            current_result_data: Optional[Any] = None
            current_cropped_image_base64: Optional[str] = None
            current_cropped_image_bbox: Optional[List[int]] = None
            
            if op_type == "detect_bounding_box":
                if target == "prominent_person":
                    if not person_detection_done:
                        detection_start = time.time()
                        shared_context["prominent_person_bbox"] = get_prominent_person_bbox(pil_image_rgb)
                        timing_stats["detection"] += time.time() - detection_start
                        person_detection_done = True
                    current_result_data = shared_context.get("prominent_person_bbox")
                elif target == "prominent_face":
                    detection_block_start = time.time()
                    if not person_detection_done and face_context == "prominent_person":
                        shared_context["prominent_person_bbox"] = get_prominent_person_bbox(pil_image_rgb)
                        person_detection_done = True
                    if not face_detection_done:
                        person_bbox = shared_context.get("prominent_person_bbox") if face_context == "prominent_person" else None
                        shared_context["prominent_face_bbox"] = get_prominent_face_bbox_in_region(pil_image_rgb, person_bbox)
                        face_detection_done = True
                    timing_stats["detection"] += time.time() - detection_block_start
                    current_result_data = shared_context.get("prominent_face_bbox")
            
            elif op_type == "embed_clip_vit_b_32":
                embedding_list, b64_img, bbox_used = get_embedding_for_target(target, face_context, op_id)
                current_result_data = embedding_list
                current_cropped_image_base64 = b64_img
                current_cropped_image_bbox = bbox_used
            
            elif op_type == "embed_dino_v2":
                # This part is not using the shared embedding cache, as DINO is a different embedding type.
                embedding_start = time.time()
                crop_box_for_dino = None
                if target == "prominent_person":
                    if not person_detection_done:
                        detection_start = time.time()
                        shared_context["prominent_person_bbox"] = get_prominent_person_bbox(pil_image_rgb)
                        timing_stats["detection"] += time.time() - detection_start
                        person_detection_done = True
                    if shared_context.get("prominent_person_bbox"):
                        crop_box_for_dino = shared_context["prominent_person_bbox"]
                elif target == "prominent_face":
                    if not face_detection_done:
                        detection_block_start = time.time()
                        if not person_detection_done and face_context == "prominent_person":
                            shared_context["prominent_person_bbox"] = get_prominent_person_bbox(pil_image_rgb)
                            person_detection_done = True
                        person_bbox_for_face = shared_context.get("prominent_person_bbox") if face_context == "prominent_person" else None
                        shared_context["prominent_face_bbox"] = get_prominent_face_bbox_in_region(pil_image_rgb, person_bbox_for_face)
                        face_detection_done = True
                        timing_stats["detection"] += time.time() - detection_block_start
                    if shared_context.get("prominent_face_bbox"):
                        crop_box_for_dino = shared_context["prominent_face_bbox"]
                    else:
                        raise ValueError(f"No prominent face found for operation '{op_id}'.")
                
                embedding_list, b64_img, bbox_used = get_dino_embedding(pil_image_rgb, crop_bbox=crop_box_for_dino)
                timing_stats["embedding"] += time.time() - embedding_start
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
                
                classification_start = time.time()
                try:
                    current_result_data = classify_embedding(embedding, int(collection_id))
                    current_cropped_image_base64 = b64_img
                    current_cropped_image_bbox = bbox_used
                except FileNotFoundError as e:
                    raise ValueError(f"Classifier not found for collection_id {collection_id}.") from e
                timing_stats["classification"] += time.time() - classification_start

            elif op_type == "describe_image":
                max_length = op_params.get("max_length", 1024)
                crop_box_for_desc = None
                if target == "prominent_person":
                    if not person_detection_done:
                        detection_start = time.time()
                        shared_context["prominent_person_bbox"] = get_prominent_person_bbox(pil_image_rgb)
                        timing_stats["detection"] += time.time() - detection_start
                        person_detection_done = True
                    if shared_context.get("prominent_person_bbox"):
                        crop_box_for_desc = shared_context["prominent_person_bbox"]
                    else:
                        logger.warning(f"Task {op_id}: prominent_person target, but no person found. Describing whole image.")
                elif target == "prominent_face":
                    if not face_detection_done:
                        detection_block_start = time.time()
                        if not person_detection_done and face_context == "prominent_person":
                            shared_context["prominent_person_bbox"] = get_prominent_person_bbox(pil_image_rgb)
                            person_detection_done = True
                        person_bbox_for_face = shared_context.get("prominent_person_bbox") if face_context == "prominent_person" else None
                        shared_context["prominent_face_bbox"] = get_prominent_face_bbox_in_region(pil_image_rgb, person_bbox_for_face)
                        face_detection_done = True
                        timing_stats["detection"] += time.time() - detection_block_start
                    if shared_context.get("prominent_face_bbox"):
                        crop_box_for_desc = shared_context["prominent_face_bbox"]
                    else:
                        raise ValueError(f"No prominent face found for operation '{op_id}'.")

                description_start = time.time()
                caption, b64_img, bbox_used = get_image_description(pil_image_rgb, crop_box_for_desc, max_length)
                timing_stats["description"] += time.time() - description_start
                
                current_result_data = caption
                current_cropped_image_base64 = b64_img
                current_cropped_image_bbox = bbox_used

            analysis_results[op_id] = OperationResult(
                status="success", 
                data=current_result_data,
                cropped_image_base64=current_cropped_image_base64,
                cropped_image_bbox=current_cropped_image_bbox
            )

        except ValueError as ve:
            logger.warning(f"Skipping task {op_id} due to ValueError: {ve}")
            analysis_results[op_id] = OperationResult(status="skipped", error_message=str(ve))
        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Failed to process task {op_id} ('{op_type}' on target '{target}'): {e}")
            analysis_results[op_id] = OperationResult(status="error", error_message=f"Internal error processing task: {e}")

    return analysis_results, timing_stats


@app.post("/analyze_image/", response_model=ImageAnalysisResponse, tags=["Analysis"])
async def analyze_image(request: ImageAnalysisRequest):
    req_start_time = time.time()
    image_url_str = str(request.image_url)
    logger.info(f"Received analysis request for URL: {image_url_str} with {len(request.tasks)} tasks.")

    download_start_time = time.time()
    try:
        pil_image_rgb = download_image(image_url_str)
    except HTTPException as e:
        results = {
            task.operation_id: OperationResult(status="error", error_message=f"Failed to download or process base image: {e.detail}")
            for task in request.tasks
        }
        return ImageAnalysisResponse(image_url=image_url_str, results=results)
    download_duration = time.time() - download_start_time

    analysis_results, timing_stats = _perform_analysis(pil_image_rgb, request.tasks)
    timing_stats["download"] = download_duration

    total_duration = time.time() - req_start_time
    timed_duration = sum(timing_stats.values())
    timing_stats["other"] = total_duration - timed_duration
    timing_stats["total"] = total_duration
    
    stats_str = " ".join([f"{k}={v:.4f}s" for k, v in timing_stats.items()])
    logger.info(f"Analysis timing stats for URL {image_url_str}: {stats_str}")

    return ImageAnalysisResponse(image_url=image_url_str, results=analysis_results)


@app.post("/analyze_image_upload/", response_model=ImageAnalysisResponse, tags=["Analysis"])
async def analyze_image_upload(
    tasks_json: str = Form(..., description='A JSON string of the analysis tasks list, conforming to the structure of the "tasks" field in the /analyze_image endpoint.'),
    image_file: UploadFile = File(..., description="Image file to analyze.")
):
    """
    Performs a series of analyses on a directly uploaded image.

    This endpoint is suitable for server-to-server communication where the image
    is available locally and can be sent as part of a multipart/form-data request,
    avoiding the need to expose it via a public URL.
    """
    req_start_time = time.time()
    logger.info(f"Received analysis request for uploaded file: {image_file.filename}")

    try:
        tasks_data = json.loads(tasks_json)
        # Using model_validate, which is for Pydantic v2. Use .parse_obj for v1.
        tasks = [AnalysisTask.model_validate(task) for task in tasks_data]
    except (json.JSONDecodeError, ValueError) as e: # ValueError from Pydantic validation
        raise HTTPException(status_code=400, detail=f"Invalid tasks JSON format or structure: {e}")

    processing_start_time = time.time()
    try:
        image_bytes = await image_file.read()
        pil_image_rgb = process_uploaded_image(image_bytes)
    except HTTPException as e:
        results = {
            task.operation_id: OperationResult(status="error", error_message=f"Failed to process base image: {e.detail}")
            for task in tasks
        }
        return ImageAnalysisResponse(image_url=f"uploaded:{image_file.filename}", results=results)
    processing_duration = time.time() - processing_start_time

    analysis_results, timing_stats = _perform_analysis(pil_image_rgb, tasks)
    timing_stats["processing"] = processing_duration

    total_duration = time.time() - req_start_time
    timed_duration = sum(timing_stats.values())
    timing_stats["other"] = total_duration - timed_duration
    timing_stats["total"] = total_duration

    stats_str = " ".join([f"{k}={v:.4f}s" for k, v in timing_stats.items()])
    logger.info(f"Analysis timing stats for uploaded file {image_file.filename}: {stats_str}")

    return ImageAnalysisResponse(image_url=f"uploaded:{image_file.filename}", results=analysis_results)


if __name__ == "__main__":
    import uvicorn
    # This part is for direct execution (python main.py),
    # but typically you'd use uvicorn CLI for more control.
    uvicorn.run(app, host="0.0.0.0", port=8000)
