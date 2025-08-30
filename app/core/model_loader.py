import logging
import torch
import clip
import torchvision
from facenet_pytorch import MTCNN
import os
import glob
import re
import joblib
import json
from transformers import BlipProcessor, BlipForConditionalGeneration

logger = logging.getLogger(__name__)

# --- Global Cache for Models ---
_loaded_models = {}

# --- Global for Best Models Config ---
_best_models_map = None

# --- Device Configuration ---
try:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"ModelLoader: Using device: {DEVICE}")
except Exception as e:
    logger.error(f"ModelLoader: Error determining Torch device, defaulting to CPU. Error: {e}")
    DEVICE = "cpu"


def _load_best_models_map():
    """Loads the best models map from the JSON configuration file."""
    global _best_models_map
    if _best_models_map is not None:  # Only load once
        return

    config_path = os.path.join("trained_classifiers", "best_models.json")
    _best_models_map = {}  # Default to empty map

    if not os.path.exists(config_path):
        logger.info(f"ModelLoader: '{config_path}' not found. Will use date-based model selection.")
        return

    try:
        with open(config_path, 'r') as f:
            _best_models_map = json.load(f)
        logger.info(f"ModelLoader: Successfully loaded best models configuration from '{config_path}'")
    except (IOError, json.JSONDecodeError) as e:
        logger.warning(f"ModelLoader: Could not read or parse '{config_path}': {e}. Will fall back to date-based model selection.")


def get_device() -> str:
    return DEVICE

def get_clip_model_and_preprocess(model_name: str):
    """
    Loads a specified CLIP model and its preprocess function.
    Caches the model and preprocess function for subsequent calls.
    """
    cache_key_model = f"clip_{model_name}_model"
    cache_key_preprocess = f"clip_{model_name}_preprocess"

    if cache_key_model not in _loaded_models:
        logger.info(f"ModelLoader: Loading CLIP model '{model_name}' on {DEVICE}...")
        try:
            model, preprocess = clip.load(model_name, device=DEVICE, jit=False)
            _loaded_models[cache_key_model] = model
            _loaded_models[cache_key_preprocess] = preprocess
            logger.info(f"ModelLoader: CLIP model '{model_name}' loaded and cached successfully.")
        except Exception as e:
            logger.exception(f"ModelLoader: Failed to load CLIP model '{model_name}'.")
            raise RuntimeError(f"Could not load CLIP model '{model_name}': {e}") from e
    else:
        logger.debug(f"ModelLoader: Using cached CLIP model '{model_name}'.")
        model = _loaded_models[cache_key_model]
        preprocess = _loaded_models[cache_key_preprocess]
    
    return model, preprocess

def get_person_detection_model():
    """
    Loads the Faster R-CNN person detection model.
    Caches the model for subsequent calls.
    """
    cache_key = "person_detection_model"
    if cache_key not in _loaded_models:
        logger.info("ModelLoader: Loading Person detection model (Faster R-CNN)...")
        try:
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            )
            model.to(DEVICE)
            model.eval()
            _loaded_models[cache_key] = model
            logger.info("ModelLoader: Person detection model loaded and cached successfully.")
        except Exception as e:
            logger.exception("ModelLoader: Failed to load person detection model.")
            raise RuntimeError(f"Could not load person detection model: {e}") from e
    else:
        logger.debug("ModelLoader: Using cached Person detection model.")
        model = _loaded_models[cache_key]
        
    return model

def get_face_detection_model():
    """
    Loads the MTCNN model for face detection from facenet-pytorch.
    Caches the model for subsequent calls.
    """
    cache_key = "face_detection_model"
    if cache_key not in _loaded_models:
        logger.info("ModelLoader: Loading Face detection model (MTCNN)...")
        try:
            # keep_all=True returns all faces, not just the one with the highest probability
            # this allows the service layer to decide which face is 'prominent'
            model = MTCNN(device=DEVICE, keep_all=True)
            _loaded_models[cache_key] = model
            logger.info("ModelLoader: Face detection model loaded and cached successfully.")
        except Exception as e:
            logger.exception("ModelLoader: Failed to load face detection model.")
            # Set to None so we don't retry on every call in this session
            _loaded_models[cache_key] = None
    
    return _loaded_models[cache_key]


def get_image_captioning_model_and_processor(model_name: str = "Salesforce/blip-image-captioning-large"):
    """
    Loads and caches an image captioning model and its processor from Hugging Face.
    """
    cache_key_model = f"caption_{model_name}_model"
    cache_key_processor = f"caption_{model_name}_processor"

    if cache_key_model not in _loaded_models:
        logger.info(f"ModelLoader: Loading image captioning model '{model_name}' on {DEVICE}...")
        try:
            processor = BlipProcessor.from_pretrained(model_name)
            model = BlipForConditionalGeneration.from_pretrained(model_name).to(DEVICE)
            _loaded_models[cache_key_model] = model
            _loaded_models[cache_key_processor] = processor
            logger.info(f"ModelLoader: Image captioning model '{model_name}' loaded and cached successfully.")
        except Exception as e:
            logger.exception(f"ModelLoader: Failed to load image captioning model '{model_name}'.")
            raise RuntimeError(f"Could not load image captioning model '{model_name}': {e}") from e
    else:
        logger.debug(f"ModelLoader: Using cached image captioning model '{model_name}'.")
        model = _loaded_models[cache_key_model]
        processor = _loaded_models[cache_key_processor]
    
    return model, processor


def get_classifier_model(collection_id: int):
    """
    Loads the classifier for a given collection_id. It first attempts to use the
    model specified in 'trained_classifiers/best_models.json'. If not specified
    or if the file is not found, it falls back to loading the latest dated model.
    Caches the model for subsequent calls.
    """
    cache_key = f"classifier_{collection_id}"
    if cache_key not in _loaded_models:
        _load_best_models_map()  # Ensure the map is loaded

        classifier_dir = "trained_classifiers"
        if not os.path.isdir(classifier_dir):
            raise FileNotFoundError(f"Classifier directory not found: '{classifier_dir}'")

        latest_file = None
        
        # 1. Try to get model from the best_models config
        best_model_filename = _best_models_map.get(str(collection_id))
        if best_model_filename:
            candidate_path = os.path.join(classifier_dir, best_model_filename)
            if os.path.exists(candidate_path):
                latest_file = candidate_path
                logger.info(f"ModelLoader: Using best model '{os.path.basename(latest_file)}' for collection {collection_id} from config.")
            else:
                logger.warning(f"ModelLoader: Best model '{best_model_filename}' for collection {collection_id} not found in filesystem. Falling back to date-based search.")

        # 2. If not found via config, fall back to searching for the latest dated file
        if not latest_file:
            logger.info(f"ModelLoader: No configured best model for collection {collection_id}. Searching by date...")
            file_pattern = os.path.join(classifier_dir, f"collection_{collection_id}_classifier_*.pkl")
            model_files = glob.glob(file_pattern)

            if not model_files:
                raise FileNotFoundError(f"No classifier model found for collection ID {collection_id}")

            # Find the file with the most recent date in the filename (YYYY-MM-DD or YYYY-MM-DD_HHMMSS).
            latest_date_str = ""
            date_pattern = re.compile(r"_(\d{4}-\d{2}-\d{2}(?:_\d{6})?)\.pkl$")

            for f in model_files:
                match = date_pattern.search(f)
                if match:
                    date_str = match.group(1)
                    if date_str > latest_date_str:
                        latest_date_str = date_str
                        latest_file = f
            
            if not latest_file:
                logger.warning(f"No classifier file for collection {collection_id} matched naming pattern. Falling back to most recently modified file.")
                latest_file = max(model_files, key=os.path.getmtime)

        if not latest_file:
             raise FileNotFoundError(f"Could not determine a classifier to load for collection ID {collection_id}")

        logger.info(f"ModelLoader: Loading classifier '{os.path.basename(latest_file)}'...")
        try:
            with open(latest_file, "rb") as f:
                model = joblib.load(f)
            _loaded_models[cache_key] = model
            logger.info(f"ModelLoader: Classifier for collection {collection_id} loaded and cached successfully.")
        except Exception as e:
            logger.exception(f"ModelLoader: Failed to load classifier model from '{latest_file}'.")
            raise RuntimeError(f"Could not load classifier model from '{latest_file}': {e}") from e

    else:
        logger.debug(f"ModelLoader: Using cached classifier model for collection ID {collection_id}.")
    
    return _loaded_models[cache_key]


def preload_all_models(clip_model_name: str):
    """
    Pre-loads all models into the cache at startup.
    This is called from the main application's lifespan event.
    """
    logger.info("--- Starting Model Pre-loading ---")
    get_clip_model_and_preprocess(clip_model_name)
    get_person_detection_model()
    get_face_detection_model()
    get_image_captioning_model_and_processor()
    logger.info("--- Model Pre-loading Complete ---")
