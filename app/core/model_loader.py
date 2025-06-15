import logging
import torch
import clip
import torchvision
from facenet_pytorch import MTCNN

logger = logging.getLogger(__name__)

# --- Global Cache for Models ---
_loaded_models = {}

# --- Device Configuration ---
try:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"ModelLoader: Using device: {DEVICE}")
except Exception as e:
    logger.error(f"ModelLoader: Error determining Torch device, defaulting to CPU. Error: {e}")
    DEVICE = "cpu"


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

def get_threedmm_model():
    """
    Placeholder for 3DMM model loading.
    """
    cache_key = "threedmm_model"
    if cache_key not in _loaded_models:
        logger.warning("ModelLoader: 3DMM model not implemented. Returning None.")
        _loaded_models[cache_key] = None # Cache the fact that it's None
    return _loaded_models[cache_key]

def preload_all_models(clip_model_name: str):
    """
    Pre-loads all models into the cache at startup.
    This is called from the main application's lifespan event.
    """
    logger.info("--- Starting Model Pre-loading ---")
    get_clip_model_and_preprocess(clip_model_name)
    get_person_detection_model()
    get_face_detection_model() # Mock, will log warning
    get_threedmm_model() # Mock, will log warning
    logger.info("--- Model Pre-loading Complete ---")
