import logging
from typing import Any, Dict, List
import time

import numpy as np
from PIL import Image
from app.core import model_loader

logger = logging.getLogger(__name__)


def classify_embedding(embedding: List[float], collection_id: int) -> Dict[str, Any]:
    """
    Classifies an embedding to determine if it belongs to a given collection.
    This uses a binary classifier trained for the specified collection_id where the
    label is True if the item is in the collection and False otherwise.

    Args:
        embedding: The embedding vector to classify.
        collection_id: The ID of the classifier collection to use.

    Returns:
        A dictionary indicating if the item is in the collection and the probability.
        Example: {'is_in_collection': True, 'probability': 0.95}
    
    Raises:
        FileNotFoundError: If no classifier model is found for the collection_id.
        RuntimeError: For any other errors during classification.
    """
    logger.info(f"Attempting classification for collection_id {collection_id}...")
    try:
        model = model_loader.get_classifier_model(collection_id)
        
        # Models from scikit-learn expect a 2D array, so reshape the single embedding.
        embedding_array = np.array(embedding).reshape(1, -1)
        
        # model.predict() returns a boolean array e.g., [True]
        prediction = bool(model.predict(embedding_array)[0])
        
        # model.predict_proba() returns a 2D array e.g., [[prob_false, prob_true]]
        probabilities = model.predict_proba(embedding_array)[0]
        
        # Find the index of the 'True' class in model.classes_ (which is usually [False, True])
        # and get the corresponding probability.
        try:
            true_class_index = list(model.classes_).index(True)
            probability = probabilities[true_class_index]
        except ValueError:
            # This case should be prevented by the training script which ensures at least
            # two classes (True and False) are present.
            logger.error(f"Classifier for collection {collection_id} is malformed: does not have a 'True' class.")
            raise RuntimeError(f"Classifier for collection {collection_id} is malformed.")

        result = {
            "is_in_collection": prediction,
            "probability": probability
        }
        logger.info(f"Classification successful for collection_id {collection_id}. Result: {result}")
        return result
    except FileNotFoundError:
        logger.warning(f"Classification failed: No model found for collection_id {collection_id}.")
        raise # Re-raise to be handled by the endpoint
    except Exception as e:
        logger.exception(f"An unexpected error occurred during classification for collection_id {collection_id}.")
        raise RuntimeError(f"Failed to classify embedding for collection {collection_id}: {e}") from e


def classify_embedding_from_image(
    pil_image: Image.Image, 
    collection_id: int,
    shared_context: Dict[str, Any],
    timing_stats: Dict[str, float]
) -> Dict[str, Any]:
    """
    Classifies an image for a collection by generating the correct embedding type.
    
    This function:
    1. Loads classifier metadata to determine which embedding type to use
    2. Performs detection if needed based on derivative_type
    3. Generates the correct embedding (CLIP or DINO)
    4. Classifies the embedding
    
    Args:
        pil_image: PIL Image in RGB format
        collection_id: ID of the collection classifier
        shared_context: Shared detection/embedding cache
        timing_stats: Performance timing dictionary
    
    Returns:
        Dict with keys: is_in_collection (bool), probability (float)
    
    Raises:
        FileNotFoundError: If no classifier or metadata found
        ValueError: If derivative_type or embedding_type is invalid
    """
    from app.services.detection_service import get_prominent_person_bbox, get_prominent_face_bbox_in_region
    from app.services.embedding_service import get_clip_embedding, get_dino_embedding, get_dino_v3_embedding
    
    # Get classifier metadata to determine embedding type and target
    metadata = model_loader.get_classifier_metadata(collection_id)
    embedding_type = metadata['embedding_type']
    derivative_type = metadata['derivative_type']
    
    logger.info(f"Classifying for collection {collection_id} using {embedding_type} on {derivative_type}")
    
    # Map derivative_type to target
    target_map = {
        'whole_image': 'whole_image',
        'prominent_person': 'prominent_person',
        'prominent_face': 'prominent_face'
    }
    target = target_map.get(derivative_type)
    
    if not target:
        raise ValueError(f"Invalid derivative_type '{derivative_type}' for collection {collection_id}")
    
    # Determine crop_box based on target
    crop_box = None
    if target == 'prominent_person':
        if 'prominent_person_bbox' not in shared_context:
            detection_start = time.time()
            shared_context['prominent_person_bbox'] = get_prominent_person_bbox(pil_image)
            timing_stats['detection'] += time.time() - detection_start
        crop_box = shared_context.get('prominent_person_bbox')
        if not crop_box:
            logger.warning(f"No person detected for collection {collection_id}, using whole image")
    
    elif target == 'prominent_face':
        if 'prominent_face_bbox' not in shared_context:
            detection_start = time.time()
            if 'prominent_person_bbox' not in shared_context:
                shared_context['prominent_person_bbox'] = get_prominent_person_bbox(pil_image)
            person_bbox = shared_context.get('prominent_person_bbox')
            shared_context['prominent_face_bbox'] = get_prominent_face_bbox_in_region(pil_image, person_bbox)
            timing_stats['detection'] += time.time() - detection_start
        crop_box = shared_context.get('prominent_face_bbox')
        
        if not crop_box:
            raise ValueError(f"No face detected for collection {collection_id} with face target")
    
    # Generate correct embedding type with caching
    embedding_cache_key = f"embedding_{embedding_type}_{target}"
    
    if embedding_cache_key not in shared_context:
        embedding_start = time.time()
        
        if embedding_type == 'embed_clip_vit_b_32':
            from main import MODEL_NAME_CLIP
            embedding_list, _, _ = get_clip_embedding(pil_image, MODEL_NAME_CLIP, crop_box)
        elif embedding_type == 'embed_dino_v2':
            # Use default DINO model name
            embedding_list, _, _ = get_dino_embedding(pil_image, crop_bbox=crop_box)
        elif embedding_type == 'embed_dino_v3':
            embedding_list, _, _ = get_dino_v3_embedding(pil_image, crop_bbox=crop_box)
        else:
            raise ValueError(f"Unsupported embedding_type '{embedding_type}' for collection {collection_id}")
        
        timing_stats['embedding'] += time.time() - embedding_start
        shared_context[embedding_cache_key] = embedding_list
        logger.debug(f"Generated and cached {embedding_type} embedding for {target}")
    else:
        embedding_list = shared_context[embedding_cache_key]
        logger.debug(f"Using cached {embedding_type} embedding for {target}")
    
    # Classify using the existing function
    return classify_embedding(embedding_list, collection_id)
