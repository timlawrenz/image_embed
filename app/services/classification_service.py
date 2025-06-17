import logging
from typing import Any, Dict, List

import numpy as np
from app.core import model_loader

logger = logging.getLogger(__name__)


def classify_embedding(embedding: List[float], collection_id: int) -> Dict[str, Any]:
    """
    Classifies an embedding using the model for the given collection_id.

    Args:
        embedding: The embedding vector to classify.
        collection_id: The ID of the classifier collection to use.

    Returns:
        A dictionary containing the predicted label and a dictionary of class probabilities.
        Example: {'label': 'cat', 'probabilities': {'cat': 0.9, 'dog': 0.1}}
    
    Raises:
        FileNotFoundError: If no classifier model is found for the collection_id.
        RuntimeError: For any other errors during classification.
    """
    logger.info(f"Attempting classification for collection_id {collection_id}...")
    try:
        model = model_loader.get_classifier_model(collection_id)
        
        # Models from scikit-learn expect a 2D array, so reshape the single embedding.
        embedding_array = np.array(embedding).reshape(1, -1)
        
        label = model.predict(embedding_array)[0]
        probabilities = model.predict_proba(embedding_array)[0]
        
        # Create a dictionary of class_name -> probability
        class_probabilities = {str(model.classes_[i]): prob for i, prob in enumerate(probabilities)}
        
        result = {
            "label": str(label),
            "probabilities": class_probabilities
        }
        logger.info(f"Classification successful for collection_id {collection_id}. Label: '{label}'.")
        return result
    except FileNotFoundError:
        logger.warning(f"Classification failed: No model found for collection_id {collection_id}.")
        raise # Re-raise to be handled by the endpoint
    except Exception as e:
        logger.exception(f"An unexpected error occurred during classification for collection_id {collection_id}.")
        raise RuntimeError(f"Failed to classify embedding for collection {collection_id}: {e}") from e
