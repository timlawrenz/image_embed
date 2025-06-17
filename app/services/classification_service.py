import logging
from typing import Any, Dict, List

import numpy as np
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
