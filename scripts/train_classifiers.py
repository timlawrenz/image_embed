import os
import requests
import joblib
import json
import logging
import re
import glob
import pickle
from datetime import datetime, timezone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# --- Configuration ---
LOGGING_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT)

BASE_URL = "https://crawlr.lawrenz.com"
COLLECTIONS_ENDPOINT = f"{BASE_URL}/collections.json"
TRAINING_DATA_ENDPOINT_TEMPLATE = f"{BASE_URL}/collections/{{collection_id}}/training_data.json"
CLASSIFIER_DIR = "trained_classifiers"
MAX_MODELS_PER_COLLECTION = 10

# --- Helper Functions ---

def clean_json_response(text: str) -> str:
    """
    Cleans the response text to extract the JSON part.
    The endpoint sometimes wraps JSON in HTML.
    """
    # Use regex to find a string that looks like a JSON array or object
    match = re.search(r'(\[.*\]|\{.*\})', text, re.DOTALL)
    if match:
        return match.group(1)
    logging.warning("Could not find JSON in response text.")
    # Return original text; json.loads() will then fail and log the problematic text.
    return text

def find_existing_models(collection_id: int):
    """Finds all classifier files for a given collection ID."""
    model_pattern = os.path.join(CLASSIFIER_DIR, f"collection_{collection_id}_classifier_*.pkl")
    logging.info(f"Searching for existing models with pattern: {model_pattern}")
    found_models = glob.glob(model_pattern)
    logging.info(f"Found {len(found_models)} existing models for collection {collection_id}.")
    return found_models


def fetch_collections():
    """Fetches the list of available collections with their focus metadata."""
    logging.info(f"Fetching collections from {COLLECTIONS_ENDPOINT}...")
    try:
        response = requests.get(COLLECTIONS_ENDPOINT)
        response.raise_for_status()
        cleaned_text = clean_json_response(response.text)
        collections = json.loads(cleaned_text)
        logging.info(f"Found {len(collections)} collections.")
        
        # Extract and validate collection_focus metadata
        for collection in collections:
            focus = collection.get('collection_focus')
            if focus:
                collection['derivative_type'] = focus.get('derivative_type_name')
                collection['embedding_type'] = focus.get('embedding_type_name')
                logging.debug(f"Collection {collection.get('id')}: derivative_type={collection['derivative_type']}, embedding_type={collection['embedding_type']}")
            else:
                collection['derivative_type'] = None
                collection['embedding_type'] = None
                logging.warning(f"Collection {collection.get('id')} '{collection.get('name')}' has no focus defined.")
        
        return collections
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching collections: {e}")
        return None
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from collections response: {e}")
        logging.debug(f"Response text was: {response.text}")
        return None

def fetch_training_data(collection_id: int):
    """Fetches training data for a specific collection."""
    url = TRAINING_DATA_ENDPOINT_TEMPLATE.format(collection_id=collection_id)
    logging.info(f"Fetching training data for collection ID {collection_id} from {url}...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        cleaned_text = clean_json_response(response.text)
        training_data = json.loads(cleaned_text)
        return training_data
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching training data for collection {collection_id}: {e}")
        return None
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON for training data (collection {collection_id}): {e}")
        logging.debug(f"Response text was: {response.text}")
        return None

def train_and_save_model(collection_id: int, collection_name: str, training_data: list, derivative_type: str, embedding_type: str):
    """
    Trains a new classifier, saves it, and then evaluates all available versions
    for the collection (including the new one) to identify the best-performing one.
    
    Args:
        collection_id: The ID of the collection
        collection_name: The name of the collection
        training_data: List of dicts with 'embedding' and 'label' keys
        derivative_type: The derivative type (e.g., 'whole_image', 'prominent_person', 'prominent_face')
        embedding_type: The embedding type (e.g., 'embed_clip_vit_b_32', 'embed_dino_v2')
    
    Returns:
        Dict with metadata about the best model, or None if training failed
    """
    if not training_data:
        logging.warning(f"No training data for collection {collection_name} (ID: {collection_id}). Skipping.")
        return None

    logging.info(f"Processing collection: '{collection_name}' (ID: {collection_id})")
    logging.info(f"  Focus: {embedding_type} on {derivative_type}")

    try:
        X = [item['embedding'] for item in training_data]
        y = [item['label'] for item in training_data]
    except KeyError as e:
        logging.error(f"Training data for collection {collection_id} is malformed. Missing key: {e}")
        return None

    # A model cannot be trained if all data belongs to a single class.
    if len(set(y)) < 2:
        logging.warning(f"Collection {collection_id} has only one class ({set(y)}). Cannot train a model. Skipping.")
        return None

    # Split data into a training set and a held-out test set for the bake-off.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    logging.info(f"Training new model for '{collection_name}'. Train set size: {len(X_train)}, Test set size: {len(X_test)}")

    # 'class_weight="balanced"' is key for imbalanced collections.
    model = LogisticRegression(class_weight="balanced", C=0.1, solver='liblinear', random_state=42)
    model.fit(X_train, y_train)

    # Save the newly trained model with the current UTC date and time.
    date_str = datetime.now(timezone.utc).strftime('%Y-%m-%d_%H%M%S')
    model_filename = f"collection_{collection_id}_classifier_{date_str}.pkl"
    model_filepath = os.path.join(CLASSIFIER_DIR, model_filename)

    logging.info(f"Saving new model to {model_filepath}")
    joblib.dump(model, model_filepath)
    logging.info(f"Successfully saved new model for '{collection_name}'.")

    # --- Classifier Bake-off ---
    logging.info(f"--- Starting Bake-off for Collection '{collection_name}' ---")
    logging.info("Evaluating all model versions on the latest test data.")

    model_files = find_existing_models(collection_id)
    if not model_files:
        logging.error("Could not find any models for bake-off, not even the one just saved. Check permissions or path.")
        return

    results = []
    for mf_path in model_files:
        try:
            loaded_model = joblib.load(mf_path)
            y_pred = loaded_model.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

            # Using macro average precision as the key metric.
            precision = report.get('macro avg', {}).get('precision', 0)

            results.append({
                'model_file': os.path.basename(mf_path),
                'macro_precision': precision
            })
        except Exception as e:
            logging.warning(f"Could not evaluate model {os.path.basename(mf_path)}: {e}")
            continue

    if not results:
        logging.error("Bake-off failed: no models could be evaluated.")
        return None

    # Sort results by precision, descending.
    results.sort(key=lambda x: x['macro_precision'], reverse=True)

    logging.info(f"--- Bake-off Report for '{collection_name}' (ID: {collection_id}) ---")
    for res in results:
        logging.info(f"  - Model: {res['model_file']:<50} Macro Avg Precision: {res['macro_precision']:.4f}")

    # --- Pruning old classifiers ---
    if len(results) > MAX_MODELS_PER_COLLECTION:
        logging.info(f"Pruning old, low-performing classifiers. Keeping top {MAX_MODELS_PER_COLLECTION}.")
        models_to_prune = results[MAX_MODELS_PER_COLLECTION:]
        for model_info in models_to_prune:
            model_filename = model_info['model_file']
            try:
                model_path = os.path.join(CLASSIFIER_DIR, model_filename)
                os.remove(model_path)
                logging.info(f"  - Removed old model: {model_filename}")
            except OSError as e:
                logging.error(f"Error removing model file {model_filename}: {e}")

    best_model_info = results[0]
    best_model_filename = best_model_info['model_file']
    logging.info(f"--- Best model is '{best_model_filename}' with a precision of {best_model_info['macro_precision']:.4f} ---")

    # Save the best model to a consistently named file for compatibility.
    compatible_filename = f"collection_{collection_id}_compatible_classifier.pkl"
    compatible_filepath = os.path.join(CLASSIFIER_DIR, compatible_filename)
    
    try:
        best_model_path = os.path.join(CLASSIFIER_DIR, best_model_filename)
        best_model_object = joblib.load(best_model_path)
        
        logging.info(f"Saving best model to '{compatible_filepath}' with pickle protocol 4.")
        with open(compatible_filepath, 'wb') as f:
            pickle.dump(best_model_object, f, protocol=4)
        logging.info("Successfully saved compatible model file.")
    except Exception as e:
        logging.error(f"Could not save compatible model file '{compatible_filename}': {e}")

    # Infer dimensionality from training data
    dimensionality = len(X[0]) if X else None
    
    # Return metadata about the best model
    return {
        'model_file': best_model_filename,
        'derivative_type': derivative_type,
        'embedding_type': embedding_type,
        'dimensionality': dimensionality
    }

def main():
    """Main function to run the training pipeline."""
    logging.info("Starting classifier training process...")

    # Ensure the directory for classifiers exists.
    os.makedirs(CLASSIFIER_DIR, exist_ok=True)
    logging.info(f"Models will be saved in '{CLASSIFIER_DIR}/' directory.")

    collections = fetch_collections()
    if not collections:
        logging.error("Could not retrieve collections. Exiting.")
        return

    best_models_map = {}
    for collection in collections:
        collection_id = collection.get('id')
        collection_name = collection.get('name')
        derivative_type = collection.get('derivative_type')
        embedding_type = collection.get('embedding_type')

        if not isinstance(collection_id, int) or not isinstance(collection_name, str):
            logging.warning(f"Skipping invalid collection entry: {collection}")
            continue
        
        # Skip collections without focus metadata
        if not derivative_type or not embedding_type:
            logging.warning(f"Skipping collection '{collection_name}' (ID: {collection_id}) - missing focus metadata.")
            continue

        training_data = fetch_training_data(collection_id)
        if training_data:
            model_metadata = train_and_save_model(
                collection_id, 
                collection_name, 
                training_data,
                derivative_type,
                embedding_type
            )
            if model_metadata:
                best_models_map[str(collection_id)] = model_metadata
        else:
            logging.warning(f"Could not retrieve or process training data for '{collection_name}' (ID: {collection_id}).")

    if best_models_map:
        config_path = os.path.join(CLASSIFIER_DIR, 'best_models.json')
        logging.info(f"Saving best models configuration to {config_path}")
        try:
            with open(config_path, 'w') as f:
                json.dump(best_models_map, f, indent=4)
            logging.info("Successfully saved best models configuration.")
        except IOError as e:
            logging.error(f"Could not write best models configuration file: {e}")

    logging.info("Classifier training process finished.")

if __name__ == '__main__':
    main()
