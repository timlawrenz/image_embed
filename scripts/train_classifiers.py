import os
import requests
import joblib
import json
import logging
import re
from datetime import datetime
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

def fetch_collections():
    """Fetches the list of available collections."""
    logging.info(f"Fetching collections from {COLLECTIONS_ENDPOINT}...")
    try:
        response = requests.get(COLLECTIONS_ENDPOINT)
        response.raise_for_status()
        cleaned_text = clean_json_response(response.text)
        collections = json.loads(cleaned_text)
        logging.info(f"Found {len(collections)} collections.")
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

def train_and_save_model(collection_id: int, collection_name: str, training_data: list):
    """Trains a classifier, evaluates it, and saves it to a file."""
    if not training_data:
        logging.warning(f"No training data for collection {collection_name} (ID: {collection_id}). Skipping.")
        return

    logging.info(f"Processing collection: '{collection_name}' (ID: {collection_id})")

    try:
        X = [item['embedding'] for item in training_data]
        y = [item['label'] for item in training_data]
    except KeyError as e:
        logging.error(f"Training data for collection {collection_id} is malformed. Missing key: {e}")
        return

    # A model cannot be trained if all data belongs to a single class.
    if len(set(y)) < 2:
        logging.warning(f"Collection {collection_id} has only one class ({set(y)}). Cannot train a model. Skipping.")
        return

    # Split data for evaluation. Stratify is important for imbalanced data.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    logging.info(f"Training model for '{collection_name}'. Train set size: {len(X_train)}, Test set size: {len(X_test)}")

    # 'class_weight="balanced"' is key for imbalanced collections.
    model = LogisticRegression(class_weight="balanced", C=0.1, solver='liblinear', random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model before saving.
    y_pred = model.predict(X_test)
    logging.info(f"--- Evaluation Report for Collection '{collection_name}' (ID: {collection_id}) ---")
    # Use print() for the report as it's more readable than logging it line-by-line.
    print(classification_report(y_test, y_pred, zero_division=0))
    logging.info("--- End of Report ---")

    # Save the trained model with the current UTC date.
    date_str = datetime.utcnow().strftime('%Y-%m-%d')
    model_filename = f"collection_{collection_id}_classifier_{date_str}.pkl"
    model_filepath = os.path.join(CLASSIFIER_DIR, model_filename)

    logging.info(f"Saving model to {model_filepath}")
    joblib.dump(model, model_filepath)
    logging.info(f"Successfully saved model for '{collection_name}'.")

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

    for collection in collections:
        collection_id = collection.get('id')
        collection_name = collection.get('name')

        if not isinstance(collection_id, int) or not isinstance(collection_name, str):
            logging.warning(f"Skipping invalid collection entry: {collection}")
            continue

        training_data = fetch_training_data(collection_id)
        if training_data:
            train_and_save_model(collection_id, collection_name, training_data)
        else:
            logging.warning(f"Could not retrieve or process training data for '{collection_name}' (ID: {collection_id}).")

    logging.info("Classifier training process finished.")

if __name__ == '__main__':
    main()
