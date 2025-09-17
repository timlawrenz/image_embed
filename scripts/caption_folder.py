import os
import requests
import json
import logging
import argparse
import glob
from typing import List, Dict, Any

# --- Configuration ---
LOGGING_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT)

# Assuming the service is running locally on the default port
API_ENDPOINT = "http://localhost:8000/analyze_image_upload/"
SUPPORTED_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']

# --- Helper Functions ---

def get_image_paths(folder_path: str) -> List[str]:
    """Finds all image files in a given folder."""
    image_paths = []
    for extension in SUPPORTED_EXTENSIONS:
        pattern = os.path.join(folder_path, f"*{extension}")
        image_paths.extend(glob.glob(pattern))
    logging.info(f"Found {len(image_paths)} images in '{folder_path}'.")
    return image_paths

def get_description_for_image(image_path: str, max_length: int) -> str | None:
    """
    Sends an image to the API and returns the generated description.
    """
    tasks_json = json.dumps([{
        "operation_id": "caption",
        "type": "describe_image",
        "params": {
            "target": "whole_image",
            "max_length": max_length
        }
    }])

    try:
        with open(image_path, 'rb') as f:
            files = {
                'image_file': (os.path.basename(image_path), f, 'image/jpeg'),
                'tasks_json': (None, tasks_json)
            }
            response = requests.post(API_ENDPOINT, files=files, timeout=30)
            response.raise_for_status()

        response_data = response.json()
        result = response_data.get("results", {}).get("caption", {})

        if result.get("status") == "success":
            return result.get("data")
        else:
            error_msg = result.get("error_message", "No description returned.")
            logging.error(f"API failed to describe '{os.path.basename(image_path)}': {error_msg}")
            return None

    except requests.exceptions.RequestException as e:
        logging.error(f"API request failed for '{os.path.basename(image_path)}': {e}")
        return None
    except json.JSONDecodeError:
        logging.error(f"Failed to decode JSON response for '{os.path.basename(image_path)}'.")
        return None

def main(folder_path: str, force_overwrite: bool, max_length: int):
    """
    Main function to iterate through images in a folder, get descriptions,
    and save them to text files.
    """
    logging.info(f"Starting to caption images in folder: {folder_path}")

    if not os.path.isdir(folder_path):
        logging.error(f"Error: The provided path '{folder_path}' is not a valid directory.")
        return

    image_paths = get_image_paths(folder_path)
    if not image_paths:
        logging.warning("No images found in the specified folder. Exiting.")
        return

    for image_path in image_paths:
        base_filename, _ = os.path.splitext(image_path)
        caption_path = f"{base_filename}.txt"
        image_filename = os.path.basename(image_path)

        if os.path.exists(caption_path) and not force_overwrite:
            logging.info(f"Caption file already exists for '{image_filename}'. Skipping.")
            continue

        logging.info(f"Requesting description for '{image_filename}'...")
        description = get_description_for_image(image_path, max_length)

        if description:
            try:
                with open(caption_path, 'w', encoding='utf-8') as f:
                    f.write(description)
                logging.info(f"Successfully saved caption to '{os.path.basename(caption_path)}'.")
            except IOError as e:
                logging.error(f"Failed to write caption file for '{image_filename}': {e}")

    logging.info("Image captioning process finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Caption all images in a folder using the image analysis API.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "folder_path",
        type=str,
        help="The absolute or relative path to the folder containing images to caption."
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="If set, overwrite existing .txt caption files. Default is to skip them."
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=50,
        help="The maximum length of the generated description in tokens. Default is 50."
    )

    args = parser.parse_args()
    main(args.folder_path, args.force, args.max_length)
