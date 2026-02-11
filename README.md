# Advanced Image Analysis API

[ ![CircleCI](https://dl.circleci.com/status-badge/img/gh/timlawrenz/image_embed/tree/main.svg?style=svg) ](https://dl.circleci.com/status-badge/redirect/gh/timlawrenz/image_embed/tree/main)

An advanced FastAPI service that performs various analyses on images. You provide an image—either via a URL or by direct file upload—and a list of analysis tasks, and the API returns the results for each task.

## Features

*   Accepts an image via URL or direct upload, along with a list of analysis tasks.
*   Retrieves the image once per request (by downloading from URL or processing an upload).
*   Converts the image to RGB format if necessary.
*   Performs requested operations, which can include:
    *   **Human Detection:** Detects the most prominent person using Faster R-CNN.
    *   **Face Detection:** Detects the most prominent face using MTCNN.
    *   **CLIP Embedding:** Generates semantic image embeddings using a specified CLIP model (default: "ViT-B/32") on the whole image, a detected person, or a detected face.
    *   **DINOv2 Embedding:** Generates visual feature embeddings optimized for similarity search based on composition, color, and texture.
    *   **DINOv3 Embedding:** Generates modern visual embeddings via Hugging Face (default checkpoint: `facebook/dinov3-vitl16-pretrain-lvd1689m`).
    *   **Image Classification:** Uses trained binary classifiers to determine if an image belongs to specific collections.
    *   **Image Captioning:** Generates natural language descriptions of images using pre-trained captioning models.
*   For operations involving cropping (e.g., embedding a detected face), the API returns:
    *   The primary result of the operation (e.g., embedding vector).
    *   The bounding box coordinates used for the crop.
    *   A base64 encoded PNG string of the actual cropped image.
*   Returns the original image URL and a dictionary of results, keyed by a user-provided `operation_id` for each task.
*   Efficiently reuses detected bounding boxes for subsequent tasks within the same request.
*   Basic error handling for image download, processing, and individual task execution.
*   An endpoint to discover available operations at runtime.

## API Endpoints

### `POST /analyze_image/`

Performs a series of analyses on an image from a given URL based on a list of tasks.

**Request Body:**

```json
{
  "image_url": "YOUR_IMAGE_URL_HERE",
  "tasks": [
    {
      "operation_id": "unique_task_id_1",
      "type": "embed_clip_vit_b_32",
      "params": {"target": "whole_image"}
    },
    {
      "operation_id": "unique_task_id_2",
      "type": "detect_bounding_box",
      "params": {"target": "prominent_person"}
    }
    // ... more tasks
  ]
}
```

*   `image_url` (string, required): A valid HTTP or HTTPS URL pointing to an image.
*   `tasks` (array of objects, required): A list of analysis tasks to perform. Each task object contains:
    *   `operation_id` (string, required): A unique identifier for this task, which will be used as a key in the response's `results` dictionary.
    *   `type` (string, required): The type of operation to perform (see "Available Operations" below).
    *   `params` (object, optional): Parameters specific to the operation type. Common parameters include:
        *   `target` (string): Specifies the region of the image to operate on (e.g., "whole\_image", "prominent\_person", "prominent\_face"). Defaults vary by operation type.
        *   `face_context` (string): For face-related operations, specifies whether to search for a face within a "prominent\_person" bounding box or the "whole\_image". Defaults to "prominent\_person".

**Response Body (Success - 200 OK):**

```json
{
  "image_url": "THE_IMAGE_URL_PROVIDED",
  "results": {
    "unique_task_id_1": {
      "status": "success", // or "error", "skipped"
      "data": [0.123, ..., 0.789], // e.g., embedding vector
      "cropped_image_bbox": null, // null if not applicable
      "cropped_image_base64": null, // null if not applicable
      "error_message": null
    },
    "unique_task_id_2": {
      "status": "success",
      "data": [100, 150, 250, 350], // e.g., bounding box [xmin, ymin, xmax, ymax]
      "cropped_image_bbox": null, // Bbox detection itself doesn't return a *cropped image* of the bbox
      "cropped_image_base64": null,
      "error_message": null
    },
    "task_embedding_cropped_face": {
        "status": "success",
        "data": [0.321, ..., 0.987], // Embedding of the cropped face
        "cropped_image_bbox": [50, 60, 150, 180], // Bbox of the face used for this embedding
        "cropped_image_base64": "iVBORw0KGgoAAAANSUhEUgAAAGQAAABkCAYAAABw4pVU...", // Base64 PNG of the cropped face
        "error_message": null
    }
    // ... results for other tasks
  }
}
```

*   `image_url` (string): The URL of the image analyzed.
*   `results` (object): A dictionary where each key is an `operation_id` from the request, and the value is an `OperationResult` object:
    *   `status` (string): "success", "error", or "skipped".
    *   `data` (any): The primary result of the operation (e.g., embedding vector, bounding box coordinates).
    *   `cropped_image_bbox` (array of int, optional): If the operation involved cropping (e.g., embedding a face), this is the `[xmin, ymin, xmax, ymax]` bounding box used for the crop.
    *   `cropped_image_base64` (string, optional): If the operation involved cropping, this is a base64 encoded PNG string of the cropped image.
    *   `error_message` (string, optional): Details if the status is "error" or "skipped".

**Error Responses:**

*   **400 Bad Request:** If the image URL is invalid, the image cannot be downloaded/processed, or the request structure is invalid.
    ```json
    { "detail": "Could not download image from URL: <error_details>" }
    ```
    ```json
    { "detail": "Could not process image: <error_details>" }
    ```
*   Individual task errors/skips are reported within the `results` dictionary for each `operation_id` (see `status` and `error_message` fields above).

### `POST /analyze_image_upload/`

Performs a series of analyses on a directly uploaded image file. This endpoint uses a `multipart/form-data` request.

**Request Body:**

*   `image_file` (file, required): The image file to be analyzed.
*   `tasks_json` (string, form-data, required): A JSON string representing the list of analysis tasks. The structure of this JSON string should be identical to the `tasks` array in the request for the `/analyze_image/` endpoint.

**Example `tasks_json` value:**
```json
[
  {
    "operation_id": "face_bbox_from_upload",
    "type": "detect_bounding_box",
    "params": {"target": "prominent_face"}
  }
]
```

**Response Body (Success - 200 OK):**

The response structure is identical to the `/analyze_image/` endpoint. The `image_url` field in the response will contain a placeholder string like `uploaded:your_filename.jpg`.

**Error Responses:**

*   **400 Bad Request:** If the `tasks_json` is malformed, the uploaded file is not a valid image, or other request errors occur.

### `GET /available_operations/`

Provides a list of available analysis operations that can be used in the `tasks` array for the `/analyze_image` endpoint. It details each operation's allowed targets and default target.

**Response Body (Success - 200 OK):**

```json
{
  "operations": {
    "detect_bounding_box": {
      "description": "Detects a bounding box for a specified target.",
      "allowed_targets": [
        "prominent_person",
        "prominent_face"
      ],
      "default_target": "prominent_person"
    },
    "embed_clip_vit_b_32": {
      "description": "Generates an embedding using the CLIP ViT-B/32 model.",
      "allowed_targets": [
        "whole_image",
        "prominent_person",
        "prominent_face"
      ],
      "default_target": "whole_image"
    },
    "classify": {
      "description": "Determines if an image region belongs to a specific collection using a binary classifier.",
      "allowed_targets": [
        "whole_image",
        "prominent_person",
        "prominent_face"
      ],
      "default_target": "whole_image"
    }
  }
}
```

## Setup and Installation

### Requirements

*   Python 3.12
*   GPU acceleration (recommended):
    *   **NVIDIA:** CUDA
    *   **AMD (ROCm/HIP):** ROCm (e.g. Strix Halo / gfx1151)
    *   CPU fallback is supported

On AMD, install a recent ROCm stack (e.g. ROCm 7.1.1) and verify the GPU is visible before installing Python deps:
```bash
rocminfo | head
```
(You should see the GPU agent like `gfx1151` and `ROCk module is loaded`.)

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/timlawrenz/image_embed.git
    cd image_embed
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**

    **AMD ROCm (Strix Halo / gfx1151):**
    ```bash
    # installs ROCm-enabled torch/torchvision/torchaudio + the rest of deps
    pip install -r requirements-rocm.txt

    # quick sanity check (expects a non-zero device count)
    python -c "import torch; print('torch', torch.__version__, 'hip', torch.version.hip); print('device_count', torch.cuda.device_count()); print('device0', torch.cuda.get_device_name(0) if torch.cuda.device_count() else None)"
    ```

    **Default (CPU-only PyTorch):**
    ```bash
    pip install -r requirements.txt
    ```

4.  **(Optional) Train classifiers:**
    
    If you want to use the classification feature, train the binary classifiers:
    ```bash
    python scripts/train_classifiers.py
    ```
    
    This will download training data and create models in the `trained_classifiers/` directory.

## Running the Service

To run the FastAPI application locally for development:

```bash
uvicorn main:app --reload
```

The application will typically be available at `http://localhost:8000`. You can access the auto-generated API documentation at `http://localhost:8000/docs`.

## Usage Example

You can use `curl` or any API client (like Postman or Insomnia) to send requests.

**Using `curl`:**

The command below should be run as a single line.

```bash
curl -X POST "http://localhost:8000/analyze_image/" -H "Content-Type: application/json" -d '{"image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e9/Official_portrait_of_Barack_Obama.jpg/800px-Official_portrait_of_Barack_Obama.jpg", "tasks": [{"operation_id": "whole_image_embedding", "type": "embed_clip_vit_b_32", "params": {"target": "whole_image"}}, {"operation_id": "person_bbox", "type": "detect_bounding_box", "params": {"target": "prominent_person"}}, {"operation_id": "face_bbox_from_person", "type": "detect_bounding_box", "params": {"target": "prominent_face", "face_context": "prominent_person"}}, {"operation_id": "face_embedding", "type": "embed_clip_vit_b_32", "params": {"target": "prominent_face", "face_context": "prominent_person"}}]}'
```

This will return a JSON response containing the results for each requested analysis task.

**Using `curl` for file upload:**

The command below uploads a local file `some_person.jpg` and asks to find a face in it.

```bash
curl -X POST "http://localhost:8000/analyze_image_upload/" -H "Content-Type: multipart/form-data" -F "image_file=@/path/to/some_person.jpg" -F 'tasks_json=[{"operation_id": "face_from_file", "type": "detect_bounding_box", "params": {"target": "prominent_face"}}]'
```

## Available Operations

The `type` field in each task object specifies the operation to perform. You can retrieve a live list of these from the `GET /available_operations/` endpoint.

### `detect_bounding_box`
Detects a bounding box for a specified target.
*   **`params`**:
    *   `target` (string, required):
        *   `"prominent_person"`: Detects the bounding box of the most prominent person. (Default if `target` is omitted for this operation type).
        *   `"prominent_face"`: Detects the bounding box of the most prominent face.
    *   `face_context` (string, optional, default: `"prominent_person"`): When `target` is `"prominent_face"`, this specifies where to look for the face:
        *   `"prominent_person"`: Looks for a face within the bounding box of the already detected prominent person.
        *   `"whole_image"`: Looks for a face in the entire image.
*   **`data` in result**: An array `[xmin, ymin, xmax, ymax]` representing the bounding box. `null` if not found.

### `embed_clip_vit_b_32`
Generates an embedding using the CLIP ViT-B/32 model.
*   **`params`**:
    *   `target` (string, optional, default: `"whole_image"`):
        *   `"whole_image"`: Generates embedding for the entire image.
        *   `"prominent_person"`: Generates embedding for the cropped region of the most prominent person. If no person is found, falls back to the whole image.
        *   `"prominent_face"`: Generates embedding for the cropped region of the most prominent face. Requires a face to be found.
    *   `face_context` (string, optional, default: `"prominent_person"`): Same as in `detect_bounding_box`, used when `target` is `"prominent_face"`.
*   **`data` in result**: An array of floats representing the embedding vector.
*   **`cropped_image_bbox` / `cropped_image_base64` in result**: Populated if `target` was `"prominent_person"` (and a person was found and cropped) or `"prominent_face"` (and a face was found and cropped).

### `embed_dino_v2`
Generates a visual embedding using the DINOv2 model. This is useful for similarity search based on visual features like composition, color, and texture, rather than semantic content.
*   **`params`**:
    *   `target` (string, optional, default: `"whole_image"`):
        *   `"whole_image"`: Generates embedding for the entire image.
        *   `"prominent_person"`: Generates embedding for the cropped region of the most prominent person. If no person is found, falls back to the whole image.
        *   `"prominent_face"`: Generates embedding for the cropped region of the most prominent face. Requires a face to be found.
    *   `face_context` (string, optional, default: `"prominent_person"`): Same as in `detect_bounding_box`, used when `target` is `"prominent_face"`.
*   **`data` in result**: An array of floats representing the DINOv2 embedding vector.
*   **`cropped_image_bbox` / `cropped_image_base64` in result**: Populated if a crop was performed.

### `embed_dino_v3`
Generates a visual embedding using DINOv3 (default checkpoint: `facebook/dinov3-vitl16-pretrain-lvd1689m`).
*   **`params`**:
    *   `target` (string, optional, default: `"whole_image"`): Same as in `embed_dino_v2`.
    *   `face_context` (string, optional, default: `"prominent_person"`): Same as in `embed_dino_v2`.
*   **`data` in result**: An array of floats representing the DINOv3 embedding vector.
*   **`cropped_image_bbox` / `cropped_image_base64` in result**: Populated if a crop was performed.

### `classify`
Determines if an image region belongs to a specific collection using a pre-trained binary classifier. For each `collection_id`, a unique model is trained to predict whether an item is part of that collection (`true`) or not (`false`). The embedding logic used to get the vector for classification is identical to `embed_clip_vit_b_32`.
*   **`params`**:
    *   `collection_id` (integer, required): The ID of the collection to check against. This corresponds to the models trained by `scripts/train_classifiers.py`.
    *   `target` (string, optional, default: `"whole_image"`): Same as in `embed_clip_vit_b_32`.
    *   `face_context` (string, optional, default: `"prominent_person"`): Same as in `embed_clip_vit_b_32`, used when `target` is `"prominent_face"`.
*   **`data` in result**: A dictionary containing a boolean `is_in_collection` and the `probability` (float from 0.0 to 1.0) of that being true. Example: `{"is_in_collection": true, "probability": 0.95}`. A task will be skipped with an error if no classifier model for the requested `collection_id` is found on the server.
*   **`cropped_image_bbox` / `cropped_image_base64` in result**: Populated if the `target` for classification was not `"whole_image"`, following the same logic as `embed_clip_vit_b_32`.

### `describe_image`
Generates a text description of an image region using a pre-trained image captioning model.
*   **`params`**:
    *   `target` (string, optional, default: `"whole_image"`):
        *   `"whole_image"`: Generates a description for the entire image.
        *   `"prominent_person"`: Generates a description for the cropped region of the most prominent person. If no person is found, falls back to the whole image.
        *   `"prominent_face"`: Generates a description for the cropped region of the most prominent face. Requires a face to be found.
    *   `face_context` (string, optional, default: `"prominent_person"`): Same as in `detect_bounding_box`, used when `target` is `"prominent_face"`.
    *   `max_length` (integer, optional, default: `50`): The maximum number of tokens for the generated description.
*   **`data` in result**: A string containing the generated text description.
*   **`cropped_image_bbox` / `cropped_image_base64` in result**: Populated if `target` was `"prominent_person"` (and a person was found and cropped) or `"prominent_face"` (and a face was found and cropped).

## Helper Scripts

### `scripts/caption_folder.py`

This script provides a convenient way to generate text captions for all images in a specified folder. It iterates through each image, sends it to the running API's `/analyze_image_upload/` endpoint to get a description, and saves the resulting text to a `.txt` file with the same name as the image.

**Usage:**

First, ensure the main FastAPI service is running. Then, execute the script from your terminal:

```bash
python scripts/caption_folder.py /path/to/your/image_folder
```

*   The script will automatically find all images with common extensions (e.g., `.jpg`, `.png`).
*   For each `image_name.jpg`, it will create a `image_name.txt` file containing the description.
*   By default, it will skip images that already have a corresponding `.txt` file.

**Options:**

*   `--force` or `-f`: Use this flag to overwrite existing `.txt` caption files.

    ```bash
    python scripts/caption_folder.py /path/to/your/image_folder --force
    ```

### `scripts/train_classifiers.py`

This script trains binary classifiers for image collections using CLIP embeddings. It's part of the offline training pipeline that enables the `/classify` operation in the API.

**How it works:**

1. Fetches collection metadata from `https://crawlr.lawrenz.com/collections.json`
2. For each collection, downloads training data (pre-computed CLIP embeddings with labels)
3. Trains a LogisticRegression classifier with balanced class weights to handle imbalanced datasets
4. Evaluates all model versions (including previous ones) on a held-out test set (20% split)
5. Ranks models by macro precision and keeps the top 10 per collection
6. Saves the best model with a compatible pickle protocol for production use
7. Generates `trained_classifiers/best_models.json` mapping collection IDs to their best models

**Usage:**

```bash
python scripts/train_classifiers.py
```

The script will create timestamped model files in the `trained_classifiers/` directory:
- `collection_{id}_classifier_{timestamp}.pkl` - Individual model versions
- `collection_{id}_compatible_classifier.pkl` - Best model for production
- `best_models.json` - Configuration mapping collection IDs to best models

**Key Features:**

*   **Model Versioning:** Each training run creates a timestamped model for comparison
*   **Bake-off Evaluation:** All versions compete on the latest test data
*   **Automatic Pruning:** Keeps only the top 10 models per collection to manage disk space
*   **Imbalanced Data Handling:** Uses `class_weight="balanced"` for better performance on skewed datasets
*   **JSON Extraction:** Handles HTML-wrapped responses from the training data API

## Model and Device Management

The service uses multiple pre-trained models for different tasks:

*   **CLIP (ViT-B/32):** Default model for semantic embeddings. Can be changed via the `MODEL_NAME_CLIP` variable in `main.py` to other CLIP variants (e.g., `"ViT-L/14"`, `"RN50x16"`).
*   **DINOv2:** Visual embedding model for similarity search based on composition, color, and texture features.
*   **Faster R-CNN:** Person detection using TorchVision's pre-trained model.
*   **MTCNN:** Face detection via facenet-pytorch.
*   **Image Captioning Models:** Transformers-based models for generating image descriptions.
*   **Binary Classifiers:** Scikit-learn LogisticRegression models trained on CLIP embeddings, stored in `trained_classifiers/`.

All models are loaded on-demand and cached by the `app.core.model_loader` module. This module handles device selection automatically:
*   **GPU (CUDA / ROCm-HIP):** Used if available for significantly better performance
*   **CPU:** Fallback option if no GPU is detected

You can control this behavior via environment variables:
* `IMAGE_EMBED_DEVICE=cuda|cpu` to force a specific device.
* `IMAGE_EMBED_REQUIRE_GPU=1` to fail startup if no GPU is available (prevents accidentally running on CPU in production).

Models are pre-loaded at application startup via the lifespan manager to minimize first-request latency. Keep in mind that larger models offer better accuracy but require more computational resources (CPU/GPU and memory) and may be slower to load initially.

## Logging

The application uses Python's built-in `logging` module configured to output INFO level messages to the console. 

The API includes middleware that logs:
*   Request timing (duration in seconds)
*   Worker process ID (PID)
*   Request start and finish events
*   Detailed timing breakdowns for different operations (detection, embedding, classification, description)

This makes it easy to monitor performance and troubleshoot issues in production.

## To-Do / Potential Improvements

*   Allow users to specify the CLIP model via the API request.
*   Implement batch processing for multiple image URLs.
*   Add more robust error handling and input validation.
*   Add authentication.
*   Containerize the application (e.g., using Docker).

## Tech Stack

*   **Language:** Python 3.12
*   **Framework:** FastAPI with Uvicorn
*   **ML/CV Libraries:** PyTorch, TorchVision, OpenAI CLIP, DINOv2 (via Transformers), facenet-pytorch (MTCNN), scikit-learn
*   **Data Processing:** Pillow, NumPy
*   **Testing:** pytest
*   **CI/CD:** CircleCI

## Contributing

This project uses conventional commits for Git history:
*   `feat:` for new features
*   `refactor:` for code restructuring
*   `docs:` for documentation updates

See `openspec/project.md` for detailed project conventions and architecture patterns.
