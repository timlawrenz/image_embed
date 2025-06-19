# Advanced Image Analysis API

[ ![CircleCI](https://dl.circleci.com/status-badge/img/gh/timlawrenz/image_embed/tree/main.svg?style=svg) ](https://dl.circleci.com/status-badge/redirect/gh/timlawrenz/image_embed/tree/main)

An advanced FastAPI service that performs various analyses on images from a URL. You provide an image URL and a list of analysis tasks, and the API returns the results for each task.

## Features

*   Accepts an image URL and a list of analysis tasks as input.
*   Downloads the image once per request.
*   Converts the image to RGB format if necessary.
*   Performs requested operations, which can include:
    *   **Human Detection:** Detects the most prominent person using Faster R-CNN.
    *   **Face Detection:** Detects the most prominent face using MTCNN.
    *   **CLIP Embedding:** Generates image embeddings using a specified CLIP model (default: "ViT-B/32") on the whole image, a detected person, or a detected face.
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

1.  **Clone the repository (if applicable):**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

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

### `classify`
Determines if an image region belongs to a specific collection using a pre-trained binary classifier. For each `collection_id`, a unique model is trained to predict whether an item is part of that collection (`true`) or not (`false`). The embedding logic used to get the vector for classification is identical to `embed_clip_vit_b_32`.
*   **`params`**:
    *   `collection_id` (integer, required): The ID of the collection to check against. This corresponds to the models trained by `scripts/train_classifiers.py`.
    *   `target` (string, optional, default: `"whole_image"`): Same as in `embed_clip_vit_b_32`.
    *   `face_context` (string, optional, default: `"prominent_person"`): Same as in `embed_clip_vit_b_32`, used when `target` is `"prominent_face"`.
*   **`data` in result**: A dictionary containing a boolean `is_in_collection` and the `probability` (float from 0.0 to 1.0) of that being true. Example: `{"is_in_collection": true, "probability": 0.95}`. A task will be skipped with an error if no classifier model for the requested `collection_id` is found on the server.
*   **`cropped_image_bbox` / `cropped_image_base64` in result**: Populated if the `target` for classification was not `"whole_image"`, following the same logic as `embed_clip_vit_b_32`.

## CLIP Model and Device Management

The service currently uses the `"ViT-B/32"` CLIP model by default for the `embed_clip_vit_b_32` operation. This default is specified by the `MODEL_NAME_CLIP` variable in `main.py`. You can change this variable to use a different default CLIP model (e.g., `"ViT-L/14"`, `"RN50x16"`).

Models (including CLIP and detection models) are loaded on-demand and cached by the `app.core.model_loader` module. This module also handles the device selection, attempting to use a CUDA-enabled GPU if available; otherwise, it falls back to the CPU.

Keep in mind that larger models might offer better accuracy but will require more computational resources (CPU/GPU and memory) and may be slower to load initially.

## Logging

The application uses Python's built-in `logging` module. Basic logging is configured to output INFO level messages to the console.

## To-Do / Potential Improvements

*   Allow users to specify the CLIP model via the API request.
*   Implement batch processing for multiple image URLs.
*   Add more robust error handling and input validation.
*   Option to upload an image directly instead of providing a URL.
*   Add authentication.
*   Containerize the application (e.g., using Docker).
