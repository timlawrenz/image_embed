# Image Embedding API

A simple FastAPI service that generates embeddings for images using OpenAI's CLIP model. You provide an image URL, and the API returns a vector embedding of that image.

## Features

*   Accepts an image URL as input.
*   Downloads the image.
*   Converts the image to RGB format if necessary (e.g., handling alpha channels).
*   **Detects humans in the image (using Faster R-CNN with a ResNet50 FPN backbone from `torchvision.models.detection`):**
    *   If one or more humans are detected, the image is cropped to the bounding box of the human with the highest detection confidence. The embedding is then generated for this cropped region.
    *   If no human is detected, the embedding is generated for the entire image.
*   Preprocesses the (potentially cropped) image for CLIP.
*   Generates an image embedding using a specified CLIP model (default: "ViT-B/32").
*   Returns the image URL, the generated embedding, and the model name used.
*   Basic error handling for image download, processing, and human detection.

## API Endpoint

### `POST /generate_embedding_from_url/`

Generates an image embedding from a given image URL.

**Request Body:**

```json
{
  "image_url": "YOUR_IMAGE_URL_HERE"
}
```

*   `image_url` (string, required): A valid HTTP or HTTPS URL pointing to an image.

**Response Body (Success - 200 OK):**

```json
{
  "image_url": "THE_IMAGE_URL_PROVIDED",
  "embedding": [0.123, 0.456, ..., 0.789],
  "model_name": "ViT-B/32"
}
```

*   `image_url` (string): The URL of the image for which the embedding was generated.
*   `embedding` (array of floats): The generated image embedding vector.
*   `model_name` (string): The name of the CLIP model used to generate the embedding.

**Error Responses:**

*   **400 Bad Request:** If the image URL is invalid, the image cannot be downloaded, or the image cannot be processed.
    ```json
    {
      "detail": "Could not download image from URL: <error_details>"
    }
    ```
    ```json
    {
      "detail": "Could not process image or detect human: <error_details>"
    }
    ```
    ```json
    {
      "detail": "Cropped human region resulted in an empty image."
    }
    ```
*   **500 Internal Server Error:** If there's an issue generating the embedding on the server side or an internal error during processing (e.g., invalid bounding box from detection).
    ```json
    {
      "detail": "Failed to generate embedding: <error_details>"
    }
    ```
    ```json
    {
      "detail": "Invalid bounding box detected."
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

The application will typically be available at `http://localhost:8000`.

## Usage Example

You can use `curl` or any API client (like Postman or Insomnia) to send requests.

**Using `curl`:**

```bash
curl -X POST "http://localhost:8000/generate_embedding_from_url/" \
     -H "Content-Type: application/json" \
     -d '{"image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"}'
```

This will return a JSON response containing the embedding for the provided cat image.

## CLIP Model

The service currently uses the `"ViT-B/32"` CLIP model by default. You can change the `MODEL_NAME` variable in `main.py` to use other available CLIP models (e.g., `"ViT-L/14"`, `"RN50x16"`). Keep in mind that larger models might offer better accuracy but will require more computational resources (CPU/GPU and memory) and may be slower.

The code attempts to use a CUDA-enabled GPU if available (`device = "cuda"`); otherwise, it falls back to the CPU (`device = "cpu"`).

## Logging

The application uses Python's built-in `logging` module. Basic logging is configured to output INFO level messages to the console.

## To-Do / Potential Improvements

*   Allow users to specify the CLIP model via the API request.
*   Implement batch processing for multiple image URLs.
*   Add more robust error handling and input validation.
*   Option to upload an image directly instead of providing a URL.
*   Add authentication.
*   Containerize the application (e.g., using Docker).
