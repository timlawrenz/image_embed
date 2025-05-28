import io
import logging

import clip
import requests
import torch
import torchvision # Added for detection model
import torchvision.transforms as T # Added for detection model
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, HttpUrl

# --- Configuration & Initialization ---

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Image Embedding API",
    description="Generates image embeddings from a URL using OpenAI CLIP.",
    version="0.1.0",
)

# Load CLIP model
# You can choose other CLIP models like "ViT-L/14", "RN50x16", etc.
# Larger models might provide better accuracy but will be slower and use more memory.
MODEL_NAME = "ViT-B/32"
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    model, preprocess = clip.load(MODEL_NAME, device=device, jit=False)
    logger.info(f"CLIP model '{MODEL_NAME}' loaded successfully on {device}.")
except Exception as e:
    logger.exception("Failed to load CLIP model. Ensure it's installed and accessible.")
    # If the model can't load, the app shouldn't start or should indicate a critical error.
    # For simplicity here, we'll let it raise, but in production, you might handle this more gracefully.
    raise RuntimeError(f"Could not load CLIP model: {e}") from e

# Load human detection model (e.g., Faster R-CNN)
try:
    detection_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    detection_model.to(device)
    detection_model.eval() # Set to evaluation mode
    logger.info("Human detection model (Faster R-CNN) loaded successfully.")
except Exception as e:
    logger.exception("Failed to load human detection model.")
    raise RuntimeError(f"Could not load human detection model: {e}") from e


# --- Pydantic Models for Request and Response ---

class ImageUrlRequest(BaseModel):
    image_url: HttpUrl # pydantic will validate if this is a valid HTTP/HTTPS URL

class EmbeddingResponse(BaseModel):
    image_url: str
    embedding: list[float]
    model_name: str

# --- API Endpoint ---

@app.post("/generate_embedding_from_url/", response_model=EmbeddingResponse)
async def generate_embedding_from_url(request_data: ImageUrlRequest):
    """
    Accepts an image URL, downloads the image, generates its embedding using CLIP,
    and returns the embedding as a JSON list of floats.
    """
    image_url_str = str(request_data.image_url)
    logger.info(f"Received request to generate embedding for URL: {image_url_str}")

    try:
        # 1. Download the image from the URL
        headers = {'User-Agent': 'ImageEmbeddingAPI/0.1'} # Some servers block requests without a User-Agent
        response = requests.get(image_url_str, stream=True, timeout=10, headers=headers) # 10-second timeout
        response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)
        image_bytes = response.content
        logger.info(f"Successfully downloaded image from {image_url_str}")

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download image from {image_url_str}: {e}")
        raise HTTPException(status_code=400, detail=f"Could not download image from URL: {e}")

    try:
        # 2. Load the image
        pil_image = Image.open(io.BytesIO(image_bytes))

        # Ensure image is RGB for consistency for detection and CLIP
        pil_image_rgb = pil_image
        if pil_image.mode != 'RGB':
            # Handle common cases like RGBA, LA (Luminance Alpha), P (Palette with transparency)
            if pil_image.mode == 'RGBA' or pil_image.mode == 'LA' or \
               (pil_image.mode == 'P' and 'transparency' in pil_image.info):
                pil_image_rgb = pil_image.convert('RGB')
                logger.info(f"Converted image from mode {pil_image.mode} (with alpha/palette) to RGB.")
            else:
                # General conversion for other modes like L, P (without transparency hint), CMYK etc.
                pil_image_rgb = pil_image.convert('RGB')
                logger.info(f"Converted image from mode {pil_image.mode} to RGB.")
        
        # --- Human Detection ---
        # Transform for detection model
        transform_detection = T.Compose([T.ToTensor()])
        img_tensor_detection = transform_detection(pil_image_rgb).to(device)

        with torch.no_grad(): # Disable gradients for detection
            predictions = detection_model([img_tensor_detection])

        # Extract bounding box for the human with the highest score
        # COCO class ID for 'person' is 1
        scores = predictions[0]['scores']
        labels = predictions[0]['labels']
        boxes = predictions[0]['boxes']

        person_indices = (labels == 1).nonzero(as_tuple=True)[0]
        
        image_for_clip_processing = pil_image_rgb # Default to the full RGB image

        if len(person_indices) > 0:
            # Human(s) detected, proceed to crop
            # Get the index of the person with the highest score among detected persons
            person_scores = scores[person_indices]
            highest_score_person_local_idx = person_scores.argmax()
            highest_score_person_global_idx = person_indices[highest_score_person_local_idx]
            
            bbox = boxes[highest_score_person_global_idx].cpu().numpy().astype(int)
            xmin, ymin, xmax, ymax = bbox
            
            detection_confidence = scores[highest_score_person_global_idx].item()
            logger.info(f"Human detected with bounding box: [{xmin}, {ymin}, {xmax}, {ymax}] and confidence: {detection_confidence:.2f}")

            # Crop the original RGB PIL image using the bounding box
            # Ensure box coordinates are valid (e.g. xmin < xmax)
            if xmin >= xmax or ymin >= ymax:
                logger.error(f"Invalid bounding box from detection: [{xmin}, {ymin}, {xmax}, {ymax}] for image {image_url_str}")
                raise HTTPException(status_code=500, detail="Invalid bounding box detected.")
                
            cropped_pil_image = pil_image_rgb.crop((xmin, ymin, xmax, ymax))
            if cropped_pil_image.width == 0 or cropped_pil_image.height == 0:
                logger.error(f"Cropped image has zero dimension: w={cropped_pil_image.width}, h={cropped_pil_image.height} from bbox [{xmin},{ymin},{xmax},{ymax}] on image {image_url_str}")
                raise HTTPException(status_code=400, detail="Cropped human region resulted in an empty image.")
            
            image_for_clip_processing = cropped_pil_image
            logger.info("Image cropped to human bounding box for CLIP processing.")
        else:
            # No human detected, use the full image
            logger.warning(f"No human detected in image from {image_url_str}. Using full image for CLIP processing.")
        # --- End Human Detection Logic ---

        # Preprocess the selected image (either cropped or full) for CLIP
        image_input = preprocess(image_for_clip_processing).unsqueeze(0).to(device)
        logger.info("Image (full or cropped) preprocessed successfully for CLIP.")

    except HTTPException: # Re-raise HTTPExceptions directly
        raise
    except Exception as e:
        logger.error(f"Failed to process image or detect human in {image_url_str}: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Could not process image or detect human: {e}")

    try:
        # 3. Generate the embedding for the (potentially cropped) image
        with torch.no_grad(): # Disables gradient calculations, important for inference
            image_embedding_tensor = model.encode_image(image_input)
        logger.info("Image embedding generated successfully.")

        # 4. Normalize the embedding (optional but common for cosine similarity)
        # image_embedding_tensor = image_embedding_tensor / image_embedding_tensor.norm(dim=-1, keepdim=True)

        # 5. Convert to Python list for JSON response
        # The tensor is typically of shape [1, embedding_dim]. We want the embedding_dim part.
        embedding_list = image_embedding_tensor[0].cpu().numpy().tolist()

        return EmbeddingResponse(
            image_url=image_url_str,
            embedding=embedding_list,
            model_name=MODEL_NAME
        )

    except Exception as e:
        logger.exception(f"Failed to generate or process embedding for {image_url_str}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate embedding: {e}")

# --- How to Run (for development) ---
# Save this code as main.py
# Run in your terminal: uvicorn main:app --reload
#
# Then you can make a POST request to http://localhost:8000/generate_embedding_from_url/
# with a JSON body like:
# {
#   "image_url": "YOUR_IMAGE_URL_HERE"
# }
# Example using curl:
# curl -X POST "http://localhost:8000/generate_embedding_from_url/" \
#      -H "Content-Type: application/json" \
#      -d '{"image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"}'

if __name__ == "__main__":
    import uvicorn
    # This part is for direct execution (python main.py),
    # but typically you'd use uvicorn CLI for more control.
    uvicorn.run(app, host="0.0.0.0", port=8000)
