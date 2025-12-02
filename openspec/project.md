# Project Context

## Purpose
Advanced Image Analysis API - A FastAPI service that performs various computer vision analyses on images including:
- Object and face detection
- Multi-modal embeddings (CLIP for semantic, DINOv2 for visual features)
- Image classification using trained binary classifiers
- Image captioning/description
- Support for analyzing whole images, detected persons, or detected faces

The project includes both:
1. **API Service** (`main.py`) - FastAPI server for real-time image analysis
2. **Training Pipeline** (`scripts/train_classifiers.py`) - Automated classifier training from remote data source

## Tech Stack
- **Language:** Python 3.12
- **Framework:** FastAPI 
- **Web Server:** Uvicorn
- **ML/CV Libraries:**
  - PyTorch, TorchVision, TorchAudio
  - OpenAI CLIP (for semantic embeddings)
  - DINOv2 via Transformers (for visual embeddings)
  - facenet-pytorch (MTCNN for face detection)
  - Faster R-CNN (for person detection)
  - Transformers + Accelerate (for image captioning)
- **Data Handling:** Pillow, NumPy
- **HTTP:** Requests, python-multipart
- **ML:** scikit-learn (for classifiers)
- **Validation:** Pydantic
- **Testing:** pytest
- **CI/CD:** CircleCI

## Project Conventions

### Code Style
- **Architecture:** Service-oriented with clear separation of concerns
  - `/app/core/` - Core utilities (model_loader)
  - `/app/services/` - Business logic services (detection, embedding, classification, description, image_utils)
  - `/app/pydantic_models.py` - Request/response models
  - `main.py` - FastAPI app and endpoints
- **Naming:**
  - Functions: snake_case (e.g., `get_clip_embedding`, `download_image`)
  - Classes: PascalCase (e.g., `AnalysisTask`, `OperationResult`)
  - Constants: UPPER_SNAKE_CASE (e.g., `MODEL_NAME_CLIP`, `AVAILABLE_OPERATIONS`)
- **Logging:** Use Python's built-in logging module at INFO level
- **Error Handling:** HTTPException for API errors, ValueError for skippable tasks
- **Type Hints:** Use throughout (from typing import List, Dict, Optional, Any, Tuple)

### Architecture Patterns
- **Async/Await:** Endpoints are async, but ML operations are sync (blocking)
- **Lifespan Management:** Models pre-loaded at startup via asynccontextmanager
- **Shared Context:** Within single request, detection results cached to avoid redundant computations
- **Lazy Loading with Caching:** Models loaded on-demand and cached by model_loader
- **Task-Based API:** Clients submit lists of analysis tasks, each with operation_id for result mapping
- **Service Layer:** Each major capability (detection, embedding, classification, description) in separate service module
- **Device Management:** Automatic CUDA/CPU detection handled by model_loader
- **Classifier Training Workflow:**
  1. Fetch collections from remote API (`BASE_URL/collections.json`)
  2. For each collection, fetch training data (embeddings + labels)
  3. Train LogisticRegression classifier with balanced class weights (C=0.1, liblinear solver)
  4. Run bake-off: evaluate all versions (old + new) on held-out test set (20% split)
  5. Keep top 10 models per collection by macro precision, prune rest
  6. Save best model as `collection_{id}_compatible_classifier.pkl` (pickle protocol 4)
  7. Generate `best_models.json` mapping collection_id → best model filename

### Testing Strategy
- **Location:** `tests/` directory with `test_unit/` and `test_integration/` subdirectories
- **Framework:** pytest
- **Configuration:** pytest.ini configures test discovery
- **Excluded:** `.git`, `.venv`, and specific model directories from test discovery
- **CI:** Tests run on CircleCI for every push

### Git Workflow
- **Main Branch:** `main` (protected)
- **Commit Format:** Conventional commits style
  - `feat:` for new features
  - `refactor:` for code restructuring
  - `docs:` for documentation
  - Example: "feat: Integrate DINOv2 for visual embeddings"
- **CI:** CircleCI runs build-and-test job on all commits
- **Status Badge:** CircleCI build status displayed in README

## Domain Context
- **Computer Vision Pipeline:** Images processed through detection → cropping → embedding/analysis
- **Target Types:** Operations can target `whole_image`, `prominent_person`, or `prominent_face`
- **Face Context:** When detecting faces, can search within detected person bbox or whole image
- **Bounding Boxes:** Format is `[xmin, ymin, xmax, ymax]` in pixel coordinates
- **Embeddings:**
  - CLIP embeddings: Semantic, text-aligned features (default: ViT-B/32, 512 dims)
  - DINO embeddings: Visual features for composition, color, texture similarity
- **Operation Results:** Each task returns status (success/error/skipped), data, and optional cropped image (bbox + base64 PNG)
- **Classifiers:** 
  - Binary classifiers trained per collection_id using scikit-learn LogisticRegression on CLIP embeddings
  - Remote data source at `https://crawlr.lawrenz.com` provides collections and training data
  - Training data format: `[{embedding: [...], label: 0/1}, ...]`
  - Each classifier predicts probability of image belonging to specific collection
  - Multiple model versions kept, best selected via macro precision on test set
  - Handles class imbalance via `class_weight="balanced"`
- **Model Loading:** Models cached in memory after first use; pre-loaded at startup for performance
- **Model Versioning:** Timestamped models (YYYY-MM-DD_HHMMSS) with automatic pruning of low-performers

## Important Constraints
- **Synchronous ML Operations:** ML inference is blocking (not async)
- **Memory:** Multiple large models loaded in memory (CLIP, DINOv2, Faster R-CNN, MTCNN, caption model)
- **GPU Recommended:** CUDA significantly improves performance but CPU fallback available
- **Single Image Per Request:** Each request processes one image (URL or upload) with multiple tasks
- **No Batch Processing:** Currently one image at a time (noted as TODO in README)
- **Model Files:** Trained classifiers stored in `trained_classifiers/` directory
- **Request Timing:** Middleware logs per-request timing and worker PID

## External Dependencies
- **Image Sources:** Accepts HTTP/HTTPS URLs or direct file uploads
- **Pre-trained Models:**
  - OpenAI CLIP models (downloaded by clip library)
  - DINOv2 from Hugging Face Transformers
  - Faster R-CNN from TorchVision
  - MTCNN from facenet-pytorch
  - Image captioning models from Transformers
- **Training Data API:** `https://crawlr.lawrenz.com`
  - `/collections.json` - List of available collections
  - `/collections/{collection_id}/training_data.json` - Embeddings and labels per collection
  - Note: Responses may be wrapped in HTML, requires JSON extraction via regex
- **Runtime:** All inference done locally (no external API calls during image analysis)
- **CI/CD:** CircleCI for automated testing
