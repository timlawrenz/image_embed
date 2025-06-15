from typing import Any, Dict, List, Optional
from pydantic import BaseModel, HttpUrl, Field

class AnalysisTask(BaseModel):
    operation_id: str = Field(..., description="A unique ID for this specific requested operation, will be used as a key in the results.")
    type: str = Field(..., description="Type of operation to perform.", examples=["embed_clip_vit_b_32", "detect_bounding_box"])
    # Make params optional for default behaviors
    params: Optional[Dict[str, Any]] = Field(None, description="Parameters for the operation, e.g., {'target': 'whole_image' | 'prominent_person' | 'prominent_face'}")

class ImageAnalysisRequest(BaseModel):
    image_url: HttpUrl
    tasks: List[AnalysisTask]

class OperationResult(BaseModel):
    status: str = "success" # "success" or "error" or "skipped"
    data: Optional[Any] = None
    cropped_image_bbox: Optional[List[int]] = Field(None, description="Bounding box used to generate the cropped_image_base64, if applicable.")
    cropped_image_base64: Optional[str] = Field(None, description="Base64 encoded string of the PNG cropped image processed by this operation, if applicable.")
    error_message: Optional[str] = None

class ImageAnalysisResponse(BaseModel):
    image_url: str
    results: Dict[str, OperationResult]

class OperationInfo(BaseModel):
    description: str = Field(..., description="A description of what the operation does.")
    allowed_targets: List[str] = Field(..., description="A list of valid targets for this operation.")
    default_target: str = Field(..., description="The default target used if one is not specified in the request.")

class AvailableOperationsResponse(BaseModel):
    operations: Dict[str, OperationInfo] = Field(..., description="A dictionary of available operations, keyed by operation type.")

class OperationInfo(BaseModel):
    description: str = Field(..., description="A description of what the operation does.")
    allowed_targets: List[str] = Field(..., description="A list of valid targets for this operation.")
    default_target: str = Field(..., description="The default target used if one is not specified in the request.")

class AvailableOperationsResponse(BaseModel):
    operations: Dict[str, OperationInfo] = Field(..., description="A dictionary of available operations, keyed by operation type.")
