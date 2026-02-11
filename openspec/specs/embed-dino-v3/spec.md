# embed-dino-v3 Specification

## Purpose
Provide a first-class `embed_dino_v3` operation for generating modern DINOv3 visual embeddings (defaulting to `facebook/dinov3-vitl16-pretrain-lvd1689m`) for whole images and detected regions.
## Requirements
### Requirement: DINOv3 embedding operation
The system SHALL provide an `embed_dino_v3` operation that generates a visual embedding for an image region.

#### Scenario: Embed whole image
- **WHEN** a task requests `type=embed_dino_v3` with `target=whole_image`
- **THEN** the system returns an embedding vector for the full image

#### Scenario: Embed cropped region
- **WHEN** a task requests `type=embed_dino_v3` with `target=prominent_person` or `target=prominent_face`
- **THEN** the system detects the requested region and generates the embedding from the cropped region
- **AND** the response includes the bounding box used for the crop and a base64 PNG of the crop

### Requirement: Default DINOv3 model
The system SHALL use `facebook/dinov3-vitl16-pretrain-lvd1689m` as the default DINOv3 checkpoint if the client does not specify a different model identifier.

#### Scenario: Client does not specify model
- **WHEN** a task requests `embed_dino_v3` without a model identifier parameter
- **THEN** the system uses the default checkpoint

