from PIL import Image
from app.services.image_utils import get_cropped_image, crop_image_and_get_base64


def test_get_cropped_image_caches_in_shared_context():
    """Second call with same bbox returns the cached result without re-cropping."""
    img = Image.new('RGB', (200, 200), color='red')
    bbox = [50, 50, 150, 150]
    ctx = {}

    # First call — should crop
    cropped1, b64_1 = get_cropped_image(img, bbox, shared_context=ctx)
    assert cropped1.size == (100, 100)
    assert isinstance(b64_1, str)
    cache_key = f"crop_{(50, 50, 150, 150)}"
    assert cache_key in ctx

    # Second call — should return cached instance
    cropped2, b64_2 = get_cropped_image(img, bbox, shared_context=ctx)
    assert cropped2 is cropped1  # same object (cached)
    assert b64_2 == b64_1


def test_get_cropped_image_without_shared_context():
    """Without shared_context, still returns valid crop (no caching)."""
    img = Image.new('RGB', (200, 200), color='blue')
    bbox = [0, 0, 100, 100]
    cropped, b64 = get_cropped_image(img, bbox)
    assert cropped.size == (100, 100)
    assert isinstance(b64, str)


def test_get_cropped_image_different_bboxes():
    """Different bboxes produce different cached entries."""
    img = Image.new('RGB', (200, 200), color='green')
    ctx = {}

    bbox_a = [0, 0, 100, 100]
    bbox_b = [100, 100, 200, 200]

    a1, _ = get_cropped_image(img, bbox_a, shared_context=ctx)
    b1, _ = get_cropped_image(img, bbox_b, shared_context=ctx)

    assert a1 is not b1  # different crops
    assert f"crop_{(0, 0, 100, 100)}" in ctx
    assert f"crop_{(100, 100, 200, 200)}" in ctx


def test_crop_image_and_get_base64_base():
    """Verify the base crop function still works as expected."""
    img = Image.new('RGB', (100, 100), color='yellow')
    bbox = [25, 25, 75, 75]
    cropped, b64 = crop_image_and_get_base64(img, bbox)
    assert cropped.size == (50, 50)
    assert isinstance(b64, str)
    assert b64.startswith('iVBOR')  # PNG magic
