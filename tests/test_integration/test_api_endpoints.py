from PIL import Image

def test_analyze_image_classification_on_face(client, mocker):
    """
    Tests a full pipeline: detect person -> detect face -> get embedding -> classify.
    Verifies that intermediate results are shared between tasks.
    """
    # 1. Arrange: Mock all external dependencies and service calls
    # We patch the functions in the namespace where they are *used* (main.py),
    # not where they are defined. This is a common requirement when using mock.
    mocker.patch("main.download_image", return_value=Image.new('RGB', (800, 600)))

    # Mock the service layer functions that the endpoint calls
    mock_get_person = mocker.patch("main.get_prominent_person_bbox", return_value=[100, 50, 300, 250])
    mock_get_face = mocker.patch("main.get_prominent_face_bbox_in_region", return_value=[110, 70, 160, 130])
    mock_get_embedding = mocker.patch("main.get_clip_embedding", return_value=([0.1]*512, "base64_string", [110, 70, 160, 130]))
    mock_classify = mocker.patch("main.classify_embedding", return_value={"is_in_collection": True, "probability": 0.98})

    # 2. Arrange: Define a complex request with multiple dependent tasks
    request_data = {
        "image_url": "http://example.com/image.jpg",
        "tasks": [
            {
                "operation_id": "classify_face",
                "type": "classify",
                "params": {
                    "target": "prominent_face",
                    "collection_id": 42
                }
            },
            {
                "operation_id": "find_person",
                "type": "detect_bounding_box",
                "params": {
                    "target": "prominent_person"
                }
            }
        ]
    }

    # 3. Act
    response = client.post("/analyze_image/", json=request_data)

    # 4. Assert
    assert response.status_code == 200
    results = response.json()["results"]

    # Check classification result
    assert results["classify_face"]["status"] == "success"
    assert results["classify_face"]["data"]["is_in_collection"] is True
    assert results["classify_face"]["cropped_image_bbox"] == [110, 70, 160, 130]

    # Check person detection result
    assert results["find_person"]["status"] == "success"
    assert results["find_person"]["data"] == [100, 50, 300, 250]

    # IMPORTANT: Assert that expensive operations were only called once
    # This proves the internal caching (`shared_context`) is working.
    mock_get_person.assert_called_once()
    # Face detection should also be called once, as it's needed by the first task.
    mock_get_face.assert_called_once()
    mock_get_embedding.assert_called_once()
    mock_classify.assert_called_once()


def test_analyze_image_describe_image(client, mocker):
    """
    Tests the 'describe_image' operation through the API.
    """
    # 1. Arrange
    mocker.patch("main.download_image", return_value=Image.new('RGB', (800, 600)))
    mock_get_description = mocker.patch(
        "main.get_image_description", 
        return_value=("A detailed description of the image.", None, None)
    )

    request_data = {
        "image_url": "http://example.com/image.jpg",
        "tasks": [
            {
                "operation_id": "describe_whole_image",
                "type": "describe_image",
                "params": {
                    "target": "whole_image"
                }
            }
        ]
    }

    # 3. Act
    response = client.post("/analyze_image/", json=request_data)

    # 4. Assert
    assert response.status_code == 200
    results = response.json()["results"]

    assert "describe_whole_image" in results
    assert results["describe_whole_image"]["status"] == "success"
    assert results["describe_whole_image"]["data"] == "A detailed description of the image."
    assert results["describe_whole_image"]["cropped_image_base64"] is None
    assert results["describe_whole_image"]["cropped_image_bbox"] is None

    mock_get_description.assert_called_once()
