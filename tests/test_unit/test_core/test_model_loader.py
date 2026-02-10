import logging


def test_log_accelerator_info_optional_property_failure(mocker, caplog):
    """Optional accelerator property queries should be non-fatal and logged."""
    from app.core import model_loader

    mocker.patch.object(model_loader.torch.cuda, "device_count", return_value=1)
    mocker.patch.object(model_loader.torch.cuda, "get_device_name", return_value="fake-gpu")
    mocker.patch.object(
        model_loader.torch.cuda,
        "get_device_properties",
        side_effect=RuntimeError("boom"),
    )

    caplog.set_level(logging.DEBUG, logger=model_loader.logger.name)

    model_loader._log_accelerator_info()

    assert any(
        "Unable to query optional device properties" in r.message for r in caplog.records
    )
