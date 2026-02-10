## 1. Model loader logging

- [x] 1.1 Update `_log_accelerator_info()` to log a debug message (and keep going) when optional device property queries fail.

## 2. Tests

- [x] 2.1 Add a unit test that simulates `torch.cuda.get_device_properties(0)` raising, and asserts `_log_accelerator_info()` does not raise (and emits a debug log).

## 3. Verify

- [x] 3.1 Run the unit tests (pytest) for the updated behavior.
