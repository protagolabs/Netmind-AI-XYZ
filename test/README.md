## Test Method

### Introduction

If you have modified the xyz module, you need to update the unit tests in the test/xyz directory accordingly.

The file directory structure of `./test/xyz` mirrors that of `./xyz/`.

### Testing Requirements

1. Use pytest for running the tests.
2. All components require unit tests, except for `xyz/utils/llm`.
3. Any code section requiring llm should use the TestClient from `./xyz/utils/test_tool/test_client.py` as the `llm_client`.
4. The purpose of testing is to:
    1. Verify the expected behavior for every object, method, and function.
    2. Reveal potential edge cases or exception scenarios.

