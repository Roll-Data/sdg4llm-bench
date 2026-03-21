"""pytest configuration for SDG4LLM-Bench tests."""


def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests requiring network access "
        "(deselect with -m 'not integration')",
    )
    config.addinivalue_line(
        "markers",
        "gpu: marks tests that require a CUDA-capable GPU (deselect with -m 'not gpu')",
    )
