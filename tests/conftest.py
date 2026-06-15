"""Shared pytest fixtures."""

import os

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="session")
def sample_pdf_path() -> str:
    path = os.path.join(_ROOT, "sample_data", "sample_permit.pdf")
    if not os.path.exists(path):
        pytest.skip("sample_permit.pdf not generated; run make_sample_permit.py")
    return path
