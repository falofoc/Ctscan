import pytest

from wiqaya import phantom


@pytest.fixture(scope="session")
def ph():
    return phantom.build()
