import pytest
from infscale.module.zoo import Zoo
from tests.module.conftest import supported_model_names

bad_testdata = [
    ("albert-base-v1", KeyError),  # not supported
    ("noexist_model", OSError),  # doesn't exist
]


@pytest.mark.parametrize("model_name", supported_model_names)
def test_get_model_metadata_success(model_name):
    model_md = Zoo.get_model_metadata(model_name)
    assert model_md is not None


@pytest.mark.parametrize("model_name,expected", bad_testdata)
def test_get_model_metadata_fail(model_name, expected):
    with pytest.raises((EnvironmentError, KeyError)) as excinfo:
        _ = Zoo.get_model_metadata(model_name)
    assert excinfo.type is expected
