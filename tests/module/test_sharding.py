import pytest
from infscale.module.sharding import Sharder
from infscale.module.zoo import Zoo
from tests.module.conftest import model_input_names_pairs


@pytest.mark.parametrize("model_name,input_names", model_input_names_pairs)
def test_shard(model_name, input_names):
    mmd = Zoo.get_model_metadata(model_name)
    assert mmd is not None

    layers = Sharder.shard(mmd, input_names)
    assert layers is not None and len(layers) > 0
