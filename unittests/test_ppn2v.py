import pytest
from os.path import join
from pytest_mock import mocker

from utils.toolset import yaml2namespace
from models.ppn2v import PPN2V


@pytest.fixture
def model_config():
    conf_path = join('unittests', 'assets', 'ppn2v_model.yaml')
    model_conf = yaml2namespace(conf_path)
    return model_conf

@pytest.fixture
def mock_init(mocker):
    return mocker.patch('test_ppn2v.PPN2V.__init__')

def test_model_build(mock_init, model_config):
    mock_init.return_value = None
    model = PPN2V(model_config)
    model._build_model(model_config)
    assert hasattr(model, 'unet')
    assert hasattr(model, 'noise_model')

    # TODO: call the super constructor in the mock somehow
