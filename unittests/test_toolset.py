from os.path import join
from utils.toolset import yaml2namespace

def test_yaml2namespace():
    path = join('unittests', 'assets', 'ppn2v_model.yaml')
    config = yaml2namespace(path)
    assert config.depth == 3
    assert config.num_classes == 800
    assert config.patch_size == 64
    assert config.initial_filters == 64
    assert config.type == "pn2v"