import os
from pathlib import Path

from src.util import load_config
from ef_vfm.modules.main_modules import UniModMLP


CONFIG_PATH = Path(__file__).resolve().parent.parent / "ef_vfm" / "configs" / "ef_vfm_configs.toml"


def test_load_config_returns_dict():
    config = load_config(CONFIG_PATH)
    assert isinstance(config, dict)


def test_config_has_expected_sections():
    config = load_config(CONFIG_PATH)
    for key in ['data', 'unimodmlp_params', 'train', 'sample']:
        assert key in config, f"Missing section '{key}'"


def test_unimodmlp_params_complete():
    config = load_config(CONFIG_PATH)
    params = config['unimodmlp_params']
    required = ['num_layers', 'd_token', 'n_head', 'factor', 'bias', 'dim_t', 'use_mlp', 'activation']
    for key in required:
        assert key in params, f"Missing param '{key}' in unimodmlp_params"


def test_activation_value_is_valid():
    config = load_config(CONFIG_PATH)
    activation = config['unimodmlp_params']['activation']
    assert activation in ('relu', 'gelu', 'silu'), f"Invalid activation '{activation}'"


def test_train_main_has_new_params():
    """Verify the recently added config params are present."""
    config = load_config(CONFIG_PATH)
    train = config['train']['main']
    assert 'max_grad_norm' in train
    assert 'warmup_epochs' in train
    assert isinstance(train['max_grad_norm'], (int, float))
    assert isinstance(train['warmup_epochs'], (int, float))


def test_config_values_create_model():
    config = load_config(CONFIG_PATH)
    params = config['unimodmlp_params']
    # Use dummy dimensions; the point is that config params are valid for the constructor
    model = UniModMLP(
        d_numerical=4,
        categories=[3, 5, 2],
        num_layers=params['num_layers'],
        d_token=params['d_token'],
        n_head=params['n_head'],
        factor=params['factor'],
        bias=params['bias'],
        dim_t=params['dim_t'],
        use_mlp=params['use_mlp'],
        activation=params['activation'],
    )
    assert model is not None
