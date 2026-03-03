import pytest
import torch
from ef_vfm.modules.transformer import Transformer


def test_output_shape_preserved(make_transformer):
    t = make_transformer(d_token=16, n_layers=2)
    x = torch.randn(4, 5, 16)
    out = t(x)
    assert out.shape == x.shape


def test_activation_gelu(make_transformer):
    t = make_transformer(d_token=16, activation='gelu')
    x = torch.randn(4, 5, 16)
    out = t(x)
    assert out.shape == x.shape


def test_activation_silu(make_transformer):
    t = make_transformer(d_token=16, activation='silu')
    x = torch.randn(4, 5, 16)
    out = t(x)
    assert out.shape == x.shape


def test_activation_relu(make_transformer):
    t = make_transformer(d_token=16, activation='relu')
    x = torch.randn(4, 5, 16)
    out = t(x)
    assert out.shape == x.shape


def test_invalid_activation_raises():
    with pytest.raises(ValueError, match="Unknown activation"):
        Transformer(2, 16, 1, 16, 4, activation='bad')


def test_prenorm_first_layer_no_norm0():
    t = Transformer(2, 16, 1, 16, 4, prenormalization=True)
    assert 'norm0' not in t.layers[0]
    # Second layer should have norm0
    assert 'norm0' in t.layers[1]


def test_no_prenorm_all_layers_have_norm0():
    t = Transformer(2, 16, 1, 16, 4, prenormalization=False)
    for layer in t.layers:
        assert 'norm0' in layer


def test_single_layer():
    t = Transformer(1, 16, 1, 16, 4)
    x = torch.randn(4, 5, 16)
    out = t(x)
    assert out.shape == x.shape


def test_multi_layer():
    t = Transformer(4, 16, 1, 16, 4)
    x = torch.randn(4, 5, 16)
    out = t(x)
    assert out.shape == x.shape


def test_gradient_flows(make_transformer):
    t = make_transformer(d_token=16, n_layers=2)
    x = torch.randn(4, 5, 16, requires_grad=True)
    out = t(x)
    out.sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    # Check gradients through at least the first layer's linear0
    assert t.layers[0]['linear0'].weight.grad is not None
