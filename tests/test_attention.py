import pytest
import torch
from ef_vfm.modules.transformer import MultiheadAttention


def test_output_shape_single_head():
    attn = MultiheadAttention(d=16, n_heads=1, dropout=0.0)
    x = torch.randn(4, 5, 16)
    out = attn(x, x)
    assert out.shape == (4, 5, 16)


def test_output_shape_multi_head():
    attn = MultiheadAttention(d=16, n_heads=4, dropout=0.0)
    x = torch.randn(4, 5, 16)
    out = attn(x, x)
    assert out.shape == (4, 5, 16)


def test_no_W_out_single_head():
    attn = MultiheadAttention(d=16, n_heads=1, dropout=0.0)
    assert attn.W_out is None


def test_W_out_exists_multi_head():
    attn = MultiheadAttention(d=16, n_heads=4, dropout=0.0)
    assert attn.W_out is not None


def test_cross_attention_diff_seq_len():
    attn = MultiheadAttention(d=16, n_heads=1, dropout=0.0)
    x_q = torch.randn(4, 3, 16)
    x_kv = torch.randn(4, 7, 16)
    out = attn(x_q, x_kv)
    assert out.shape == (4, 3, 16)  # output seq_len matches query


def test_invalid_d_nheads_raises():
    with pytest.raises(AssertionError):
        MultiheadAttention(d=15, n_heads=4, dropout=0.0)


def test_gradient_flows():
    attn = MultiheadAttention(d=16, n_heads=2, dropout=0.0)
    x = torch.randn(4, 5, 16, requires_grad=True)
    out = attn(x, x)
    out.sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    for name in ['W_q', 'W_k', 'W_v']:
        param = getattr(attn, name)
        assert param.weight.grad is not None and param.weight.grad.abs().sum() > 0
