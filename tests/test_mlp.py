import torch
import torch.nn as nn
from ef_vfm.modules.main_modules import MLP, PositionalEmbedding


# ---- PositionalEmbedding tests ----

def test_positional_embedding_shape():
    pe = PositionalEmbedding(num_channels=64)
    x = torch.rand(8)
    out = pe(x)
    assert out.shape == (8, 64)


def test_positional_embedding_bounded():
    pe = PositionalEmbedding(num_channels=64)
    x = torch.rand(8)
    out = pe(x)
    assert out.min() >= -1.0
    assert out.max() <= 1.0


def test_positional_embedding_deterministic():
    pe = PositionalEmbedding(num_channels=64)
    x = torch.tensor([0.1, 0.5, 0.9])
    out1 = pe(x)
    out2 = pe(x)
    assert torch.equal(out1, out2)


def test_positional_embedding_different_timesteps():
    pe = PositionalEmbedding(num_channels=64)
    t1 = torch.tensor([0.1])
    t2 = torch.tensor([0.9])
    assert not torch.allclose(pe(t1), pe(t2))


# ---- MLP tests ----

def test_mlp_output_shape(make_mlp):
    mlp = make_mlp(d_in=32, dim_t=64)
    x = torch.randn(8, 32)
    t = torch.rand(8)
    out = mlp(x, t)
    assert out.shape == (8, 32)


def test_mlp_use_mlp_true(make_mlp):
    mlp = make_mlp(d_in=32, dim_t=64, use_mlp=True)
    assert isinstance(mlp.mlp, nn.Sequential)


def test_mlp_use_mlp_false(make_mlp):
    mlp = make_mlp(d_in=32, dim_t=64, use_mlp=False)
    assert isinstance(mlp.mlp, nn.Linear)


def test_mlp_time_conditioning(make_mlp):
    mlp = make_mlp(d_in=32, dim_t=64)
    mlp.eval()
    x = torch.randn(4, 32)
    t1 = torch.zeros(4)
    t2 = torch.ones(4)
    out1 = mlp(x, t1)
    out2 = mlp(x, t2)
    assert not torch.allclose(out1, out2)


def test_mlp_gradient_flows(make_mlp):
    mlp = make_mlp(d_in=32, dim_t=64)
    x = torch.randn(4, 32)
    t = torch.rand(4)
    out = mlp(x, t)
    out.sum().backward()
    assert mlp.proj.weight.grad is not None and mlp.proj.weight.grad.abs().sum() > 0
    assert mlp.map_noise.num_channels == 64  # sanity check on PE config


def test_mlp_different_dim_t(make_mlp):
    for dim_t in [32, 128, 256]:
        mlp = make_mlp(d_in=16, dim_t=dim_t)
        x = torch.randn(4, 16)
        t = torch.rand(4)
        out = mlp(x, t)
        assert out.shape == (4, 16)
