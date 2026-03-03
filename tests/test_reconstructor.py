import torch
import numpy as np
from ef_vfm.modules.transformer import Reconstructor


def test_output_shapes_mixed(make_reconstructor, dims):
    d = dims
    r = make_reconstructor(d["d_numerical"], d["categories"], d["d_token"])
    seq_len = d["d_numerical"] + len(d["categories"])
    h = torch.randn(d["batch_size"], seq_len, d["d_token"])
    x_num, x_cat = r(h)
    assert x_num.shape == (d["batch_size"], d["d_numerical"])
    assert len(x_cat) == len(d["categories"])
    for i, k in enumerate(d["categories"]):
        assert x_cat[i].shape == (d["batch_size"], k)


def test_categorical_count(make_reconstructor, dims):
    d = dims
    r = make_reconstructor(d["d_numerical"], d["categories"], d["d_token"])
    seq_len = d["d_numerical"] + len(d["categories"])
    h = torch.randn(d["batch_size"], seq_len, d["d_token"])
    _, x_cat = r(h)
    assert len(x_cat) == len(d["categories"])


def test_empty_categories(make_reconstructor):
    r = make_reconstructor(4, np.array([]), 16)
    h = torch.randn(8, 4, 16)
    x_num, x_cat = r(h)
    assert x_num.shape == (8, 4)
    assert len(x_cat) == 0


def test_weight_shape(make_reconstructor, dims):
    d = dims
    r = make_reconstructor(d["d_numerical"], d["categories"], d["d_token"])
    assert r.weight.shape == (d["d_numerical"], d["d_token"])


def test_gradient_flows(make_reconstructor, dims):
    d = dims
    r = make_reconstructor(d["d_numerical"], d["categories"], d["d_token"])
    seq_len = d["d_numerical"] + len(d["categories"])
    h = torch.randn(d["batch_size"], seq_len, d["d_token"])
    x_num, x_cat = r(h)
    loss = x_num.sum() + sum(c.sum() for c in x_cat)
    loss.backward()
    assert r.weight.grad is not None and r.weight.grad.abs().sum() > 0
    for recon in r.cat_recons:
        assert recon.weight.grad is not None
