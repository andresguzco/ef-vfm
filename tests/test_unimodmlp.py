import torch
import numpy as np


def test_forward_shapes_mixed(make_unimodmlp, make_dummy_inputs, dims):
    d = dims
    model = make_unimodmlp(d["d_numerical"], d["categories"], d_token=d["d_token"])
    x_num, x_cat_oh, _, t = make_dummy_inputs(d["d_numerical"], d["categories"], d["batch_size"])
    x_num_pred, x_cat_pred = model(x_num, x_cat_oh, t)
    assert x_num_pred.shape == (d["batch_size"], d["d_numerical"])
    assert x_cat_pred.shape == (d["batch_size"], sum(d["categories"]))


def test_forward_shapes_numerical_only(make_unimodmlp, make_dummy_inputs, dims_numerical_only):
    d = dims_numerical_only
    model = make_unimodmlp(d["d_numerical"], d["categories"], d_token=d["d_token"])
    x_num, _, _, t = make_dummy_inputs(d["d_numerical"], d["categories"], d["batch_size"])
    x_cat = torch.zeros(d["batch_size"], 0)
    x_num_pred, x_cat_pred = model(x_num, x_cat, t)
    assert x_num_pred.shape == (d["batch_size"], d["d_numerical"])
    # When no categories, cat_pred should be zeros with shape matching x_cat
    assert x_cat_pred.shape[0] == d["batch_size"]
    assert torch.all(x_cat_pred == 0)


def test_forward_shapes_single_feature(make_unimodmlp, make_dummy_inputs, dims_single):
    d = dims_single
    model = make_unimodmlp(d["d_numerical"], d["categories"], d_token=d["d_token"])
    x_num, x_cat_oh, _, t = make_dummy_inputs(d["d_numerical"], d["categories"], d["batch_size"])
    x_num_pred, x_cat_pred = model(x_num, x_cat_oh, t)
    assert x_num_pred.shape == (d["batch_size"], d["d_numerical"])
    assert x_cat_pred.shape == (d["batch_size"], sum(d["categories"]))


def test_d_in_computation(make_unimodmlp, dims):
    d = dims
    model = make_unimodmlp(d["d_numerical"], d["categories"], d_token=d["d_token"])
    expected = d["d_token"] * (d["d_numerical"] + len(d["categories"]))
    assert model.mlp.proj.in_features == expected


def test_output_dtypes(make_unimodmlp, make_dummy_inputs, dims):
    d = dims
    model = make_unimodmlp(d["d_numerical"], d["categories"], d_token=d["d_token"])
    x_num, x_cat_oh, _, t = make_dummy_inputs(d["d_numerical"], d["categories"], d["batch_size"])
    x_num_pred, x_cat_pred = model(x_num, x_cat_oh, t)
    assert x_num_pred.dtype == torch.float32
    assert x_cat_pred.dtype == torch.float32


def test_gradient_flows_end_to_end(make_unimodmlp, make_dummy_inputs, dims):
    d = dims
    model = make_unimodmlp(d["d_numerical"], d["categories"], d_token=d["d_token"])
    x_num, x_cat_oh, _, t = make_dummy_inputs(d["d_numerical"], d["categories"], d["batch_size"])
    x_num_pred, x_cat_pred = model(x_num, x_cat_oh, t)
    loss = x_num_pred.sum() + x_cat_pred.sum()
    loss.backward()
    params_with_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    total_params = sum(1 for _ in model.parameters())
    # Transformer.head is defined but unused in forward(), so not all params get gradients
    assert params_with_grad > total_params * 0.8, f"Only {params_with_grad}/{total_params} params got gradients"


def test_different_activations(make_unimodmlp, make_dummy_inputs, dims):
    d = dims
    x_num, x_cat_oh, _, t = make_dummy_inputs(d["d_numerical"], d["categories"], d["batch_size"])
    for act in ['relu', 'gelu', 'silu']:
        model = make_unimodmlp(d["d_numerical"], d["categories"], d_token=d["d_token"], activation=act)
        x_num_pred, x_cat_pred = model(x_num, x_cat_oh, t)
        assert x_num_pred.shape == (d["batch_size"], d["d_numerical"])
        assert torch.isfinite(x_num_pred).all()
        assert torch.isfinite(x_cat_pred).all()
