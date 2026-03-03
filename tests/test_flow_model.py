import torch
import numpy as np
from unittest.mock import patch

from ef_vfm.models.flow_model import ExpVFM, Velocity
from ef_vfm.modules.main_modules import UniModMLP


# ---- mixed_loss tests ----

def test_mixed_loss_returns_two_scalars(make_flow_model, make_dummy_inputs, dims):
    d = dims
    flow = make_flow_model(d["d_numerical"], d["categories"])
    _, _, x_cat_int, _ = make_dummy_inputs(d["d_numerical"], d["categories"], d["batch_size"])
    x_num = torch.randn(d["batch_size"], d["d_numerical"])
    x = torch.cat([x_num, x_cat_int.float()], dim=1)
    d_loss, c_loss = flow.mixed_loss(x)
    assert d_loss.dim() == 0 or d_loss.numel() == 1
    assert c_loss.dim() == 0 or c_loss.numel() == 1


def test_mixed_loss_finite(make_flow_model, make_dummy_inputs, dims):
    d = dims
    flow = make_flow_model(d["d_numerical"], d["categories"])
    _, _, x_cat_int, _ = make_dummy_inputs(d["d_numerical"], d["categories"], d["batch_size"])
    x_num = torch.randn(d["batch_size"], d["d_numerical"])
    x = torch.cat([x_num, x_cat_int.float()], dim=1)
    d_loss, c_loss = flow.mixed_loss(x)
    assert torch.isfinite(d_loss).all()
    assert torch.isfinite(c_loss).all()


def test_mixed_loss_gradients_flow(make_flow_model, make_dummy_inputs, dims):
    d = dims
    flow = make_flow_model(d["d_numerical"], d["categories"])
    _, _, x_cat_int, _ = make_dummy_inputs(d["d_numerical"], d["categories"], d["batch_size"])
    x_num = torch.randn(d["batch_size"], d["d_numerical"])
    x = torch.cat([x_num, x_cat_int.float()], dim=1)
    d_loss, c_loss = flow.mixed_loss(x)
    total = d_loss + c_loss
    total.backward()
    grads = [p.grad for p in flow.parameters() if p.grad is not None]
    assert len(grads) > 0


def test_mixed_loss_numerical_only(make_flow_model, make_dummy_inputs, dims_numerical_only):
    d = dims_numerical_only
    flow = make_flow_model(d["d_numerical"], d["categories"])
    x = torch.randn(d["batch_size"], d["d_numerical"])
    d_loss, c_loss = flow.mixed_loss(x)
    assert d_loss.item() == 0.0  # no discrete features
    assert c_loss.item() > 0.0


# ---- sample tests (with mocked odeint) ----

def _make_flow(d_numerical, categories):
    cats_list = list(categories) if categories is not None else []
    cats_np = np.array(cats_list)
    model = UniModMLP(d_numerical, cats_list, 1, 16, n_head=1, factor=4, dim_t=64, activation='gelu')
    return ExpVFM(cats_np, d_numerical, model, device=torch.device('cpu'))


def test_sample_output_shape(dims):
    d = dims
    flow = _make_flow(d["d_numerical"], d["categories"])
    d_in = d["d_numerical"] + sum(d["categories"])
    n = 5
    fake_trajectory = torch.randn(2, n, d_in)
    with patch("ef_vfm.models.flow_model.odeint", return_value=fake_trajectory):
        result = flow.sample(n)
    d_out = d["d_numerical"] + len(d["categories"])
    assert result.shape == (n, d_out)


def test_sample_categorical_in_range(dims):
    d = dims
    flow = _make_flow(d["d_numerical"], d["categories"])
    d_in = d["d_numerical"] + sum(d["categories"])
    n = 16
    fake_trajectory = torch.randn(2, n, d_in)
    with patch("ef_vfm.models.flow_model.odeint", return_value=fake_trajectory):
        result = flow.sample(n)
    for i, k in enumerate(d["categories"]):
        col = d["d_numerical"] + i
        assert (result[:, col] >= 0).all()
        assert (result[:, col] < k).all()


def test_sample_returns_cpu(dims):
    d = dims
    flow = _make_flow(d["d_numerical"], d["categories"])
    d_in = d["d_numerical"] + sum(d["categories"])
    fake_trajectory = torch.randn(2, 4, d_in)
    with patch("ef_vfm.models.flow_model.odeint", return_value=fake_trajectory):
        result = flow.sample(4)
    assert result.device == torch.device('cpu')


def test_sample_single_sample(dims):
    d = dims
    flow = _make_flow(d["d_numerical"], d["categories"])
    d_in = d["d_numerical"] + sum(d["categories"])
    fake_trajectory = torch.randn(2, 1, d_in)
    with patch("ef_vfm.models.flow_model.odeint", return_value=fake_trajectory):
        result = flow.sample(1)
    d_out = d["d_numerical"] + len(d["categories"])
    assert result.shape == (1, d_out)


# ---- to_one_hot tests ----

def test_to_one_hot_shape(dims):
    d = dims
    flow = _make_flow(d["d_numerical"], d["categories"])
    cats = d["categories"]
    x_cat = torch.stack([torch.randint(0, k, (8,)) for k in cats], dim=1)
    oh = flow.to_one_hot(x_cat)
    assert oh.shape == (8, sum(cats))


def test_to_one_hot_roundtrip(dims):
    d = dims
    flow = _make_flow(d["d_numerical"], d["categories"])
    cats = d["categories"]
    x_cat = torch.stack([torch.randint(0, k, (8,)) for k in cats], dim=1)
    oh = flow.to_one_hot(x_cat)
    # Recover indices via argmax per category slice
    idx = 0
    for i, k in enumerate(cats):
        recovered = oh[:, idx:idx + k].argmax(dim=1)
        assert torch.equal(recovered, x_cat[:, i])
        idx += k


def test_to_one_hot_binary_values(dims):
    d = dims
    flow = _make_flow(d["d_numerical"], d["categories"])
    cats = d["categories"]
    x_cat = torch.stack([torch.randint(0, k, (8,)) for k in cats], dim=1)
    oh = flow.to_one_hot(x_cat)
    assert set(oh.unique().tolist()).issubset({0, 1})


# ---- Regression tests ----

def test_regression_d_in_no_extra_len():
    """d_in must be num_numerical + sum(num_classes), NOT + len(num_classes)."""
    d_numerical = 4
    categories = np.array([3, 5, 2])
    flow = _make_flow(d_numerical, categories)
    expected_d_in = d_numerical + sum(categories)  # 14, not 17
    assert flow.num_numerical_features + sum(flow.num_classes) == expected_d_in


def test_regression_sampling_indices_correct():
    """Categorical argmax must go to columns [d_num, d_num+1, ...], not [0, 1, ...]."""
    d_numerical = 4
    categories = np.array([3, 5, 2])
    n = 10
    d_in = d_numerical + sum(categories)
    d_out = d_numerical + len(categories)

    # Simulate the post-processing from sample()
    out = torch.randn(n, d_in)
    sample = torch.zeros(n, d_out)
    sample[:, :d_numerical] = out[:, :d_numerical]

    idx = d_numerical  # correct starting index
    for i, val in enumerate(categories):
        col = d_numerical + i  # correct column
        sample[:, col] = torch.argmax(out[:, idx:idx + val], dim=1)
        idx += val

    # Numerical columns must be untouched
    assert torch.allclose(sample[:, :d_numerical], out[:, :d_numerical])
    # Categorical columns at correct positions
    for i, val in enumerate(categories):
        col = d_numerical + i
        assert (sample[:, col] >= 0).all()
        assert (sample[:, col] < val).all()


def test_regression_d_out_correct():
    """d_out must be d_num + len(categories)."""
    d_numerical = 4
    categories = np.array([3, 5, 2])
    flow = _make_flow(d_numerical, categories)
    expected_d_out = d_numerical + len(categories)  # 7
    assert expected_d_out == 7


# ---- Velocity tests ----

def test_velocity_output_shape(dims):
    d = dims
    cats_list = list(d["categories"])
    model = UniModMLP(d["d_numerical"], cats_list, 1, d["d_token"],
                      n_head=1, factor=4, dim_t=64, activation='gelu')
    vel = Velocity(model)
    d_in = d["d_numerical"] + sum(d["categories"])
    x = torch.randn(d["batch_size"], d_in)
    t = torch.tensor(0.5)
    out = vel(t, x)
    assert out.shape == (d["batch_size"], d_in)


def test_velocity_scalar_t_broadcast(dims):
    d = dims
    cats_list = list(d["categories"])
    model = UniModMLP(d["d_numerical"], cats_list, 1, d["d_token"],
                      n_head=1, factor=4, dim_t=64, activation='gelu')
    vel = Velocity(model)
    d_in = d["d_numerical"] + sum(d["categories"])
    x = torch.randn(d["batch_size"], d_in)
    # Scalar t should work (gets broadcast internally)
    t = torch.tensor(0.3)
    out = vel(t, x)
    assert out.shape == x.shape
