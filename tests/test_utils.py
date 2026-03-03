import torch
import numpy as np

from utils_train import update_ema, concat_y_to_X


# ---- update_ema tests ----

def test_update_ema_basic():
    target = [torch.tensor([1.0, 2.0])]
    source = [torch.tensor([3.0, 4.0])]
    target[0].requires_grad_(False)
    rate = 0.9
    update_ema(target, source, rate=rate)
    expected = 0.9 * torch.tensor([1.0, 2.0]) + 0.1 * torch.tensor([3.0, 4.0])
    assert torch.allclose(target[0], expected)


def test_update_ema_rate_zero():
    target = [torch.tensor([1.0, 2.0])]
    source = [torch.tensor([3.0, 4.0])]
    target[0].requires_grad_(False)
    update_ema(target, source, rate=0.0)
    assert torch.allclose(target[0], torch.tensor([3.0, 4.0]))


def test_update_ema_rate_one():
    target = [torch.tensor([1.0, 2.0])]
    source = [torch.tensor([3.0, 4.0])]
    target[0].requires_grad_(False)
    update_ema(target, source, rate=1.0)
    assert torch.allclose(target[0], torch.tensor([1.0, 2.0]))


# ---- concat_y_to_X tests ----

def test_concat_y_to_X_with_X():
    X = np.array([[1, 2], [3, 4]])
    y = np.array([10, 20])
    result = concat_y_to_X(X, y)
    expected = np.array([[10, 1, 2], [20, 3, 4]])
    np.testing.assert_array_equal(result, expected)


def test_concat_y_to_X_without_X():
    y = np.array([10, 20, 30])
    result = concat_y_to_X(None, y)
    expected = np.array([[10], [20], [30]])
    np.testing.assert_array_equal(result, expected)
