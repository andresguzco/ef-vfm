import torch
import numpy as np


def test_forward_shape_mixed(make_tokenizer, make_dummy_inputs, dims):
    tok = make_tokenizer(dims["d_numerical"], dims["categories"], dims["d_token"])
    x_num, x_cat_oh, _, _ = make_dummy_inputs(dims["d_numerical"], dims["categories"], dims["batch_size"])
    out = tok(x_num, x_cat_oh)
    expected_seq = 1 + dims["d_numerical"] + len(dims["categories"])
    assert out.shape == (dims["batch_size"], expected_seq, dims["d_token"])


def test_forward_shape_numerical_only(make_tokenizer, make_dummy_inputs, dims_numerical_only):
    d = dims_numerical_only
    tok = make_tokenizer(d["d_numerical"], d["categories"], d["d_token"])
    x_num, _, _, _ = make_dummy_inputs(d["d_numerical"], d["categories"], d["batch_size"])
    out = tok(x_num, None)
    expected_seq = 1 + d["d_numerical"]
    assert out.shape == (d["batch_size"], expected_seq, d["d_token"])


def test_forward_shape_single_feature(make_tokenizer, make_dummy_inputs, dims_single):
    d = dims_single
    tok = make_tokenizer(d["d_numerical"], d["categories"], d["d_token"])
    x_num, x_cat_oh, _, _ = make_dummy_inputs(d["d_numerical"], d["categories"], d["batch_size"])
    out = tok(x_num, x_cat_oh)
    expected_seq = 1 + d["d_numerical"] + len(d["categories"])
    assert out.shape == (d["batch_size"], expected_seq, d["d_token"])


def test_n_tokens_property(make_tokenizer, dims):
    tok = make_tokenizer(dims["d_numerical"], dims["categories"], dims["d_token"])
    expected = dims["d_numerical"] + 1 + len(dims["categories"])
    assert tok.n_tokens == expected


def test_n_tokens_numerical_only(make_tokenizer, dims_numerical_only):
    d = dims_numerical_only
    tok = make_tokenizer(d["d_numerical"], d["categories"], d["d_token"])
    assert tok.n_tokens == d["d_numerical"] + 1


def test_cls_token_position(make_tokenizer, make_dummy_inputs, dims):
    tok = make_tokenizer(dims["d_numerical"], dims["categories"], dims["d_token"], bias=False)
    x_num, x_cat_oh, _, _ = make_dummy_inputs(dims["d_numerical"], dims["categories"], dims["batch_size"])
    out = tok(x_num, x_cat_oh)
    # CLS token: ones * weight[0], so all batch rows should have the same CLS token
    cls_tokens = out[:, 0, :]
    assert torch.allclose(cls_tokens[0], cls_tokens[1])
    assert torch.allclose(cls_tokens[0], tok.weight[0])


def test_bias_vs_no_bias(make_tokenizer, make_dummy_inputs, dims):
    d = dims
    tok_bias = make_tokenizer(d["d_numerical"], d["categories"], d["d_token"], bias=True)
    tok_no_bias = make_tokenizer(d["d_numerical"], d["categories"], d["d_token"], bias=False)
    assert tok_bias.bias is not None
    assert tok_no_bias.bias is None


def test_category_offsets_values(make_tokenizer):
    cats = np.array([3, 5, 2])
    tok = make_tokenizer(4, cats, 16)
    assert torch.equal(tok.category_offsets, torch.tensor([0, 3, 8]))
    assert torch.equal(tok.category_ends, torch.tensor([3, 8, 10]))


def test_cat_weight_shape(make_tokenizer, dims):
    tok = make_tokenizer(dims["d_numerical"], dims["categories"], dims["d_token"])
    assert tok.cat_weight.shape == (sum(dims["categories"]), dims["d_token"])


def test_weight_shape(make_tokenizer, dims):
    tok = make_tokenizer(dims["d_numerical"], dims["categories"], dims["d_token"])
    assert tok.weight.shape == (dims["d_numerical"] + 1, dims["d_token"])


def test_gradient_flows(make_tokenizer, make_dummy_inputs, dims):
    tok = make_tokenizer(dims["d_numerical"], dims["categories"], dims["d_token"])
    x_num, x_cat_oh, _, _ = make_dummy_inputs(dims["d_numerical"], dims["categories"], dims["batch_size"])
    out = tok(x_num, x_cat_oh)
    out.sum().backward()
    assert tok.weight.grad is not None and tok.weight.grad.abs().sum() > 0
    assert tok.cat_weight.grad is not None and tok.cat_weight.grad.abs().sum() > 0
    assert tok.bias.grad is not None and tok.bias.grad.abs().sum() > 0
