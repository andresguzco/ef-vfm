import pytest
import numpy as np
import torch
import torch.nn.functional as F
from unittest.mock import MagicMock


# --------------- dimension configs ---------------

@pytest.fixture
def dims():
    """Standard mixed-data dimensions."""
    return {"d_numerical": 4, "categories": np.array([3, 5, 2]), "batch_size": 8, "d_token": 16}


@pytest.fixture
def dims_numerical_only():
    """Numerical-only scenario (no categorical features)."""
    return {"d_numerical": 5, "categories": None, "batch_size": 8, "d_token": 16}


@pytest.fixture
def dims_single():
    """Minimal scenario: 1 numerical, 1 categorical with 2 classes."""
    return {"d_numerical": 1, "categories": np.array([2]), "batch_size": 4, "d_token": 8}


# --------------- dummy input factory ---------------

@pytest.fixture
def make_dummy_inputs():
    """Factory: returns (x_num, x_cat_onehot, x_cat_int, timesteps) from any dims."""
    def _make(d_numerical, categories, batch_size):
        torch.manual_seed(42)
        x_num = torch.randn(batch_size, d_numerical)
        if categories is not None and len(categories) > 0:
            cat_parts = []
            for k in categories:
                indices = torch.randint(0, k, (batch_size,))
                cat_parts.append(F.one_hot(indices, k).float())
            x_cat_onehot = torch.cat(cat_parts, dim=1)
            x_cat_int = torch.stack(
                [torch.randint(0, k, (batch_size,)) for k in categories], dim=1
            )
        else:
            x_cat_onehot = None
            x_cat_int = None
        timesteps = torch.rand(batch_size)
        return x_num, x_cat_onehot, x_cat_int, timesteps
    return _make


# --------------- model factories ---------------

@pytest.fixture
def make_tokenizer():
    from ef_vfm.modules.transformer import Tokenizer
    def _make(d_numerical, categories, d_token, bias=True):
        cats = list(categories) if categories is not None else None
        return Tokenizer(d_numerical, cats, d_token, bias)
    return _make


@pytest.fixture
def make_transformer():
    from ef_vfm.modules.transformer import Transformer
    def _make(d_token, n_layers=2, n_heads=1, d_ffn_factor=4, activation='gelu'):
        return Transformer(n_layers, d_token, n_heads, d_token, d_ffn_factor, activation=activation)
    return _make


@pytest.fixture
def make_reconstructor():
    from ef_vfm.modules.transformer import Reconstructor
    def _make(d_numerical, categories, d_token):
        cats = list(categories) if categories is not None else []
        return Reconstructor(d_numerical, cats, d_token)
    return _make


@pytest.fixture
def make_mlp():
    from ef_vfm.modules.main_modules import MLP
    def _make(d_in, dim_t=128, use_mlp=True):
        return MLP(d_in, dim_t=dim_t, use_mlp=use_mlp)
    return _make


@pytest.fixture
def make_unimodmlp():
    from ef_vfm.modules.main_modules import UniModMLP
    def _make(d_numerical, categories, d_token=16, n_layers=1, n_head=1,
              factor=4, dim_t=64, activation='gelu'):
        cats = list(categories) if categories is not None else []
        return UniModMLP(
            d_numerical, cats, n_layers, d_token,
            n_head=n_head, factor=factor, dim_t=dim_t, activation=activation,
        )
    return _make


@pytest.fixture
def make_flow_model():
    from ef_vfm.modules.main_modules import UniModMLP
    from ef_vfm.models.flow_model import ExpVFM
    def _make(d_numerical, categories, d_token=16, n_layers=1, dim_t=64):
        cats_list = list(categories) if categories is not None else []
        cats_np = np.array(cats_list)
        model = UniModMLP(
            d_numerical, cats_list, n_layers, d_token,
            n_head=1, factor=4, dim_t=dim_t, activation='gelu',
        )
        flow = ExpVFM(
            num_classes=cats_np,
            num_numerical_features=d_numerical,
            vf_fn=model,
            device=torch.device('cpu'),
        )
        return flow
    return _make


@pytest.fixture
def make_trainer():
    """Factory: creates a minimal Trainer with mocked external dependencies."""
    from ef_vfm.modules.main_modules import UniModMLP
    from ef_vfm.models.flow_model import ExpVFM
    from ef_vfm.trainer import Trainer

    def _make(d_numerical=4, categories=np.array([3, 5, 2]),
              lr=0.001, max_grad_norm=1.0, warmup_epochs=0,
              lr_scheduler='reduce_lr_on_plateau', steps=10, tmp_path=None):

        cats_list = list(categories) if categories is not None else []
        cats_np = np.array(cats_list)

        model = UniModMLP(
            d_numerical, cats_list, 1, 16,
            n_head=1, factor=4, dim_t=64, activation='gelu',
        )
        flow = ExpVFM(
            num_classes=cats_np,
            num_numerical_features=d_numerical,
            vf_fn=model,
            device=torch.device('cpu'),
        )

        # Build a small synthetic dataset: [N, d_num + len(cats)] with int cat indices
        n_samples = 32
        x_num = torch.randn(n_samples, d_numerical)
        if len(cats_list) > 0:
            x_cat = torch.stack(
                [torch.randint(0, k, (n_samples,)) for k in cats_list], dim=1
            ).float()
            data = torch.cat([x_num, x_cat], dim=1)
        else:
            data = x_num

        dataset = torch.utils.data.TensorDataset(data)
        train_iter = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)
        # DataLoader wraps in tuples; Trainer expects raw tensors, so use a wrapper
        class _UnwrapLoader:
            def __init__(self, loader):
                self._loader = loader
            def __iter__(self):
                for (batch,) in self._loader:
                    yield batch
            def __len__(self):
                return len(self._loader)

        save_path = str(tmp_path) if tmp_path else "/tmp"
        trainer = Trainer(
            flow=flow,
            train_iter=_UnwrapLoader(train_iter),
            dataset=MagicMock(),
            test_dataset=MagicMock(),
            metrics=MagicMock(),
            logger=MagicMock(),
            lr=lr,
            weight_decay=0,
            steps=steps,
            batch_size=8,
            check_val_every=steps + 1,  # never evaluate during test
            sample_batch_size=8,
            model_save_path=save_path,
            result_save_path=save_path,
            lr_scheduler=lr_scheduler,
            max_grad_norm=max_grad_norm,
            warmup_epochs=warmup_epochs,
            device=torch.device('cpu'),
        )
        return trainer
    return _make
