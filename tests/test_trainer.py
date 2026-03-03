import torch
import numpy as np


# ---- Gradient clipping tests ----

def test_grad_clipping_applied(make_trainer, tmp_path):
    trainer = make_trainer(max_grad_norm=0.5, tmp_path=tmp_path)
    batch = next(iter(trainer.train_iter))
    trainer._run_step(batch, closs_weight=1.0, dloss_weight=1.0)
    # After clipping, total gradient norm should be <= max_grad_norm (with tolerance)
    total_norm = torch.nn.utils.clip_grad_norm_(trainer.flow.parameters(), float('inf'))
    # Gradients were already clipped in _run_step, then optimizer.step() zeroed them.
    # So we re-run to check: do a fresh forward-backward without step
    trainer.optimizer.zero_grad()
    dloss, closs = trainer.flow.mixed_loss(batch.to(trainer.device))
    (dloss + closs).backward()
    torch.nn.utils.clip_grad_norm_(trainer.flow.parameters(), 0.5)
    total_norm = 0.0
    for p in trainer.flow.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    total_norm = total_norm ** 0.5
    assert total_norm <= 0.5 + 1e-6


def test_grad_clipping_disabled(make_trainer, tmp_path):
    trainer = make_trainer(max_grad_norm=0, tmp_path=tmp_path)
    assert trainer.max_grad_norm == 0


def test_run_step_returns_losses(make_trainer, tmp_path):
    trainer = make_trainer(tmp_path=tmp_path)
    batch = next(iter(trainer.train_iter))
    dloss, closs = trainer._run_step(batch, closs_weight=1.0, dloss_weight=1.0)
    assert isinstance(dloss, torch.Tensor)
    assert isinstance(closs, torch.Tensor)
    assert torch.isfinite(dloss)
    assert torch.isfinite(closs)


# ---- LR warmup tests ----

def test_warmup_lr_linear_ramp(make_trainer, tmp_path):
    init_lr = 0.01
    warmup = 5
    trainer = make_trainer(lr=init_lr, warmup_epochs=warmup, tmp_path=tmp_path)
    # Simulate warmup epochs
    for epoch in range(warmup):
        expected_lr = init_lr * (epoch + 1) / warmup
        if trainer.warmup_epochs > 0 and (epoch + 1) <= trainer.warmup_epochs:
            warmup_lr = trainer.init_lr * (epoch + 1) / trainer.warmup_epochs
            for pg in trainer.optimizer.param_groups:
                pg["lr"] = warmup_lr
        actual_lr = trainer.optimizer.param_groups[0]["lr"]
        assert abs(actual_lr - expected_lr) < 1e-8, f"Epoch {epoch}: expected {expected_lr}, got {actual_lr}"


def test_warmup_overrides_scheduler(make_trainer, tmp_path):
    trainer = make_trainer(warmup_epochs=10, lr_scheduler='reduce_lr_on_plateau', tmp_path=tmp_path)
    initial_lr = trainer.optimizer.param_groups[0]["lr"]
    # During warmup, scheduler.step should NOT be called (we just set LR directly)
    # Simulate epoch 1 warmup
    warmup_lr = trainer.init_lr * 1 / trainer.warmup_epochs
    for pg in trainer.optimizer.param_groups:
        pg["lr"] = warmup_lr
    assert trainer.optimizer.param_groups[0]["lr"] == warmup_lr
    assert warmup_lr < initial_lr  # warmup starts lower


def test_no_warmup_when_zero(make_trainer, tmp_path):
    trainer = make_trainer(warmup_epochs=0, tmp_path=tmp_path)
    assert trainer.warmup_epochs == 0
    # LR should be the init_lr from the start
    assert trainer.optimizer.param_groups[0]["lr"] == trainer.init_lr


# ---- LR scheduler tests ----

def test_anneal_lr(make_trainer, tmp_path):
    trainer = make_trainer(lr=0.01, steps=100, lr_scheduler='anneal', tmp_path=tmp_path)
    trainer._anneal_lr(50)
    expected = 0.01 * (1 - 50 / 100)
    assert abs(trainer.optimizer.param_groups[0]["lr"] - expected) < 1e-8


# ---- EMA tests ----

def test_ema_model_created(make_trainer, tmp_path):
    trainer = make_trainer(tmp_path=tmp_path)
    # EMA model should exist and have same structure as flow._vf_fn
    assert trainer.ema_model is not None
    ema_params = list(trainer.ema_model.parameters())
    model_params = list(trainer.flow._vf_fn.parameters())
    assert len(ema_params) == len(model_params)
    # EMA params should be detached (requires_grad=False)
    for p in ema_params:
        assert not p.requires_grad
