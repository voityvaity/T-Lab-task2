"""Validate dense causal GPT-2 fit and TRAIN-only smoothed reward residuals."""
import numpy as np
import pytest

from tlab2.data import Examples
from tlab2.reward_priors import estimate_completion_prior

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from tlab2.models import RewardModel, make_gpt2
from tlab2.experiment import (dense_autoregressive_loss, reward_validation_mse,
                              train_reward, train_base, reward_quality)


def sample_data():
    contexts = np.asarray([[0, 0, 2, 3, 4],
                           [0, 2, 3, 4, 5],
                           [2, 3, 4, 5, 6],
                           [0, 0, 4, 5, 6]], dtype=np.int32)
    target_times = np.asarray([100, 200, 300, 400], dtype=np.int64)
    offsets = np.asarray([50, 40, 30, 20, 10], dtype=np.int64)
    context_times = np.where(contexts != 0, target_times[:, None] - offsets, -1)
    return Examples(
        context=contexts,
        target=np.asarray([5, 6, 7, 7], dtype=np.int64),
        completion=np.asarray([.1, .9, .8, .75], dtype=np.float32),
        user=np.asarray([1, 1, 2, 2], dtype=np.int64),
        timestamp=target_times,
        context_timestamps=context_times,
    )


def test_dense_loss_uses_all_valid_causal_transitions_and_gradients():
    torch.manual_seed(42)
    model = make_gpt2(vocab_size=14, context_length=5, hidden=32,
                      layers=1, heads=4)
    x = torch.tensor([[0, 0, 2, 3, 4],
                      [0, 0, 1, 4, 5]], dtype=torch.long)
    y = torch.tensor([5, 6], dtype=torch.long)
    model.train()
    times = torch.tensor([[-1, -1, 10, 20, 30],
                          [-1, -1, 10, 20, 30]], dtype=torch.long)
    end_times = torch.tensor([40, 40], dtype=torch.long)
    loss, stats = dense_autoregressive_loss(
        model, x, y, dense_weight=.75,
        context_timestamps=times, target_timestamps=end_times,
    )
    assert torch.isfinite(loss)
    # Three valid predicted targets per sequence, including one predicted
    # from an OOV input token. OOV is excluded only as a TARGET.
    assert stats["causal_transitions"] == 6
    assert stats["dense_ce"] > 0
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters())


def test_dense_supervised_training_selects_valid_checkpoint():
    torch.manual_seed(4)
    x = sample_data()
    model = make_gpt2(vocab_size=14, context_length=5, hidden=32,
                      layers=1, heads=4)
    logs = train_base(model, x, steps=3, batch_size=2, device=torch.device("cpu"),
                      learning_rate=.001, seed=42, validation=x,
                      dense_weight=.6, patience=3)
    assert any(row.get("checkpoint_selection") == "minimum_in_vocab_validation_ce"
               for row in logs)
    assert any(row.get("causal_transitions", 0) > 2 for row in logs)
    with pytest.raises(ValueError):
        dense_autoregressive_loss(model,
                                  torch.tensor(x.context[:2].copy(), dtype=torch.long),
                                  torch.tensor(x.target[:2].copy(), dtype=torch.long), -0.1)


def test_empirical_bayes_prior_shrinks_sparse_items_without_future_labels():
    train = sample_data()
    a = estimate_completion_prior(train, vocab_size=14, smoothing=4)
    assert a.item_mean.shape == (14,)
    assert a.item_count[7] == 2
    assert a.item_mean[7] > a.global_mean  # more successful than global
    assert a.item_mean[13] == pytest.approx(a.global_mean)
    assert a.item_mean[0] == pytest.approx(a.global_mean)
    assert a.global_mean == pytest.approx(float(train.completion.mean()))
    changed_future = Examples(train.context.copy(), train.target.copy(),
                              np.full(len(train), 0.99, dtype=np.float32),
                              train.user.copy(), train.timestamp.copy())
    # Deliberately construct a future-only alternative without touching train.
    future = changed_future
    assert estimate_completion_prior(train, 14, smoothing=4).item_mean[7] == a.item_mean[7]
    assert estimate_completion_prior(future, 14, smoothing=4).item_mean[7] != a.item_mean[7]
    with pytest.raises(ValueError):
        estimate_completion_prior(train, 14, smoothing=0)


def test_reward_head_initially_equals_own_train_item_prior():
    train = sample_data()
    prior = estimate_completion_prior(train, vocab_size=14, smoothing=5)
    model = make_gpt2(vocab_size=14, context_length=5, hidden=32,
                      layers=1, heads=4)
    rm = RewardModel(model, hidden=32, item_prior=prior.item_mean,
                     item_count=prior.item_count, item_duration=np.ones(14) * 180)
    x = torch.tensor(train.context[:2].copy(), dtype=torch.long)
    one = torch.tensor([5, 7], dtype=torch.long)
    two = torch.tensor([[5, 6], [7, 13]], dtype=torch.long)
    rm.eval()
    assert torch.allclose(rm(x, one), rm.prior(one), atol=1e-6)
    assert torch.allclose(rm(x, two), rm.prior(two), atol=1e-6)
    assert rm(x, two).shape == (2, 2)
    assert not any(p.requires_grad for p in model.parameters())


def test_reward_fitting_restores_best_observed_validation_mse():
    train = sample_data()
    prior = estimate_completion_prior(train, vocab_size=14, smoothing=4)
    base = make_gpt2(vocab_size=14, context_length=5, hidden=32,
                     layers=1, heads=4)
    rm = RewardModel(base, hidden=32, item_prior=prior.item_mean,
                     item_count=prior.item_count, item_duration=np.ones(14) * 180)
    dev = torch.device("cpu")
    baseline_mse = reward_validation_mse(rm, train, batch_size=2, device=dev)
    logs = train_reward(rm, train, steps=5, batch_size=2,
                        device=dev, learning_rate=.001, seed=14,
                        validation=train, patience=4)
    final = reward_validation_mse(rm, train, batch_size=2, device=dev)
    assert final <= baseline_mse + 1e-6
    assert any(row.get("checkpoint_selection") == "minimum_validation_mse" for row in logs)
    report = reward_quality(rm, train, prior.global_mean, batch_size=2,
                            device=dev, prior_median=prior.global_median)
    assert report["n_observed"] == 4
    assert report["smoothed_item_prior_rmse"] >= 0
    assert report["constant_median_mae"] is not None


def test_dense_loss_drops_all_tied_time_transitions_but_keeps_strictly_later() -> None:
    torch.manual_seed(7)
    base = make_gpt2(vocab_size=16, context_length=5, hidden=32,
                     layers=1, heads=4)
    x = torch.tensor([[0, 2, 3, 4, 5]], dtype=torch.long)
    y = torch.tensor([6], dtype=torch.long)
    # Tokens 2 and 3 share second=100; tokens 4 and 5 share second=105.
    times = torch.tensor([[-1, 100, 100, 105, 105]], dtype=torch.long)
    ts = torch.tensor([110], dtype=torch.long)
    base.eval()
    loss, details = dense_autoregressive_loss(
        base, x, y, dense_weight=.6,
        context_timestamps=times, target_timestamps=ts
    )
    assert torch.isfinite(loss)
    assert details["causal_transitions"] == 2
    assert details["same_timestamp_transitions_excluded"] == 2
    # The unsafe implementation might have trained on two extra arbitrary
    # intra-bin next-item targets, and the test must catch that.
    with pytest.raises(ValueError, match="timestamp metadata"):
        dense_autoregressive_loss(base, x, y, dense_weight=.6)
    # No metadata is needed when fitting ONLY the supervised final target.
    next_only, meta = dense_autoregressive_loss(base, x, y, dense_weight=0.0)
    assert torch.isfinite(next_only)
    assert meta["causal_transitions"] == 0


def test_same_time_targets_in_train_examples_preserve_original_mask(tmp_path):
    import pandas as pd
    from tlab2.data import DataConfig, prepare
    rows = []
    for user in (1, 2, 3, 4):
        for timestamp in range(0, 105, 5):
            rows.append(dict(uid=user, item_id=100 + timestamp // 5 % 6,
                             timestamp=timestamp, played_ratio_pct=70,
                             track_length_seconds=100))
            if timestamp == 50:
                rows.append(dict(uid=user, item_id=103,
                                 timestamp=timestamp, played_ratio_pct=95,
                                 track_length_seconds=100))
    p = tmp_path / "ties.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    prepared = prepare(p, DataConfig(users=4, vocab_size=16, context_length=8,
                                      train_events=500, eval_events_per_user=5))
    assert prepared.train.context_timestamps is not None
    assert np.all(np.where(prepared.train.context == 0,
                           prepared.train.context_timestamps == -1,
                           prepared.train.context_timestamps < prepared.train.timestamp[:, None]))
    same_time = (np.diff(prepared.train.context_timestamps, axis=1) == 0)
    both_real = ((prepared.train.context[:, 1:] != 0) &
                 (prepared.train.context[:, :-1] != 0))
    ties = np.any(same_time & both_real, axis=1)
    assert ties.any()
    idx = np.flatnonzero(ties)[0]
    model = make_gpt2(vocab_size=len(prepared.train_counts), context_length=8,
                      hidden=32, layers=1, heads=4)
    _, diag = dense_autoregressive_loss(
        model, torch.tensor(prepared.train.context[idx:idx + 1], dtype=torch.long),
        torch.tensor(prepared.train.target[idx:idx + 1], dtype=torch.long),
        dense_weight=.75,
        context_timestamps=torch.tensor(prepared.train.context_timestamps[idx:idx + 1], dtype=torch.long),
        target_timestamps=torch.tensor(prepared.train.timestamp[idx:idx + 1], dtype=torch.long))
    assert diag["same_timestamp_transitions_excluded"] >= 1
