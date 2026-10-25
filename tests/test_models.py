import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from tlab2.data import Examples
from tlab2.experiment import bootstrap_delta, evaluate, reward_quality
from tlab2.models import RewardModel, _history_mask_and_positions, make_gpt2, reinforce_step, reinforce_rollout_step, shaped_reward, trainable_actor
from tlab2.baselines import PopularityBaseline


def test_model_is_gpt2_and_reward_is_conditioned() -> None:
    model = make_gpt2(vocab_size=16, context_length=5, hidden=32, layers=1, heads=4)
    assert model.config.model_type == "gpt2"
    x = torch.tensor([[0, 0, 2, 3, 4], [0, 3, 4, 5, 6]])
    y = torch.tensor([7, 8])
    rm = RewardModel(model, hidden=32)
    value = rm(x, y)
    many = rm(x, torch.tensor([[7, 8, 9], [6, 7, 8]]))
    assert value.shape == (2,)
    assert many.shape == (2, 3)
    assert torch.all((value >= 0) & (value <= 1))
    assert not any(p.requires_grad for p in model.parameters())


def test_rl_loss_is_finite_and_gradients_exist() -> None:
    model = make_gpt2(vocab_size=16, context_length=5, hidden=32, layers=1, heads=4)
    reward = RewardModel(model, hidden=32)
    actor = trainable_actor(model)
    actor.eval()
    x = torch.tensor([[0, 0, 2, 3, 4], [0, 3, 4, 5, 6],
                      [0, 2, 4, 6, 8], [2, 3, 4, 5, 6]])
    counts = torch.ones(16)
    durations = torch.full((16,), 180.)
    torch.manual_seed(42)
    loss, report = reinforce_step(actor, model, reward, x, counts, durations,
                                  kind="balanced", kl_beta=0.2)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in actor.parameters())
    assert np.isfinite(report["sampled_reward"])
    assert report["kl"] >= -1e-5


def test_reward_variants_have_declared_behavior() -> None:
    x = torch.tensor([[0, 0, 2, 3], [0, 2, 3, 4]])
    a = torch.tensor([3, 5])
    score = torch.tensor([0.8, 0.8])
    counts = torch.tensor([0, 0, 1, 100, 1, 1], dtype=torch.float32)
    durations = torch.tensor([0, 0, 60, 60, 60, 300], dtype=torch.float32)
    watch = shaped_reward(score, a, x, counts, durations, "watch_time")
    assert torch.all(watch <= score)
    balanced = shaped_reward(score, a, x, counts, durations, "balanced")
    assert balanced[0] < balanced[1]
    with pytest.raises(ValueError):
        shaped_reward(score, a, x, counts, durations, "unsupported")


def test_bootstrap_is_user_clustered() -> None:
    baseline = np.array([0., 0., 0.])
    rl = np.array([0.1, 0.1, 0.4])
    uids = np.array([7, 7, 8])
    result = bootstrap_delta(rl, baseline, uids, seed=42, draws=100)
    assert np.isclose(result["mean"], 0.2)
    assert result["user_clusters"] == 2
    assert len(result["ci_95"]) == 2


def test_multi_step_reinforce_returns_finite_gradients() -> None:
    model = make_gpt2(vocab_size=16, context_length=5, hidden=32, layers=1, heads=4)
    rm = RewardModel(model, hidden=32)
    actor = trainable_actor(model)
    actor.eval()
    x = torch.tensor([[0, 0, 2, 3, 4], [0, 3, 4, 5, 6], [0, 2, 4, 6, 8]])
    counts = torch.ones(16)
    durations = torch.full((16,), 200.)
    torch.manual_seed(42)
    loss, metrics = reinforce_rollout_step(actor, model, rm, x, counts, durations,
                                           "completion", kl_beta=0.1, horizon=2)
    assert torch.isfinite(loss)
    loss.backward()
    assert metrics["horizon"] == 2
    assert any(p.grad is not None for p in actor.parameters())
    with pytest.raises(ValueError):
        reinforce_rollout_step(actor, model, rm, x, counts, durations,
                               "completion", kl_beta=0.1, horizon=0)


def test_left_padding_gpt2_position_ids_exclude_padding() -> None:
    history = torch.tensor([[0, 0, 2, 3, 4], [0, 5, 6, 7, 8]])
    mask, pos = _history_mask_and_positions(history)
    assert mask.tolist() == [[0, 0, 1, 1, 1], [0, 1, 1, 1, 1]]
    assert pos.tolist() == [[0, 0, 0, 1, 2], [0, 0, 1, 2, 3]]


def test_out_of_catalog_next_items_are_misses_not_discarded() -> None:
    from tlab2.experiment import evaluate
    model = make_gpt2(vocab_size=12, context_length=5, hidden=32, layers=1, heads=4)
    model.eval()
    rm = RewardModel(model, hidden=32)
    # Item 1 is OOV and cannot be recommended by any model.
    sample = Examples(
        np.asarray([[0, 0, 2, 3, 4]] * 3, dtype=np.int32),
        np.asarray([2, 1, 2], dtype=np.int64),
        np.asarray([.6, .8, .9], dtype=np.float32),
        np.asarray([8, 8, 9], dtype=np.int64),
        np.asarray([100, 200, 300], dtype=np.int64))
    counts = torch.ones(12)
    counts[2] = 50
    policy = PopularityBaseline(counts.numpy())
    result, per_query = evaluate(policy, sample, rm, rm, counts, torch.ones(12) * 200,
                                 torch.device("cpu"), batch_size=2)
    assert result["queries"] == 3
    assert result["in_vocab_targets"] == 2
    assert result["target_catalog_coverage"] == pytest.approx(2 / 3)
    assert len(per_query) == 3
    assert per_query[1] == 0
    assert result["recall_at_10"] == pytest.approx(2 / 3)
    assert result["conditional_recall_at_10"] == pytest.approx(1.0)


def test_supervised_validation_loss_ignores_oov_target() -> None:
    from tlab2.experiment import supervised_validation_loss, train_base
    model = make_gpt2(vocab_size=16, context_length=5, hidden=32, layers=1, heads=4)
    train = Examples(
        np.asarray([[0, 0, 2, 3, 4], [0, 2, 3, 4, 5]], dtype=np.int32),
        np.asarray([6, 7], dtype=np.int64),
        np.asarray([.7, .8], dtype=np.float32),
        np.asarray([1, 2], dtype=np.int64),
        np.asarray([1, 2], dtype=np.int64))
    validation = Examples(
        train.context.copy(), np.asarray([1, 7], dtype=np.int64),
        train.completion.copy(), train.user.copy(), train.timestamp.copy())
    metric = supervised_validation_loss(model, validation, batch_size=2,
                                        device=torch.device("cpu"))
    assert np.isfinite(metric)
    logs = train_base(model, train, steps=2, batch_size=2,
                      device=torch.device("cpu"), learning_rate=0.001,
                      seed=42, validation=validation)
    assert any(row.get("checkpoint_selection") == "minimum_in_vocab_validation_ce"
               for row in logs)
    assert model.training is False
