import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from tlab2.data import Examples
from tlab2.experiment import bootstrap_delta, evaluate, reward_quality
from tlab2.models import RewardModel, make_gpt2, reinforce_step, shaped_reward, trainable_actor


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
