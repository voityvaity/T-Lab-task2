"""Train-only baseline and reward-uncertainty invariants."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from tlab2.baselines import PopularityBaseline, SequentialRulesBaseline, SmoothedTransitionBaseline
from tlab2.data import Examples
from tlab2.models import RewardModel, make_gpt2, reinforce_rollout_step, reinforce_step, shaped_reward, trainable_actor


def example_data():
    return Examples(
        context=np.asarray([[0, 0, 2, 3], [0, 0, 2, 3],
                            [0, 0, 5, 6], [0, 0, 2, 3]], dtype=np.int32),
        target=np.asarray([4, 4, 7, 4], dtype=np.int64),
        completion=np.asarray([.8, .9, .6, .5], dtype=np.float32),
        user=np.asarray([1, 2, 3, 4], dtype=np.int64),
        timestamp=np.asarray([1, 1, 1, 1], dtype=np.int64),
    )


def test_seqrules_outscores_unseen_transition_and_is_deterministic():
    counts = np.asarray([0, 0, 10, 10, 3, 5, 5, 1, 1, 1, 1, 1])
    train = example_data()
    pop = PopularityBaseline(counts)
    rules = SequentialRulesBaseline(train, counts)
    x = torch.tensor([[0, 0, 2, 3], [0, 0, 5, 6]])
    p = pop.score(x)
    s = rules.score(x)
    assert tuple(p.shape) == tuple(s.shape) == (2, len(counts))
    assert torch.equal(s, rules.score(x))
    assert s[0, 4] > s[0, 7]
    assert s[1, 7] > s[1, 4]
    assert s[0, 0] < -100000 and s[0, 1] < -100000


def test_invalid_rule_history_raises():
    with pytest.raises(ValueError):
        SequentialRulesBaseline(example_data(), np.ones(12), history=0)


def test_conservative_reward_disagreement_penalty():
    context = torch.tensor([[0, 2, 3], [0, 5, 6]])
    action = torch.tensor([7, 8])
    counts = torch.ones(12)
    durations = torch.ones(12) * 90
    first = torch.tensor([.8, .8])
    second = torch.tensor([.8, .2])
    robust = shaped_reward(first, action, context, counts, durations,
                           kind="conservative", second_completion=second)
    assert torch.isclose(robust[0], torch.tensor(.8))
    assert robust[1] < .5
    with pytest.raises(ValueError):
        shaped_reward(first, action, context, counts, durations, kind="conservative")


def test_conservative_rl_uses_uncertainty_reward_head():
    model = make_gpt2(vocab_size=16, context_length=5, hidden=32, layers=1, heads=4)
    actor = trainable_actor(model)
    reward = RewardModel(model, hidden=32)
    second = RewardModel(model, hidden=32)
    x = torch.tensor([[0, 0, 2, 3, 4], [0, 3, 4, 5, 6],
                      [0, 2, 4, 6, 8]])
    counts = torch.ones(16)
    durations = torch.ones(16) * 200
    torch.manual_seed(7)
    loss, summary = reinforce_step(actor, model, reward, x, counts, durations,
                                    "conservative", .2, uncertainty_model=second)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in actor.parameters())
    actor.zero_grad(set_to_none=True)
    loss, summary = reinforce_rollout_step(
        actor, model, reward, x, counts, durations,
        "conservative", .2, horizon=2, uncertainty_model=second
    )
    assert torch.isfinite(loss)
    assert summary["horizon"] == 2


def test_smoothed_multilag_transition_is_deterministic_and_train_only():
    training = example_data()
    counts = np.asarray([0, 0, 10, 10, 3, 5, 5, 1, 1, 1, 1, 1], dtype=np.int64)
    model = SmoothedTransitionBaseline(training, counts, max_lag=3)
    x = torch.tensor([[0, 0, 2, 3], [0, 0, 5, 6]])
    scores = model.score(x)
    assert torch.equal(scores, model.score(x))
    assert scores.shape == (2, len(counts))
    assert scores[0, 4] > scores[0, 7]
    assert scores[1, 7] > scores[1, 4]
    assert scores[0, 0] < -100000
    for kwargs in ({"max_lag": 0}, {"decay": 0}, {"smoothing": 0},
                   {"pop_boost": -1}):
        with pytest.raises(ValueError):
            SmoothedTransitionBaseline(training, counts, **kwargs)
