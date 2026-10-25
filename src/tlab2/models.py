"""GPT-2 recommender, observed-feedback reward model, and constrained REINFORCE.

Importing this module requires installing the optional [train] dependencies.
"""
from __future__ import annotations

from copy import deepcopy
import math

import torch
from torch import nn
from torch.nn import functional as F
from transformers import GPT2Config, GPT2LMHeadModel


def make_gpt2(vocab_size: int, context_length: int, hidden: int = 96,
              layers: int = 2, heads: int = 4) -> GPT2LMHeadModel:
    """Uninitialized genuine GPT-2 decoder with tied input/output item embeddings."""
    if hidden % heads:
        raise ValueError("hidden must be divisible by heads")
    conf = GPT2Config(
        vocab_size=vocab_size, n_positions=context_length + 8, n_ctx=context_length + 8,
        n_embd=hidden, n_layer=layers, n_head=heads,
        bos_token_id=1, eos_token_id=1, pad_token_id=0,
        resid_pdrop=0.1, embd_pdrop=0.1, attn_pdrop=0.1,
    )
    return GPT2LMHeadModel(conf)


def logits_for_next(policy: GPT2LMHeadModel, context: torch.Tensor) -> torch.Tensor:
    """Predict the next token from a left-padded real-event history."""
    attention_mask = (context != 0).long()
    result = policy(input_ids=context, attention_mask=attention_mask, return_dict=True)
    return result.logits[:, -1, :]


def mask_special(logits: torch.Tensor) -> torch.Tensor:
    masked = logits.clone()
    masked[..., :2] = torch.finfo(logits.dtype).min
    return masked


class RewardModel(nn.Module):
    """Observed completion ratio, conditional on frozen GPT-2 history/item features."""

    def __init__(self, frozen_gpt2: GPT2LMHeadModel, hidden: int):
        super().__init__()
        self.encoder = frozen_gpt2
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        self.head = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.GELU(),
                                  nn.Linear(hidden, hidden // 2), nn.GELU(),
                                  nn.Linear(hidden // 2, 1))

    def train(self, mode: bool = True):
        super().train(mode)
        # Remain frozen/deterministic even when the reward MLP is training.
        self.encoder.eval()
        return self

    def forward(self, context: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        """item shape (B,) or (B, M), output corresponding (B,) or (B, M)."""
        with torch.no_grad():
            hidden = self.encoder.transformer(
                input_ids=context, attention_mask=(context != 0).long(), return_dict=True
            ).last_hidden_state[:, -1, :].detach()
            item_embedding = self.encoder.transformer.wte(item).detach()
        if item.ndim == 2:
            hidden = hidden.unsqueeze(1).expand_as(item_embedding)
        x = torch.cat((hidden, item_embedding, hidden * item_embedding), dim=-1)
        return self.head(x).squeeze(-1).sigmoid()


def shaped_reward(pred_completion: torch.Tensor, actions: torch.Tensor,
                  context: torch.Tensor, counts: torch.Tensor, durations: torch.Tensor,
                  kind: str = "completion") -> torch.Tensor:
    """All reward statistics are computed from training data only."""
    if kind == "completion":
        return pred_completion
    if kind == "watch_time":
        return pred_completion * durations[actions].clamp(min=0, max=600) / 600.0
    if kind == "balanced":
        # Penalize items already in past context; boost unfamiliar but in-catalog items.
        repeats = (context == actions[:, None]).any(dim=1).float()
        max_pop = torch.log1p(counts[2:].max()).clamp(min=1)
        normalized_pop = torch.log1p(counts[actions].float()) / max_pop
        return pred_completion + 0.10 * (1 - repeats) - 0.15 * repeats - 0.05 * normalized_pop
    raise ValueError(f"Unknown reward kind: {kind}")


def reinforce_step(actor: GPT2LMHeadModel, reference: GPT2LMHeadModel,
                   reward_model: RewardModel, history: torch.Tensor,
                   counts: torch.Tensor, durations: torch.Tensor,
                   kind: str, kl_beta: float) -> tuple[torch.Tensor, dict]:
    """On-policy one-step REINFORCE; exact KL over admissible item vocabulary.

    Sampled reward is detached (fixed simulator); batch-centered REINFORCE
    advantage reduces variance. Does not pretend unobserved items are negatives.
    """
    logits = mask_special(logits_for_next(actor, history))
    with torch.no_grad():
        ref_logits = mask_special(logits_for_next(reference, history))
    logpi = F.log_softmax(logits, dim=-1)
    logref = F.log_softmax(ref_logits, dim=-1)
    probs = logpi.exp()
    actions = torch.multinomial(probs.detach(), num_samples=1).squeeze(1)
    selected = logpi.gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        completion = reward_model(history, actions)
        reward = shaped_reward(completion, actions, history, counts, durations, kind)
    # Leave-one-out batch baseline does not depend on an action's own reward.
    if reward.numel() > 1:
        baseline = (reward.sum() - reward) / (reward.numel() - 1)
    else:
        baseline = reward.detach()
    advantage = reward - baseline
    policy_loss = -(advantage.detach() * selected).mean()
    kl = (probs * (logpi - logref)).sum(dim=-1).mean()
    loss = policy_loss + kl_beta * kl
    return loss, {"sampled_reward": float(reward.mean().item()),
                  "pred_completion": float(completion.mean().item()),
                  "kl": float(kl.item()), "policy_loss": float(policy_loss.item())}


def reinforce_rollout_step(actor: GPT2LMHeadModel, reference: GPT2LMHeadModel,
                           reward_model: RewardModel, history: torch.Tensor,
                           counts: torch.Tensor, durations: torch.Tensor,
                           kind: str, kl_beta: float, horizon: int = 5,
                           gamma: float = 0.9) -> tuple[torch.Tensor, dict]:
    """Policy-gradient on hypothetical recommendations, not observed transitions.

    A generated action is appended to the context as if it were consumed. The
    feedback model has never observed those counterfactual transitions. This
    procedure deliberately exposes distribution shift, and cannot establish
    improved real-world multi-step reward without online experiments.
    """
    if horizon < 1 or not (0 <= gamma <= 1):
        raise ValueError("horizon must be >= 1 and gamma in [0,1]")
    context = history
    log_selected = []
    rewards = []
    kls = []
    repeated = []
    for _ in range(horizon):
        logit = mask_special(logits_for_next(actor, context))
        with torch.no_grad():
            ref_logit = mask_special(logits_for_next(reference, context))
        logpi = F.log_softmax(logit, dim=-1)
        logref = F.log_softmax(ref_logit, dim=-1)
        probs = logpi.exp()
        action = torch.multinomial(probs.detach(), 1).squeeze(1)
        selected = logpi.gather(1, action.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            completion = reward_model(context, action)
            r = shaped_reward(completion, action, context, counts, durations, kind)
            is_repeat = (context == action[:, None]).any(dim=1).float()
        log_selected.append(selected)
        rewards.append(r)
        repeated.append(is_repeat)
        kls.append((probs * (logpi - logref)).sum(-1))
        context = torch.cat((context[:, 1:], action.unsqueeze(1)), dim=1)

    returns = []
    accumulated = torch.zeros_like(rewards[0])
    for r in reversed(rewards):
        accumulated = r + gamma * accumulated
        returns.append(accumulated)
    returns.reverse()
    pg_losses = []
    for g, lp in zip(returns, log_selected):
        if g.numel() > 1:
            baseline = (g.sum() - g) / (g.numel() - 1)
        else:
            baseline = g.detach()
        pg_losses.append(-((g - baseline).detach() * lp).mean())
    pg = torch.stack(pg_losses).mean()
    kl = torch.stack([part.mean() for part in kls]).mean()
    loss = pg + kl_beta * kl
    return loss, {"sampled_reward": float(torch.stack(rewards).mean().item()),
                  "first_step_reward": float(rewards[0].mean().item()),
                  "discounted_first_return": float(returns[0].mean().item()),
                  "repeat_rate": float(torch.stack(repeated).mean().item()),
                  "kl": float(kl.item()), "policy_loss": float(pg.item()),
                  "horizon": horizon}


def trainable_actor(base: GPT2LMHeadModel) -> GPT2LMHeadModel:
    actor = deepcopy(base)
    actor.requires_grad_(True)
    return actor
