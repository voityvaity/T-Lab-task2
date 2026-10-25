"""Single command: supervised GPT-2, reward fit, KL-REINFORCE and fair comparison.

Results are exploratory offline proxies, not observed counterfactual outcomes.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import platform
import random
from typing import Iterable

import numpy as np
import torch
from torch.nn import functional as F

from tlab2.data import DataConfig, Examples, prepare, save_metadata
from tlab2.baselines import PopularityBaseline, SequentialRulesBaseline
from tlab2.models import (RewardModel, logits_for_next, make_gpt2, mask_special,
                          reinforce_step, reinforce_rollout_step, trainable_actor)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def batches(data: Examples, batch_size: int, device: torch.device,
            seed: int, shuffle: bool = True) -> Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    index = np.arange(len(data))
    if shuffle:
        np.random.default_rng(seed).shuffle(index)
    for offset in range(0, len(index), batch_size):
        ids = index[offset:offset + batch_size]
        yield (torch.as_tensor(data.context[ids], dtype=torch.long, device=device),
               torch.as_tensor(data.target[ids], dtype=torch.long, device=device),
               torch.as_tensor(data.completion[ids], dtype=torch.float32, device=device))


def train_base(policy, train: Examples, steps: int, batch_size: int, device, learning_rate: float,
               seed: int) -> list[dict]:
    policy.train()
    opt = torch.optim.AdamW(policy.parameters(), lr=learning_rate, weight_decay=0.01)
    logs = []
    iteration = 0
    while iteration < steps:
        for x, y, _ in batches(train, batch_size, device, seed + iteration):
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(mask_special(policy_logits(policy, x)), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            iteration += 1
            if iteration % max(1, steps // 10) == 0 or iteration == 1:
                logs.append({"step": iteration, "cross_entropy": float(loss.item())})
            if iteration >= steps:
                break
    policy.eval()
    return logs


def train_reward(model: RewardModel, sample: Examples, steps: int, batch_size: int,
                 device, learning_rate: float, seed: int) -> list[dict]:
    model.train()
    opt = torch.optim.AdamW(model.head.parameters(), lr=learning_rate)
    logs = []
    iteration = 0
    while iteration < steps:
        for x, y, label in batches(sample, batch_size, device, seed + iteration):
            opt.zero_grad(set_to_none=True)
            pred = model(x, y)
            loss = F.smooth_l1_loss(pred, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.head.parameters(), 1.0)
            opt.step()
            iteration += 1
            if iteration % max(1, steps // 10) == 0 or iteration == 1:
                logs.append({"step": iteration, "reward_huber": float(loss.item())})
            if iteration >= steps:
                break
    model.eval()
    return logs


def policy_logits(policy, context: torch.Tensor) -> torch.Tensor:
    """One score API for GPT-2 and both TRAIN-only non-neural baselines."""
    if hasattr(policy, "score"):
        return policy.score(context)
    return logits_for_next(policy, context)


@torch.no_grad()
def reward_quality(model: RewardModel, sample: Examples, prior_mean: float,
                   batch_size: int, device) -> dict:
    model.eval()
    predictions = []
    labels = []
    for x, y, label in batches(sample, batch_size, device, seed=0, shuffle=False):
        predictions.extend(model(x, y).cpu().tolist())
        labels.extend(label.cpu().tolist())
    pred = np.asarray(predictions)
    actual = np.asarray(labels)
    return {"mae": float(np.abs(pred - actual).mean()),
            "rmse": float(np.sqrt(np.square(pred - actual).mean())),
            "constant_mean_mae": float(np.abs(actual - prior_mean).mean()),
            "constant_mean_rmse": float(np.sqrt(np.square(actual - prior_mean).mean())),
            "mean_pred": float(pred.mean()), "mean_actual": float(actual.mean()),
            "n_observed": len(actual)}


@torch.no_grad()
def evaluate(policy, sample: Examples, reward: RewardModel, audit: RewardModel,
             counts: torch.Tensor, durations: torch.Tensor, device, batch_size: int,
             topk: int = 10, rerank_alpha: float | None = None,
             rerank_candidates: int = 50) -> tuple[dict, np.ndarray]:
    """One held-out observed next event per example; test ground truth is *not* impressions."""
    if hasattr(policy, "eval"):
        policy.eval()
    reward.eval()
    audit.eval()
    limit = min(topk, counts.numel() - 2)
    scores: list[float] = []
    hits: list[float] = []
    recommended = []
    repeated = []
    popularity = []
    lengths = []
    rm_pred = []
    rm_audit = []
    disagreement = []
    top1 = []
    expected_listened_seconds = []
    for x, y, _ in batches(sample, batch_size, device, seed=0, shuffle=False):
        logits = mask_special(policy_logits(policy, x))
        if rerank_alpha is not None:
            m = min(rerank_candidates, logits.size(-1) - 2)
            candidates = logits.topk(m, dim=-1).indices
            predicted = reward(x, candidates)
            original = F.log_softmax(logits, dim=-1).gather(1, candidates)
            final_order = (original + rerank_alpha * predicted).topk(limit, dim=-1).indices
            top = candidates.gather(1, final_order)
        else:
            top = logits.topk(limit, dim=-1).indices
        equality = top.eq(y.unsqueeze(1)).cpu().numpy()
        for row in equality:
            found = np.flatnonzero(row)
            hits.append(float(len(found) != 0))
            scores.append(float(1.0 / np.log2(int(found[0]) + 2)) if len(found) else 0.0)
        recommended.extend(top.cpu().tolist())
        top1.extend(top[:, 0].cpu().tolist())
        repeated.extend((top.unsqueeze(-1) == x.unsqueeze(1)).any(dim=-1).float().mean(dim=-1).cpu().tolist())
        popularity.extend(torch.log1p(counts[top].float()).mean(dim=-1).cpu().tolist())
        lengths.extend(durations[top].float().mean(dim=-1).cpu().tolist())
        # Reward is only a learned extrapolation for these recommended items.
        predicted = reward(x, top)
        rm_pred.extend(predicted.mean(dim=1).cpu().tolist())
        expected_listened_seconds.extend((predicted * durations[top].clamp(min=0, max=600)).mean(dim=1).cpu().tolist())
        second = audit(x, top)
        rm_audit.extend(second.mean(dim=1).cpu().tolist())
        disagreement.extend((predicted - second).abs().mean(dim=1).cpu().tolist())
    all_rec = np.asarray(recommended, dtype=np.int64)
    return ({
        "queries": len(scores), "recall_at_10": float(np.mean(hits)),
        "ndcg_at_10": float(np.mean(scores)),
        "predicted_completion_at_10": float(np.mean(rm_pred)),
        "predicted_capped_listen_seconds_at_10": float(np.mean(expected_listened_seconds)),
        "auditor_completion_at_10": float(np.mean(rm_audit)),
        "auditor_minus_optimized": float(np.mean(np.asarray(rm_audit) - np.asarray(rm_pred))),
        "rm_absolute_disagreement_at_10": float(np.mean(disagreement)),
        "most_common_top1_share": float(np.unique(top1, return_counts=True)[1].max() / max(1, len(top1))),
        "repeated_from_history_rate": float(np.mean(repeated)),
        "distinct_recommended_items": int(len(np.unique(all_rec))),
        "catalog_coverage": float(len(np.unique(all_rec)) / max(1, (counts > 0).sum().item())),
        "top10_mean_log_training_popularity": float(np.mean(popularity)),
        "top10_mean_seconds": float(np.mean(lengths)),
    }, np.asarray(scores))


@torch.no_grad()
def rollout(policy, sample: Examples, reward: RewardModel, device, batch_size: int,
            steps: int = 5, rerank_alpha: float | None = None) -> list[dict]:
    """Hypothetically feed recommended item back as if consumed: NOT real future listens."""
    policy.eval()
    out = [[] for _ in range(steps)]
    rec_seen = [[] for _ in range(steps)]
    for x, _, _ in batches(sample, batch_size, device, seed=0, shuffle=False):
        context = x.clone()
        slate = []
        for t in range(steps):
            scores = mask_special(policy_logits(policy, context))
            if rerank_alpha is None:
                action = scores.argmax(dim=-1)
            else:
                candidate = scores.topk(min(50, scores.shape[-1] - 2), dim=-1).indices
                adjusted = F.log_softmax(scores, dim=-1).gather(1, candidate) + rerank_alpha * reward(context, candidate)
                action = candidate.gather(1, adjusted.argmax(dim=-1, keepdim=True)).squeeze(1)
            out[t].extend(reward(context, action).cpu().tolist())
            seen = (context == action.unsqueeze(1)).any(dim=1)
            rec_seen[t].extend(seen.float().cpu().tolist())
            slate.append(action)
            context = torch.cat([context[:, 1:], action.unsqueeze(1)], dim=1)
    return [{"step": i + 1, "mean_predicted_completion": float(np.mean(out[i])),
             "repeat_of_context_rate": float(np.mean(rec_seen[i]))}
            for i in range(steps)]


@torch.no_grad()
def real_vs_hypothetical_rollout(policy, sample: Examples, device, reward: RewardModel,
                                 steps: int = 5, rerank_alpha: float | None = None) -> list[dict]:
    """Compare free-running fictional contexts with recorded real-event contexts.

    Selected eligible test events may have intervening OOV events. This is a
    descriptive distribution-drift diagnostic, not an estimate of user responses.
    """
    policy.eval()
    reward.eval()
    def predict(context: torch.Tensor) -> torch.Tensor:
        logits = mask_special(policy_logits(policy, context))
        if rerank_alpha is not None:
            top = logits.topk(min(50, logits.size(-1) - 2), dim=-1).indices
            adjusted = F.log_softmax(logits, dim=-1).gather(1, top) + rerank_alpha * reward(context, top)
            original = logits
            logits = torch.full_like(original, torch.finfo(original.dtype).min)
            logits.scatter_(1, top, adjusted)
        return F.softmax(logits, dim=-1)

    by_user = {}
    for i, uid in enumerate(sample.user.tolist()):
        by_user.setdefault(uid, []).append(i)
    js_by_step = [[] for _ in range(steps)]
    hit_by_step = [[] for _ in range(steps)]
    for idx in by_user.values():
        idx = sorted(idx, key=lambda j: int(sample.timestamp[j]))
        if len(idx) < steps:
            continue
        context = torch.as_tensor(sample.context[idx[0]][None, :], device=device, dtype=torch.long)
        for i in range(steps):
            real_context = torch.as_tensor(sample.context[idx[i]][None, :], device=device, dtype=torch.long)
            pred_free = predict(context)
            pred_real = predict(real_context)
            mixture = 0.5 * (pred_free + pred_real)
            js = 0.5 * (F.kl_div(mixture.clamp(min=1e-12).log(), pred_free, reduction="sum") +
                        F.kl_div(mixture.clamp(min=1e-12).log(), pred_real, reduction="sum"))
            js_by_step[i].append(float(js.item()))
            actual = int(sample.target[idx[i]])
            hit_by_step[i].append(float(actual in pred_free.topk(min(10, pred_free.size(-1) - 2)).indices[0].tolist()))
            generated = pred_free.argmax(dim=-1, keepdim=True)
            context = torch.cat([context[:, 1:], generated], dim=1)
    return [{"step": i + 1, "paired_trajectories": len(js_by_step[i]),
             "js_divergence_from_real_context": float(np.mean(js_by_step[i])) if js_by_step[i] else None,
             "free_rollout_real_next_hit_at_10": float(np.mean(hit_by_step[i])) if hit_by_step[i] else None}
            for i in range(steps)]


def bootstrap_delta(chosen: np.ndarray, baseline: np.ndarray, users: np.ndarray,
                    seed: int, draws: int = 400) -> dict:
    """Paired USER-cluster bootstrap (queries from the same user are dependent)."""
    delta = chosen - baseline
    unique = np.unique(users)
    groups = [delta[users == uid] for uid in unique]
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(draws):
        picked = rng.integers(0, len(groups), size=len(groups))
        total = sum(float(groups[j].sum()) for j in picked)
        count = sum(len(groups[j]) for j in picked)
        samples.append(total / count)
    return {"mean": float(delta.mean()),
            "ci_95": [float(q) for q in np.quantile(samples, [0.025, 0.975])],
            "bootstrap_unit": "user", "user_clusters": len(unique)}


def subset(data: Examples, mask: np.ndarray) -> Examples:
    return Examples(data.context[mask], data.target[mask], data.completion[mask],
                    data.user[mask], data.timestamp[mask])


def run(args) -> dict:
    seed_all(args.seed)
    torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    cfg = DataConfig(users=args.users, vocab_size=args.vocab, context_length=args.context,
                     train_events=args.train_events, seed=args.seed,
                     eval_events_per_user=args.eval_per_user)
    data = prepare(args.data, cfg)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    save_metadata(data, out)
    counts = torch.tensor(data.train_counts, dtype=torch.float32, device=device)
    durations = torch.tensor(data.train_duration, dtype=torch.float32, device=device)

    # Vocabulary is exactly the observed TRAIN subcatalog + PAD/OOV;
    # a fixed nominal cap can otherwise create recommendable phantom items.
    base = make_gpt2(len(data.train_counts), args.context, args.hidden,
                     args.layers, args.heads).to(device)
    logs = {"base": train_base(base, data.train, args.base_steps, args.batch, device,
                               args.base_lr, args.seed)}
    torch.save(base.state_dict(), out / "gpt2_base.pt")
    # Train the optimized reward and independent diagnostic head on DISJOINT TRAIN users.
    users = np.unique(data.train.user)
    permuted = np.random.default_rng(args.seed).permutation(users)
    if len(users) < 2:
        raise ValueError("Need at least 2 train users to make disjoint reward/audit subsets")
    cut = max(1, min(len(users) - 1, int(len(users) * 0.75)))
    optimized_mask = np.isin(data.train.user, permuted[:cut])
    auditor_mask = ~optimized_mask
    optim_train = subset(data.train, optimized_mask)
    auditor_train = subset(data.train, auditor_mask)
    if len(optim_train) == 0 or len(auditor_train) == 0:
        raise ValueError("No observed rewards in an independent reward model partition")
    reward = RewardModel(base, args.hidden).to(device)
    logs["reward"] = train_reward(reward, optim_train, args.reward_steps, args.batch,
                                  device, args.reward_lr, args.seed + 1)
    torch.save(reward.head.state_dict(), out / "reward_head.pt")
    # Second reward head with separate initialization and sample order. Both
    # heads see only the reward-training users; the audit head below remains
    # reserved for diagnostic comparison, and is not used to optimize RL.
    uncertainty = RewardModel(base, args.hidden).to(device)
    logs["uncertainty"] = train_reward(uncertainty, optim_train, args.reward_steps,
                                      args.batch, device, args.reward_lr, args.seed + 29)
    torch.save(uncertainty.head.state_dict(), out / "uncertainty_head.pt")
    audit = RewardModel(base, args.hidden).to(device)
    logs["auditor"] = train_reward(audit, auditor_train, args.reward_steps, args.batch,
                                   device, args.reward_lr, args.seed + 2)
    torch.save(audit.head.state_dict(), out / "auditor_head.pt")
    reward_metrics = {
        "optimized_valid": reward_quality(reward, data.valid, float(optim_train.completion.mean()), args.batch, device),
        "auditor_valid": reward_quality(audit, data.valid, float(auditor_train.completion.mean()), args.batch, device),
        "uncertainty_valid": reward_quality(uncertainty, data.valid, float(optim_train.completion.mean()), args.batch, device),
        "auditor_train_user_disjoint": True,
    }
    base_valid, _ = evaluate(base, data.valid, reward, audit, counts, durations, device, args.batch)
    candidates = {"gpt2": (base, None, None)}
    rerank_alphas = [float(x) for x in args.rerank_alphas.split(",")]
    for alpha in rerank_alphas:
        candidates["rerank_" + str(alpha)] = (base, alpha, None)

    rl_logs = {}
    for reward_kind in args.rewards.split(","):
        for beta_str in args.betas.split(","):
            beta = float(beta_str)
            for horizon_str in args.rl_horizons.split(","):
                horizon = int(horizon_str)
                tag = f"rl_{reward_kind}_h{horizon}_kl_{beta:g}"
                actor = trainable_actor(base).to(device)
                # Disable stochastic dropout while measuring exact policy/ref KL;
                # eval() changes dropout behavior, not gradient computation.
                actor.eval()
                optimizer = torch.optim.AdamW(actor.parameters(), lr=args.rl_lr)
                history = []
                step = 0
                while step < args.rl_steps:
                    for x, _, _ in batches(data.train, args.batch, device, args.seed + step):
                        optimizer.zero_grad(set_to_none=True)
                        if horizon == 1:
                            loss, metrics = reinforce_step(actor, base, reward, x, counts,
                                                           durations, reward_kind, beta,
                                                           uncertainty_model=uncertainty)
                        else:
                            loss, metrics = reinforce_rollout_step(actor, base, reward, x, counts,
                                                                   durations, reward_kind, beta,
                                                                   horizon=horizon,
                                                                   uncertainty_model=uncertainty)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                        optimizer.step()
                        step += 1
                        if step % max(1, args.rl_steps // 10) == 0 or step == 1:
                            history.append({"step": step, **metrics})
                        if step >= args.rl_steps:
                            break
                actor.eval()
                rl_logs[tag] = history
                candidates[tag] = (actor, None, reward_kind)
                torch.save(actor.state_dict(), out / (tag + ".pt"))
    logs["rl"] = rl_logs
    validation = {}
    for name, (policy, alpha, kind) in candidates.items():
        metrics, _ = evaluate(policy, data.valid, reward, audit, counts, durations, device,
                              args.batch, rerank_alpha=alpha)
        validation[name] = metrics

    # Freeze selection rule before test: best auditor-estimated completion with
    # validation NDCG not more than allowed_delta below supervised baseline.
    eligible = [(name, scores) for name, scores in validation.items()
                if scores["ndcg_at_10"] >= base_valid["ndcg_at_10"] - args.allowed_ndcg_drop]
    selected = max(eligible, key=lambda kv: kv[1]["auditor_completion_at_10"])[0] if eligible else "gpt2"
    # Simple, TRAIN-only controls; not eligible for RL configuration selection.
    simple = {
        "popularity": PopularityBaseline(data.train_counts),
        "sequential_rules": SequentialRulesBaseline(data.train, data.train_counts),
    }
    for name, model in simple.items():
        metrics, _ = evaluate(model, data.valid, reward, audit, counts, durations,
                              device, args.batch)
        validation[name] = metrics

    # All test evaluation happens after validation-only model selection.
    reward_metrics["optimized_test"] = reward_quality(reward, data.test, float(optim_train.completion.mean()), args.batch, device)
    reward_metrics["auditor_test"] = reward_quality(audit, data.test, float(auditor_train.completion.mean()), args.batch, device)
    reward_metrics["uncertainty_test"] = reward_quality(uncertainty, data.test, float(optim_train.completion.mean()), args.batch, device)
    test_results = {}
    test_ndcg = {}
    for name, (policy, alpha, kind) in candidates.items():
        result, ndcg = evaluate(policy, data.test, reward, audit, counts, durations, device,
                                args.batch, rerank_alpha=alpha)
        test_results[name] = result
        test_ndcg[name] = ndcg
    for name, model in simple.items():
        metrics, individual_ndcg = evaluate(model, data.test, reward, audit,
                                             counts, durations, device, args.batch)
        test_results[name] = metrics
        test_ndcg[name] = individual_ndcg
    test_deltas = {name: bootstrap_delta(v, test_ndcg["gpt2"], data.test.user, args.seed)
                   for name, v in test_ndcg.items() if name != "gpt2"}
    # Compare open-loop, hypothetical autoregressive five-item behavior.
    rollouts = {name: rollout(policy, data.test, reward, device, args.batch, steps=5, rerank_alpha=alpha)
                for name, (policy, alpha, kind) in candidates.items()
                if name == "gpt2" or name == selected or (name.startswith("rl_") and kind == "completion")}
    drift = {name: real_vs_hypothetical_rollout(policy, data.test, device, reward, rerank_alpha=alpha)
             for name, (policy, alpha, kind) in candidates.items()
             if name == "gpt2" or name == selected}
    results = {
        "status": ("synthetic_smoke_only" if args.source_kind == "synthetic"
                   else "measured_offline_model_based_not_causal"),
        "source_kind": args.source_kind,
        "selected_on_validation": selected,
        "selection_rule": f"max auditor completion subject to validation NDCG >= GPT2-{args.allowed_ndcg_drop}",
        "device": str(device),
        "seed": args.seed,
        "software": {"python": platform.python_version(), "torch": torch.__version__},
        "parameters": vars(args),
        "reward_quality_on_observed_events": reward_metrics,
        "observed_session_return_proxy_test_only": data.session_proxy,
        "validation": validation, "test": test_results,
        "test_paired_ndcg_deltas": test_deltas,
        "five_step_hypothetical_rollouts": rollouts,
        "real_context_vs_generated_context_drift": drift,
        "limitations": ["Top-N train-only catalog rather than entire Yambda",
                        "Observed-only feedback; no reliable counterfactual reward",
                        "Main/uncertainty heads share training users but vary initialization; auditor trained on held-out train users with shared frozen encoder",
                        "Generated items inserted hypothetically, without feedback simulation",
                        "Session-return associations are observational and do not identify policy-caused retention",
                        "All absolute metrics conditional on targets inside train vocabulary"]
    }
    (out / "experiment.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "training_logs.json").write_text(json.dumps(logs, indent=2, ensure_ascii=False), encoding="utf-8")
    return results


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Yambda flat/50m/listens.parquet or synthetic CSV")
    p.add_argument("--source-kind", choices=("synthetic", "yambda"), required=True,
                   help="Mandatory provenance label: never report smoke results as real Yambda")
    p.add_argument("--output", default="outputs/task2")
    p.add_argument("--users", type=int, default=128)
    p.add_argument("--vocab", type=int, default=4096)
    p.add_argument("--context", type=int, default=32)
    p.add_argument("--train-events", type=int, default=30000)
    p.add_argument("--eval-per-user", type=int, default=5)
    p.add_argument("--hidden", type=int, default=96)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--base-steps", type=int, default=500)
    p.add_argument("--reward-steps", type=int, default=200)
    p.add_argument("--rl-steps", type=int, default=100)
    p.add_argument("--base-lr", type=float, default=0.0005)
    p.add_argument("--reward-lr", type=float, default=0.001)
    p.add_argument("--rl-lr", type=float, default=0.00003)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rewards", default="completion,watch_time,balanced,conservative")
    p.add_argument("--betas", default="0.02,0.2")
    p.add_argument("--rl-horizons", default="1,5", help="Compare myopic and hypothetical multi-step policy gradients")
    p.add_argument("--rerank-alphas", default="0,1,3")
    p.add_argument("--allowed-ndcg-drop", type=float, default=0.01)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    result = run(args)
    print(json.dumps({"selection": result["selected_on_validation"],
                      "test": result["test"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
