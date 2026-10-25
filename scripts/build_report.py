"""Generate an honest, data-backed Markdown report from real completed runs only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def table(headers: list[str], rows: list[list[str]]) -> str:
    return ("| " + " | ".join(headers) + " |\n| " + " | ".join(["---"] * len(headers)) + " |\n"
            + "".join("| " + " | ".join(row) + " |\n" for row in rows))


def metric(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.5f}"
    return str(value)


def report(result_path: Path) -> str:
    if not result_path.is_file():
        raise FileNotFoundError("No actual experiment.json found: cannot generate an invented results report")
    data = json.loads(result_path.read_text(encoding="utf-8"))
    source = data.get("source_kind")
    expected = {"synthetic": "synthetic_smoke_only",
                "yambda": "measured_offline_model_based_not_causal"}
    if source not in expected or data.get("status") != expected[source]:
        raise ValueError("Unknown or mismatched data source/status; refuse to present unverified results")
    heading = ("СИНТЕТИЧЕСКИЙ SMOKE — НЕ КОНКУРСНЫЕ РЕЗУЛЬТАТЫ"
               if source == "synthetic" else "Yambda-50M — реальный офлайн-эксперимент")
    lines = [
        "# Отчёт: T-Lab 2026 / RL-последовательные рекомендации",
        "",
        f"> **{heading}.** Автоматически создан из `experiment.json`. "
        "Модельное вознаграждение не доказывает причинный рост удержания.",
        "",
        "## Протокол",
        "",
        f"- Выбранная **по validation**, до чтения теста конфигурация: `{data['selected_on_validation']}`.",
        f"- Условие выбора: {data['selection_rule']}.",
        f"- Устройство: `{data['device']}`; seed: `{data['seed']}`.",
        f"- Границы каталога и временные cutoffs: `dataset_protocol.json`.",
        "",
        "## Качество reward models на реально наблюдавшихся прослушиваниях",
        "",
    ]
    q = data["reward_quality_on_observed_events"]
    lines += [table(["Модель и выборка", "MAE", "MAE константы", "RMSE", "RMSE константы", "Объектов"],
                    [[name, metric(item["mae"]), metric(item["constant_mean_mae"]),
                      metric(item["rmse"]), metric(item["constant_mean_rmse"]), metric(item["n_observed"])]
                     for name, item in q.items() if isinstance(item, dict)]), ""]
    lines += [
        "**Основные** Recall/NDCG учитывают выбранные будущие события вне обучающего "
        "каталога как неизбежные промахи. В отдельной колонке приведено покрытие "
        "всех оцениваемых целей train-only каталогом. "
        "Conditional NDCG показывает качество только по доступным целям, "
        "поэтому не заменяет основную оценку.", "",
    ]
    if data.get("selection_diagnostics"):
        sel = data["selection_diagnostics"]
        lines += ["## Зафиксированный выбор политики на validation", "",
                  f"- Победитель: `{sel['selected']}`.",
                  f"- Validation NDCG baseline: {metric(sel['baseline_ndcg'])}; "
                  f"минимально допустимый NDCG: {metric(sel['ndcg_floor'])}.",
                  f"- Допущено снижение не более {metric(sel['max_relative_ndcg_drop'] * 100)}% "
                  f"и одновременно не более {metric(sel['max_absolute_ndcg_drop'])} абсолютных NDCG.",
                  f"- Решение: `{sel['reason']}`. Здесь нет онлайн-оценки.", ""]
    for split in ("validation", "test"):
        frontier = data.get(f"{split}_pareto_frontier", [])
        if split == "test":
            frontier = data.get("test_descriptive_pareto_frontier", [])
        if frontier:
            lines += [f"## {split.upper()}: недоминируемые конфигурации (описательно)", "",
                      "Ось X — NDCG@10, ось Y — независимая аудиторская оценка "
                      "дослушивания. **Test-фронтир не используется для выбора модели.**", ""]
            lines += [table(["Модель", "NDCG@10", "Auditor completion"],
                            [[row["model"], metric(row["ndcg_at_10"]),
                              metric(row["auditor_completion_at_10"])]
                             for row in frontier]), ""]
    for split in ("validation", "test"):
        lines += [f"## {split.upper()} — один следующий реальный item", ""]
        rows = []
        for name, item in data[split].items():
            rows.append([name, metric(item["recall_at_10"]), metric(item["ndcg_at_10"]),
                         metric(item["target_catalog_coverage"]),
                         metric(item["conditional_ndcg_at_10"]),
                         metric(item["predicted_completion_at_10"]),
                         metric(item["auditor_completion_at_10"]),
                         metric(item["predicted_capped_listen_seconds_at_10"]),
                         metric(item["repeated_from_history_rate"]),
                         metric(item["catalog_coverage"]),
                         metric(item["rm_absolute_disagreement_at_10"]),
                         metric(item["most_common_top1_share"])])
        lines += [table(["Модель", "Recall@10", "NDCG@10", "Target coverage",
                         "Conditional NDCG@10", "Optimized RM",
                         "Auditor RM", "Watch sec", "History repeats", "Catalog coverage",
                         "RM disagreement", "Top1 concentration"], rows), ""]
    proxy = data.get("observed_session_return_proxy_test_only", {})
    lines += ["## Наблюдаемая связь завершения трека с возвращением (не A/B)", "",
              "Сессия заканчивается при перерыве более 30 минут. Возвращение "
              "фиксируется в течение 24 часов после завершения сессии. "
              "Включены только наблюдения с полным горизонтом; результаты "
              "**не являются причинным эффектом рекомендаций**.", ""]
    if proxy:
        lines += [table(["Предыдущее дослушивание", "Завершений сессии",
                         "Возвратов за 24 ч", "Доля возвратов"],
                        [[key, str(x["n_session_ends"]), str(x["returned_within_24h"]),
                          metric(x["return_rate"])] for key, x in proxy.items()]), ""]
    if "reward_quality_on_observed_events" in data:
        lines += ["## Калибровка награды на наблюдавшихся прослушиваниях", "",
                  "Диапазоны predicted completion разбиты на 10 интервалов, "
                  "показаны доля реально дослушанного и абсолютная ошибка. "
                  "Калибровка на наблюдавшихся событиях **не гарантирует** "
                  "качества на новых предложениях.", ""]
        for name, result in data["reward_quality_on_observed_events"].items():
            if not isinstance(result, dict):
                continue
            lines += [f"### {name}", ""]
            buckets = result.get("calibration_bins", [])
            if buckets:
                lines += [table(["Бин", "Объектов", "Средний прогноз", "Средний факт", "MAE"],
                                [[str(row["bin"]), str(row["count"]),
                                  metric(row["mean_predicted"]),
                                  metric(row["mean_observed"]), metric(row["mae"])]
                                 for row in buckets]), ""]
            groups = result.get("mae_by_training_popularity")
            if groups:
                lines += [table(["Train-популярность", "Наблюдений", "MAE", "Средний факт"],
                                [[key, str(value["observations"]), metric(value["mae"]),
                                  metric(value["mean_observed"])]
                                 for key, value in groups.items()]), ""]
    lines += ["## Парные пользовательские bootstrap-интервалы разницы Test NDCG@10", ""]
    delta = data["test_paired_ndcg_deltas"]
    lines += [table(["Модель − GPT2", "Средняя разница", "95% CI", "Кластеров"],
                    [[key, metric(info["mean"]), ", ".join(metric(x) for x in info["ci_95"]),
                      metric(info["user_clusters"])] for key, info in delta.items()]), ""]
    lines += ["## Пять гипотетических шагов", "",
              "**Ни один сгенерированный item не является зарегистрированным прослушиванием.**", ""]
    for name, curve in data["five_step_hypothetical_rollouts"].items():
        lines += [f"### {name}", ""]
        lines += [table(["Шаг", "Предсказанное дослушивание", "Повтор в текущем контексте"],
                        [[str(row["step"]), metric(row["mean_predicted_completion"]),
                          metric(row["repeat_of_context_rate"])] for row in curve]), ""]
    window_count = data.get("real_five_step_windows", 0)
    lines += [f"**Проверенных последовательных пятишаговых траекторий: {window_count}.**",
              "Пропущенные либо совпадающие по времени события не считаются "
              "действительно соседними. При нуле траекторий динамика по "
              "реальным историям не может быть оценена.", ""]
    for name, curve in data["real_context_vs_generated_context_drift"].items():
        lines += [f"### Расхождение реальной и гипотетической истории: {name}", ""]
        lines += [table(["Шаг", "Траекторий", "JS-divergence", "Hit@10 с вымышленным контекстом"],
                        [[str(row["step"]), str(row["paired_trajectories"]),
                          metric(row["js_divergence_from_real_context"]),
                          metric(row["free_rollout_real_next_hit_at_10"])] for row in curve]), ""]
    lines += ["## Ограничения", ""]
    lines.extend(f"- {warning}" for warning in data["limitations"])
    lines += ["", "## Интерпретация", "",
              "Рост `optimized RM` недостаточен. Для содержательного улучшения нужны "
              "небольшая потеря релевантности, согласие auditor RM, отсутствие "
              "коллапса разнообразия и устойчивость на независимых временных срезах. "
              "Без рандомизированных показов и онлайн A/B нельзя приписывать причинный рост retention.",
              ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    a = parser.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(report(a.input), encoding="utf-8")
    print(f"Generated source-labeled report (see SYNTHETIC/Yambda banner): {a.output}")


if __name__ == "__main__":
    main()
