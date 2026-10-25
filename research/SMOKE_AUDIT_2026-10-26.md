# Аудит первого полного CPU smoke — 26 октября 2026

**Подтверждённый запуск:** https://github.com/voityvaity/T-Lab-task2/actions/runs/40223629274

- Workflow: `.github/workflows/manual-task2-smoke.yml`; dispatch / commit `06b5baa8841371f0823a8c889e92c9e5ab9517a2`.
- GitHub conclusion: `success`; job ID `121260322945`.
- Лог показывает `44 passed in 7.22s`, успешный `pip check`, `compileall`, обучение GPT-2 и трёх reward head, 6 RL вариантов (3 objectives × 1 β × 2 горизонта), 3 простых/нейронных контроля + 2 reranking варианта, создание Markdown-отчёта и SVG.
- Smoke-validator в логе: `models_evaluated=12`, `rl_policies=6`, `real_five_step_windows=8`.
- Артефакт: https://github.com/voityvaity/T-Lab-task2/actions/runs/40223629274/artifacts/12610337908 — `task2-synthetic-smoke` с `experiment.json`, `dataset_protocol.json`, `training_logs.json`, checkpoint-ами, `REPORT.md` и двумя SVG.
- Версии из артефакта: Python 3.11.17, `torch==2.14.1+cpu`, `transformers==4.57.6`, `numpy==2.4.6`, `pandas==2.3.3`, `pytest==9.1.1`.
- Тестовое множество: только **8 синтетических пользователей и 48 тестовых next-item запросов**; `target_catalog_coverage=1.0`, тогда как для настоящей Yambda доля OOV может быть очень большой.

## Наиболее важные реальные наблюдения в smoke

| Метод | NDCG@10 (synthetic) | Recall@10 (synthetic) |
| --- | ---: | ---: |
| GPT-2 supervised | 0.152564 | 0.229167 |
| GPT-2 rerank α=1 | 0.152564 | 0.229167 |
| RL completion h=1 | 0.152564 | 0.229167 |
| RL completion h=5 | 0.152564 | 0.229167 |
| RL balanced h=5 | 0.152865 | 0.229167 |
| Sequential Rules (simple) | 0.323325 | 0.666667 |
| Smoothed Transition (non-neural) | **0.379847** | **0.791667** |

Это *не научный результат о методах* — всего шесть supervised optimizer steps и три RL steps. Однако тест доказывает отсутствие архитектурного преимущества после игрушечного обучения и необходимость достаточного GPT-2 pretraining + конкурирующего простого baseline. Маленький прирост RL balanced в NDCG на 0.000301 ничего не доказывает; награда аудитора на test у этого RL даже ниже GPT-2.

## Проверка reward model

| Регрессор на наблюдавшихся test listens | MAE | Константный MAE |
| --- | ---: | ---: |
| Optimized RM | **0.182763** | **0.161134** |
| Uncertainty RM | **0.164661** | **0.161134** |
| Auditor RM | 0.161631 | 0.180589 |

**Реальная проблема исходного критерия:** по тому же `experiment.json` старая логика выбрала `rl_balanced_h5_kl_0.02` из-за микроскопического улучшения auditor reward на validation (~0.000001). При этом оптимизируемая RM предсказывает хуже собственной константы на test и validation. Это типичное следствие численного шума и неудачной модели награды.

**Изменения после успешного smoke (не повторно протестированы GitHub Actions):**

1. В `selection.py` теперь требуется не только полезный auditor, но и MAE оптимизируемой RM **ниже константной** на validation.
2. Минимальный auditor-выигрыш для переключения политики повышен с нуля до `0.001` средней доли дослушивания (эвристический практический порог 0.1 процентного пункта). Его нельзя считать доказательством статистической значимости.
3. В `experiment.py` публикуется название сильнейшего validation baseline по ranking и флаг, конкурентоспособен ли выбранный вариант. Это не подменяет основное сравнение RL с GPT-2.
4. Добавлены тесты на отбраковку численного микровыигрыша и бесполезной optimized RM.

**При новых правилах этот конкретный синтетический эксперимент не прошёл бы guard для перехода с GPT-2 на RL.** После изменения кода необходим новый end-to-end smoke, прежде чем говорить, что новая версия тоже успешно проходит CI.

## Отрицательные проверки и ограничения

- Не было реального Yambda-run, многократных seeds, alternate cutoffs или доказанного эффекта RL.
- 24h session-return на синтетике не измерим: в `dataset_protocol.json` ноль полностью наблюдаемых session-end примеров во всех группах. Это честный null, не нулевой возврат пользователей.
- Предсказанный reward на *новых рекомендациях* остаётся model-based и может быть неверен за пределами наблюдавшихся actions; disagreement двух heads не является доказательством калибровки.
- Node.js 20 deprecation warning в GitHub Actions не нарушил исполнение: runner автоматически использовал Node 24. В будущей ревизии workflows можно обновить версии стандартных Actions после проверки совместимости.
- Не считать 95% bootstrap-интервалы при 8 синтетических пользователях статистическим выводом о Yambda.

## Ближайшие критерии приёмки

1. Повторно запустить manual synthetic smoke после этих исправлений и подтвердить pytest/проверку целостности.
2. Запустить manual real Yambda pilot, подтвердить pinned dataset revision, полный `REPORT.md`, actual train-catalog coverage, RM MAE relative baseline и минимум одну сравнимую RL vs GPT-2 vs reranking таблицу.
3. Если RM на реальных данных не улучшает константную ошибку — RL не объявлять успешной; исследовать ошибку, качество фичей и сдвиг распределения.
4. Только затем расширять seeds/cutoffs и формулировать основные выводы конкурсной работы.
