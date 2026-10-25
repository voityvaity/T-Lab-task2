> **Проверка методологии 26.10.2026:** обнаружена и исправлена ошибка единиц `timestamp` в анализе возвращений: Yambda хранит **секунды, округлённые до пяти секунд**. В старых артефактах показатели, названные «24-часовыми», фактически измеряли горизонт 4,8 часа и **не должны цитироваться как 24h retention**. Исправление находится в `main`, но требует пересчёта реальных логов. Прочие NDCG/Recall/RMSE этого не касаются.

# T-Lab 2026 · Task 2: RL для последовательных рекомендаций

**Статус на 26.10.2026: первый реальный Yambda-50M pilot на коммите `19a903a` и 48 тестов прошли. ПОСЛЕ ЭТОГО усилены GPT-2 и reward model; новая версия кода ещё НЕ ПРОХОДИЛА CI и не обучалась на Yambda. Результаты предыдущего пилота не доказывают выигрыш RL.** Подробности: [первый пилот](research/YAMBDA_PILOT_2026-10-26.md), [доработка обучения](research/TRAINING_UPGRADE_2026-10-26.md). Этот репозиторий — самостоятельное решение задания №2 Академии Т-Банка. Задание №1 находится в другом репозитории и не подменяет результаты этой работы.

## Исследовательский вопрос

Можно ли повысить предсказанный полезный отклик пользователя с помощью RL-дообучения GPT-2, сохранив релевантность рекомендаций? Когда возникает reward hacking, массовое предпочтение популярных треков, повторение контента или накопление ошибок при генерации пяти объектов?

**Важно:** увеличение оценки обученной reward model **не является доказательством** улучшения пользовательского опыта. Yambda не содержит достоверных логов всех показанных вариантов и вероятностей исторической политики, поэтому вне наблюдавшихся действий оценка контрфактического вознаграждения ненадёжна.

**Научные источники и ограничения:** [обзор GPTRec-RL, ROLeR и Spotify Research](research/RELATED_WORK_AND_ABLATIONS.md).

**Исследовательский протокол:** [EXPERIMENT_PROTOCOL.md](research/EXPERIMENT_PROTOCOL.md) — критерии выбора, reward hacking, временные срезы и правила интерпретации отрицательного результата.

**Последующая доработка:** [TRAINING_UPGRADE_2026-10-26.md](research/TRAINING_UPGRADE_2026-10-26.md) — отдельная фиксация изменений **после знакомства с первым тестовым пилотом**, без выдуманных результатов новой модели.

**Подтверждённые эксперименты:** [отчёт реального Yambda pilot](research/YAMBDA_PILOT_2026-10-26.md), [GitHub Actions Yambda №40225318999](https://github.com/voityvaity/T-Lab-task2/actions/runs/40225318999), [повторный synthetic smoke №40225192422](https://github.com/voityvaity/T-Lab-task2/actions/runs/40225192422) и [первый smoke-аудит](research/SMOKE_AUDIT_2026-10-26.md). Результаты реального пилота **не следует** переносить на другие seed/пользовательские когорты без повторения.

**Исторический аудит до успешного smoke:** [проверка 25.10.2026](research/RE_AUDIT_2026-10-25.md). Актуальный статус — выше.

## Выбор данных

[Yandex Yambda-50M](https://huggingface.co/datasets/yandex/yambda). Официально: 10 000 пользователей, 934 057 треков, 46 467 212 прослушиваний, 881 456 лайков и 107 776 дизлайков. Плоский `listens.parquet` содержит `uid, item_id, timestamp, is_organic, played_ratio_pct, track_length_seconds`. `played_ratio_pct` может превышать 100 при повторных прослушиваниях; `timestamp` измеряется пятисекундными интервалами, а не Unix-секундами.

Первый вычислительно ограниченный прогон использует подвыборку пользователей и **top-N идентификаторов треков, выбранных исключительно на train**. Нельзя строить словарь по всему датасету: частоты из validation/test раскрыли бы будущее. Неизвестные объекты остаются OOV в истории. **Будущие OOV-треки теперь учитываются в основных Recall/NDCG как промахи**, а не исключаются из проверки. Дополнительно публикуется охват обучающего каталога и условные метрики для известных треков. Итоги не должны выдаваться за результаты на полном каталоге Yambda.

## Зафиксированный протокол

1. **Глобальный временной split.** Train, validation, test образуют последовательные непересекающиеся интервалы. Все пользователи имеют одинаковые абсолютные cutoffs. Для каждого оцениваемого события контекст состоит только из ранее наблюдавшихся событий, включая предшествующие события текущего тестового периода при прекаузальной (prequential) оценке.
2. **GPT-2 from scratch.** Causal Transformer (GPT-2 architecture), token = идентификатор трека; смешанная next-item cross-entropy по целевому следующему треку и всем допустимым causal-переходам внутри истории; `PAD=0`, `OOV=1`. При левом дополнении явно пересчитываем `position_ids` только по реальным токенам, отдельно от `attention_mask`. Размеры слоёв, длина контекста, выборки, seed и версии библиотек фиксируются в конфигурации запусков.
3. **Reward model.** Три reward-head на общем замороженном GPT-2 encoder; каждая имеет собственный сглаженный TRAIN-only item completion prior и обучаемую контекстную поправку по MSE, с выбором checkpoint по validation MSE: основная и вторая (с иной инициализацией) задают эвристику штрафа за disagreement; третья обучается на непересекающихся TRAIN-пользователях для аудита. Отдельные головы **не гарантируют** надёжного контрфактического reward. Качество проверяется на реальных наблюдавшихся воспроизведениях; MAE и RMSE сравниваются с константой. Дополнительно публикуются десять бинов калибровки регрессии, RMSE против TRAIN-среднего и TRAIN-only item prior, MAE против TRAIN-медианы и показатели для редких/популярных треков.
4. **Контроль без RL.** Popularity, собственные облегчённые Sequential Rules и Smoothed Transition (сглаженные многолаговые переходы), исходный GPT-2 и post-hoc reranking top-M по log-prob + reward. Одинаковые test queries, train-only каталог и единые метрики. Наши Sequential Rules — **не** точное воспроизведение официального Spotify PCTM/SeqRules.
5. **RL.** On-policy REINFORCE с leave-one-out baseline и регуляризацией KL(`policy || frozen GPT-2`). Меняем KL-штраф, reward и горизонт. GPT-2 предварительно сохраняется по минимальной validation cross-entropy. Выбор RL-кандидата: рост независимого auditor reward при относительной потере validation NDCG не более 5% **и** абсолютной не более 0.005; без подтверждения прироста остаётся GPT-2. **Дополнительные ограничения:** и optimized RM, и auditor RM должны превзойти константный прогноз по RMSE на validation (TRAIN-mean); выигрыш auditor reward должен быть больше **0.001**. Это эвристический фильтр от численного шума, **не доказательство статистической значимости**. Иначе выбирается GPT-2. Тест раскрывается только после выбора.
6. **Функции награды.** `completion`, `watch_time`, `balanced` и `conservative` (минимум предсказаний двух reward-head). Последняя — эвристика чувствительности к неопределённости, **не ROLeR и не калиброванная граница**. Все proxy нельзя считать доказанной долгосрочной удовлетворённостью.
7. **Оценка.** Recall@10 и NDCG@10 при one-next-item ground truth (Recall становится HitRate): основные знаменатели включают все отобранные будущие события, в том числе OOV (нулевой hit); условные метрики для in-vocabulary доступны отдельно. Средняя предсказанная награда и, по возможности, независимая диагностическая reward model; catalog coverage, unique items/user, доля повторов, item popularity, concentration, item-length distribution, KL к исходной политике.
8. **Пять шагов.** Teacher-forced контексты из реальной истории сравниваются со свободной генерацией, где рекомендованные треки добавляются в контекст *гипотетически*. Это **не** реальные пользовательские прослушивания. Измеряем рост расхождения предсказаний/сдвигов и явно показываем влияние допущений. Для сравнения с реально наблюдавшимися событиями принимаются лишь **строго последовательные** пятёрки без пропусков и одинаковых временных меток; число таких окон обязательно указывается.
9. **Долгосрочная цель.** В отчёт включается наблюдаемая доля возвращений в течение 24 часов после окончания сессии (перерыв больше 30 минут), с правым цензурированием: брать только события с доступным полным горизонтом. Связь с долей дослушивания носит **ассоциативный, не причинный** характер. Для вывода об улучшении retention нужен онлайн A/B-тест или пригодные randomized logs.

## Правила научной добросовестности

- Нельзя считать отсутствие взаимодействия доказанным дизлайком.
- Reward-model score на действиях политики *вне наблюдённых данных* может увеличиваться вследствие эксплуатации ошибок модели. Отчёт обязан проверять coverage и model-disagreement/OOD.
- Train-only построение словаря и статистик, отсутствие тестовых событий в обучении, единый протокол сравнения, фиксированные seeds и контроль асимметрии.
- Нельзя подбирать выигрышную конфигурацию по test NDCG@10. Если RL ухудшает релевантность — фиксировать отрицательный результат, а не скрывать его.
- Учитывать существенную разницу между «пять рекомендаций подряд» и «пять реально прослушанных треков».

## Ожидаемые результаты проекта

Первые реальные checkpoints GPT-2/reward/one-step RL, таблицы MAE/NDCG/Recall, графики Pareto-компромисса, пятишаговая диагностика и артефакт **получены в пилоте**. Но полноценные 5-step RL ablations, несколько seed/временных срезов, достаточно обученная GPT-2 и надёжная reward model **пока отсутствуют**. См. [пилотный отчёт](research/YAMBDA_PILOT_2026-10-26.md).

## Что реализовано и проверено пилотными GitHub Actions

- `src/tlab2/data.py`: потоковое чтение Parquet, общие временные границы, выбор пользователей по TRAIN, словарь только из TRAIN, reservoir-sampling обучающих примеров, OOV-контексты и честные OOV-промахи в оценке, защита от событий с одинаковой пятисекундной меткой.
- `src/tlab2/reward_priors.py`: сглаженная оценка доли дослушивания по item из собственной TRAIN-подвыборки каждой reward-head.
- `src/tlab2/models.py`: GPT2LMHeadModel **с нуля**, reward head на замороженных GPT-2 признаках (контекст, item, prior, длина, train-популярность, повторяемость), одношаговый и пятишаговый REINFORCE (гипотетические переходы), дисконтированную награду и точный policy KL к исходному GPT-2.
- `src/tlab2/experiment.py`: обучение GPT-2 по плотной causal-loss и выбор по валидации, обучение reward по MSE с ранней остановкой, три reward-head (основная/uncertainty на одном TRAIN-пуле, auditor — на другом), четыре вида reward, два горизонта RL, разные KL, reranking по log-prob, baseline-метрики, user-cluster bootstrap, многократная генерация и JS-divergence.
- `src/tlab2/baselines.py`: train-only Popularity, Sequential Rules и многолаговая сглаженная статистика переходов. Это собственные приближённые модели, не точные реализации Spotify PCTM.
- `src/tlab2/selection.py`: заранее зафиксированный детерминированный выбор по validation c относительным и абсолютным пределом потерь, Pareto frontier для анализа.
- `src/tlab2/diagnostics.py`: строгая проверка смежности пяти реально наблюдавшихся событий и защита от пропусков между событиями.
- `scripts/check_smoke.py`: проверяет, что синтетический workflow действительно исполнил все основные варианты моделей и не пропустил пятишаговую проверку.
- `scripts/plot_tradeoff.py`: автоматически строит отдельные SVG-графики компромисса NDCG/auditor reward для validation и test.
- `research/RELATED_WORK_AND_ABLATIONS.md`: научный обзор, конкретное происхождение идей, ограничения точности и план абляций.
- `tests/`: тесты причинного time split, RL-градиентов, uncertain reward, простых baseline, session-return proxy и запрета выдавать synthetic smoke за реальные результаты.
- `.github/workflows/`: **только ручные** smoke и настоящий Yambda-run. До запуска этих workflow нельзя называть код испытанным или результаты измеренными.

**Все приведённые числа взяты из проверенных запусков.** Повторный smoke и настоящий Yambda-50M pilot на `19a903a` успешны; реальный пилот включает 48 прошедших тестов. Следующие задачи — увеличить качество GPT-2 и reward model, провести 5-step RL и дополнительные seed/cutoff проверки, затем написать окончательный конкурсный анализ. Подробности и ограничения: [YAMBDA_PILOT_2026-10-26.md](research/YAMBDA_PILOT_2026-10-26.md).

## Воспроизведение и запуск

Python 3.11+:

```bash
python -m pip install -e ".[train,download,dev]"
pytest -q
python scripts/make_synthetic.py --users 8 --events 90
python -m tlab2.experiment --data data/synthetic_listens.csv --source-kind synthetic --output outputs/smoke --users 8 --vocab 64 --context 12 --hidden 32 --layers 1 --heads 4 --batch 4 --base-steps 6 --reward-steps 5 --rl-steps 3 --train-events 300 --eval-per-user 6 --rewards completion,balanced,conservative --betas 0.02 --rl-horizons 1,5 --rerank-alphas 0,1 --cpu
python scripts/check_smoke.py --input outputs/smoke/experiment.json
python scripts/build_report.py --input outputs/smoke/experiment.json --output outputs/smoke/REPORT.md
python scripts/plot_tradeoff.py --input outputs/smoke/experiment.json --outdir outputs/smoke
```

Этот синтетический прогон проверяет связность программ, **а не реальное качество алгоритма**.

Реальный Yambda-50M (не загружаем большие файлы в Git):

```bash
hf download yandex/yambda flat/50m/listens.parquet --repo-type dataset --revision e8b04458db77e2fb30a24e65f78923fa48b7d297 --local-dir data
python -m tlab2.experiment --data data/flat/50m/listens.parquet --source-kind yambda --output outputs/yambda --users 256 --vocab 4096 --context 48 --train-events 100000 --eval-per-user 24 --base-steps 1500 --reward-steps 650 --rl-steps 120 --dense-weight 0.65 --reward-prior-smoothing 40 --rewards completion,conservative --betas 0.1 --rl-horizons 1,5
```

Для эксперимента на GPU PyTorch автоматически выберет CUDA при её доступности. На обычном CPU подобный прогон может требовать значительного времени, длительность не гарантируется. Сохраняются `dataset_protocol.json` (включая SHA-256 точного исходного файла), `experiment.json`, `training_logs.json`, SVG-графики trade-off, `REPORT.md` и checkpoints моделей. Ручные workflows также сохраняют `requirements-frozen.txt` и `source-git-sha.txt` для воспроизводимости. Отчёт отказывается принимать противоречащие друг другу `source_kind` и `status`. Программы не считают «негативами» треки без зарегистрированного отклика.

**Без собственного запуска:** в GitHub Actions имеются [Manual Task 2 smoke](https://github.com/voityvaity/T-Lab-task2/actions/workflows/manual-task2-smoke.yml) и [Manual Yambda-50M RL research](https://github.com/voityvaity/T-Lab-task2/actions/workflows/manual-yambda-research.yml). Обе задачи запускаются только вручную. Сначала тестируем synthetic smoke; реальный CPU pilot расходует GitHub Actions minutes и тоже требует ручного запуска. В режиме **pilot** проверяются `completion` и `conservative` с одним KL и одношаговым RL; новый режим **improved** (по умолчанию) берёт 100 тыс. обучающих примеров, до 1500 SFT/650 reward шагов и 24 события на пользователя для оценки, сравнивает одношаговый и пятишаговый RL. Режим **research** расширяет исследование до четырёх видов reward, двух KL и обоих горизонтов. Для исследования устойчивости можно вручную выбирать seed 42/43/44 и train_fraction 0.65/0.70/0.75, при valid_fraction=0.15. Оба режима требуют ручного запуска и расходуют Actions minutes.

### Что обязательно проверить после реального обучения

Помимо NDCG/Recall и reward — RMSE против TRAIN-mean и сглаженного item prior, MAE против TRAIN-медианы, расхождение optimized/auditor RM, рост популярности, долю повторов, широту каталога, распределение длительностей и кривую деградации при пятишаговой генерации. Порог потери NDCG зафиксирован до теста: одновременно не более 5% относительного снижения и не более 0.005 абсолютного. Выбор метода проводится только на validation, не на test. Итоговый график тестовой Pareto-границы исключительно описателен. На итоговые значения влияют урезанный train-каталог и отсутствие настоящих случайных показов.

## Исходные работы

- Petrov & Macdonald (2024), [Aligning GPTRec with Beyond-Accuracy Goals with Reinforcement Learning](https://arxiv.org/abs/2403.04875): policy-gradient для разнообразия и популярностного смещения при Next-K генерации.
- Matveev et al. (2026), [Session-Level Optimization for Large-Scale Retrieval using REINFORCE with Multi-Step Off-Policy Correction](https://arxiv.org/abs/2607.02818): off-policy REINFORCE, многошаговые поправки и ограничения модельно-основанной оценки на Yambda-5B. **Их результаты нельзя переносить на наши другие данные/протокол без повторной проверки.**
- [Yambda dataset card](https://huggingface.co/datasets/yandex/yambda) и [Yambda reproducibility guide](https://huggingface.co/datasets/yandex/yambda/blob/main/benchmarks/README.md).
