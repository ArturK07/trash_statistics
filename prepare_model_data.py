from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

SERIES_COLUMNS = ["facility_id", "waste_category_id"]
CATEGORICAL_FEATURES = ["district", "waste_category_id"]

OBSERVATION_REQUIRED_COLUMNS = {
    "record_id",
    "date",
    "facility_id",
    "facility_name",
    "district",
    "facility_type",
    "waste_category_id",
    "waste_category",
    "received_tons",
    "capacity_tons_per_day",
}

EVENT_REQUIRED_COLUMNS = {
    "event_id",
    "occurred_at",
    "facility_id",
    "event_type",
}

FORBIDDEN_FEATURES = {
    "record_id",
    "date",
    "facility_id",
    "facility_name",
    "facility_type",
    "waste_category",
    "received_tons",
    "target_tons",
    "actual_load_pct",
    "possible_target_outlier",
    "split",
}


def load_config() -> dict[str, Any]:
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Не найден config.yaml: {CONFIG_PATH}")

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("config.yaml должен содержать YAML-словарь.")

    return config


def configure_logging(config: dict[str, Any]) -> None:
    log_path = PROJECT_ROOT / config["logging"]["file_path"]
    log_path.parent.mkdir(parents=True, exist_ok=True)

    level_name = str(config["logging"].get("level", "INFO")).upper()
    level = getattr(logging, level_name, None)
    if not isinstance(level, int):
        raise ValueError(f"Неизвестный уровень логирования: {level_name}")

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )


def read_observations(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Не найден CSV с наблюдениями: {path}")

    frame = pd.read_csv(path, encoding="utf-8-sig")

    missing_columns = OBSERVATION_REQUIRED_COLUMNS - set(frame.columns)
    if missing_columns:
        raise ValueError(
            "В CSV отсутствуют столбцы: "
            + ", ".join(sorted(missing_columns))
        )

    frame["date"] = pd.to_datetime(
        frame["date"],
        errors="coerce",
    ).dt.normalize()

    if frame["date"].isna().any():
        count = int(frame["date"].isna().sum())
        raise ValueError(f"Найдено наблюдений с некорректной датой: {count}")

    if frame["record_id"].duplicated().any():
        count = int(frame["record_id"].duplicated().sum())
        raise ValueError(f"Найдены повторяющиеся record_id: {count}")

    for column in ["facility_id", "waste_category_id"]:
        frame[column] = frame[column].astype(str)

    frame["received_tons"] = pd.to_numeric(
        frame["received_tons"],
        errors="coerce",
    )
    frame["capacity_tons_per_day"] = pd.to_numeric(
        frame["capacity_tons_per_day"],
        errors="coerce",
    )

    frame["target_valid"] = (
        frame["received_tons"].notna()
        & np.isfinite(frame["received_tons"])
        & (frame["received_tons"] >= 0)
    )

    invalid_capacity = (
        frame["capacity_tons_per_day"].isna()
        | ~np.isfinite(frame["capacity_tons_per_day"])
        | (frame["capacity_tons_per_day"] <= 0)
    )
    if invalid_capacity.any():
        count = int(invalid_capacity.sum())
        raise ValueError(f"Найдена некорректная мощность объекта: {count}")

    return frame


def read_events(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Не найден JSONL-журнал событий: {path}")

    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Некорректный JSON в строке {line_number}: {error}"
                ) from error

            if not isinstance(record, dict):
                raise ValueError(
                    f"Запись в строке {line_number} должна быть JSON-объектом."
                )

            missing = EVENT_REQUIRED_COLUMNS - set(record)
            if missing:
                raise ValueError(
                    f"В строке {line_number} отсутствуют поля: "
                    + ", ".join(sorted(missing))
                )

            records.append(record)

    if not records:
        raise ValueError("Журнал событий пуст.")

    events = pd.DataFrame(records)

    events["date"] = pd.to_datetime(
        events["occurred_at"],
        errors="coerce",
    ).dt.normalize()

    if events["date"].isna().any():
        count = int(events["date"].isna().sum())
        raise ValueError(f"В журнале событий некорректных дат: {count}")

    events["facility_id"] = events["facility_id"].astype(str)
    events["event_type"] = events["event_type"].fillna("unknown").astype(str)

    if events["event_id"].duplicated().any():
        count = int(events["event_id"].duplicated().sum())
        raise ValueError(f"В журнале повторяются event_id: {count}")

    return events


def aggregate_daily_events(
    events: pd.DataFrame,
    facility_ids: list[str],
    dates: list[pd.Timestamp],
) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    events = events.copy()

    events["severity"] = pd.to_numeric(
        events.get("severity", 0),
        errors="coerce",
    ).fillna(0)

    events["message"] = events.get(
        "message",
        pd.Series("", index=events.index),
    ).fillna("").astype(str)

    events["message_mentions_delay"] = events["message"].str.contains(
        r"задерж|сниж|огранич",
        case=False,
        regex=True,
    ).astype(int)

    events["message_mentions_growth"] = events["message"].str.contains(
        r"рост|увелич|дополнительн",
        case=False,
        regex=True,
    ).astype(int)

    events["temperature_c"] = pd.to_numeric(
        events.get(
            "temperature_c",
            pd.Series(index=events.index, dtype="float64"),
        ),
        errors="coerce",
    )

    event_type_names = sorted(events["event_type"].unique())
    event_type_columns: dict[str, str] = {}

    for index, event_type in enumerate(event_type_names, start=1):
        column = f"event_type_{index:02d}_count"
        events[column] = (events["event_type"] == event_type).astype(int)
        event_type_columns[column] = event_type

    daily_aggregations = {
        "event_count": ("event_id", "count"),
        "severity_sum": ("severity", "sum"),
        "message_mentions_delay_count": (
            "message_mentions_delay",
            "sum",
        ),
        "message_mentions_growth_count": (
            "message_mentions_growth",
            "sum",
        ),
        "temperature_sum_c": ("temperature_c", "sum"),
        "temperature_measurement_count": ("temperature_c", "count"),
    }

    for column in event_type_columns:
        daily_aggregations[column] = (column, "sum")

    daily = (
        events.groupby(["facility_id", "date"], as_index=False)
        .agg(**daily_aggregations)
        .set_index(["facility_id", "date"])
    )

    full_calendar = pd.MultiIndex.from_product(
        [facility_ids, dates],
        names=["facility_id", "date"],
    )
    daily = daily.reindex(full_calendar, fill_value=0)

    numeric_columns = list(daily.columns)
    daily[numeric_columns] = daily[numeric_columns].fillna(0)

    past_feature_sources = [
        "event_count",
        "severity_sum",
        "message_mentions_delay_count",
        "message_mentions_growth_count",
        *event_type_columns.keys(),
    ]

    facility_groups = daily.groupby(level="facility_id", sort=False)
    event_feature_columns: list[str] = []

    for source_column in past_feature_sources:
        previous_day_column = f"{source_column}_previous_day"
        previous_7d_column = f"{source_column}_previous_7d_sum"

        daily[previous_day_column] = facility_groups[source_column].shift(1)
        daily[previous_7d_column] = facility_groups[source_column].transform(
            lambda values: values.shift(1).rolling(
                window=7,
                min_periods=1,
            ).sum()
        )

        daily[previous_day_column] = daily[previous_day_column].fillna(0)
        daily[previous_7d_column] = daily[previous_7d_column].fillna(0)

        event_feature_columns.extend(
            [previous_day_column, previous_7d_column]
        )

    temperature_sum_previous_7d = facility_groups[
        "temperature_sum_c"
    ].transform(
        lambda values: values.shift(1).rolling(
            window=7,
            min_periods=1,
        ).sum()
    )

    temperature_count_previous_7d = facility_groups[
        "temperature_measurement_count"
    ].transform(
        lambda values: values.shift(1).rolling(
            window=7,
            min_periods=1,
        ).sum()
    )

    daily["weather_temperature_previous_7d_mean_c"] = (
        temperature_sum_previous_7d
        / temperature_count_previous_7d.replace(0, np.nan)
    ).fillna(0)

    event_feature_columns.append(
        "weather_temperature_previous_7d_mean_c"
    )

    daily_features = daily[event_feature_columns].reset_index()

    return (
        daily_features,
        event_feature_columns,
        event_type_columns,
    )


def add_time_splits(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.Timestamp, dict[str, str]]:
    settings = config["validation"]
    train_fraction = float(settings["train_fraction"])
    validation_fraction = float(settings["validation_fraction"])
    test_fraction = float(settings["test_fraction"])

    if min(train_fraction, validation_fraction, test_fraction) <= 0:
        raise ValueError("Все доли выборок должны быть больше нуля.")

    if abs(train_fraction + validation_fraction + test_fraction - 1.0) > 1e-9:
        raise ValueError(
            "Доли train, validation и test должны в сумме давать 1."
        )

    unique_dates = sorted(frame["date"].dropna().unique())
    if len(unique_dates) < 3:
        raise ValueError("Для временного разбиения недостаточно дат.")

    train_end_index = int(len(unique_dates) * train_fraction)
    validation_end_index = int(
        len(unique_dates) * (train_fraction + validation_fraction)
    )

    train_end_index = max(1, min(train_end_index, len(unique_dates) - 2))
    validation_end_index = max(
        train_end_index + 1,
        min(validation_end_index, len(unique_dates) - 1),
    )

    train_end = pd.Timestamp(unique_dates[train_end_index - 1])
    validation_end = pd.Timestamp(unique_dates[validation_end_index - 1])

    frame = frame.copy()
    frame["split"] = "test"
    frame.loc[frame["date"] <= train_end, "split"] = "train"

    validation_mask = (
        (frame["date"] > train_end)
        & (frame["date"] <= validation_end)
    )
    frame.loc[validation_mask, "split"] = "validation"

    boundaries = {
        "train_end": train_end.date().isoformat(),
        "validation_end": validation_end.date().isoformat(),
        "test_end": pd.Timestamp(unique_dates[-1]).date().isoformat(),
    }

    return frame, train_end, boundaries


def add_outlier_review_flag(
    frame: pd.DataFrame,
    train_end: pd.Timestamp,
) -> pd.DataFrame:
    frame = frame.copy()

    train_values = frame.loc[
        (frame["date"] <= train_end)
        & frame["target_valid"]
    ]

    quantiles = (
        train_values.groupby(SERIES_COLUMNS)["received_tons"]
        .quantile([0.25, 0.75])
        .unstack()
        .rename(columns={0.25: "q1", 0.75: "q3"})
        .reset_index()
    )

    quantiles["iqr"] = quantiles["q3"] - quantiles["q1"]
    quantiles["lower_bound"] = (
        quantiles["q1"] - 3 * quantiles["iqr"]
    ).clip(lower=0)
    quantiles["upper_bound"] = quantiles["q3"] + 3 * quantiles["iqr"]

    frame = frame.merge(
        quantiles[
            SERIES_COLUMNS + ["lower_bound", "upper_bound"]
        ],
        on=SERIES_COLUMNS,
        how="left",
        validate="many_to_one",
    )

    frame["possible_target_outlier"] = (
        frame["target_valid"]
        & (
            (frame["received_tons"] < frame["lower_bound"])
            | (frame["received_tons"] > frame["upper_bound"])
        )
    )

    return frame


def add_target_history_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values(
        SERIES_COLUMNS + ["date"],
        kind="mergesort",
    ).copy()

    # Если целевая масса отсутствует, для будущих лагов переносится
    # только последнее известное значение из прошлого.
    # Строка без цели позднее исключается из модельной выборки.
    frame["target_history_for_lags"] = (
        frame["received_tons"]
        .where(frame["target_valid"])
        .groupby(
            [frame[column] for column in SERIES_COLUMNS],
            sort=False,
        )
        .ffill()
    )

    grouped = frame.groupby(SERIES_COLUMNS, sort=False)

    for lag in (1, 7, 14):
        frame[f"target_lag_{lag}"] = grouped[
            "target_history_for_lags"
        ].shift(lag)

    frame["target_rolling_mean_7"] = grouped[
        "target_history_for_lags"
    ].transform(
        lambda values: values.shift(1).rolling(
            window=7,
            min_periods=7,
        ).mean()
    )

    frame["target_rolling_mean_14"] = grouped[
        "target_history_for_lags"
    ].transform(
        lambda values: values.shift(1).rolling(
            window=14,
            min_periods=14,
        ).mean()
    )

    return frame


def add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()

    day_of_week = frame["date"].dt.day_of_week
    day_of_year = frame["date"].dt.day_of_year

    frame["day_of_week_sin"] = np.sin(
        2 * np.pi * day_of_week / 7
    )
    frame["day_of_week_cos"] = np.cos(
        2 * np.pi * day_of_week / 7
    )
    frame["day_of_year_sin"] = np.sin(
        2 * np.pi * day_of_year / 365.25
    )
    frame["day_of_year_cos"] = np.cos(
        2 * np.pi * day_of_year / 365.25
    )
    frame["is_weekend"] = day_of_week.isin([5, 6]).astype("int8")

    return frame


def add_one_hot_features(
    frame: pd.DataFrame,
    train_end: pd.Timestamp,
) -> tuple[pd.DataFrame, list[str], dict[str, list[str]]]:
    frame = frame.copy()
    train_mask = frame["date"] <= train_end

    encoded_columns: list[str] = []
    category_mapping: dict[str, list[str]] = {}

    for source_column in CATEGORICAL_FEATURES:
        values = frame[source_column].fillna("__missing__").astype(str)
        categories = sorted(values.loc[train_mask].unique())
        category_mapping[source_column] = categories

        for index, category in enumerate(categories, start=1):
            feature_name = f"{source_column}_level_{index:02d}"
            frame[feature_name] = (values == category).astype("int8")
            encoded_columns.append(feature_name)

    return frame, encoded_columns, category_mapping


def ordered_unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def main() -> None:
    config = load_config()
    configure_logging(config)
    logger = logging.getLogger("waste_monitoring.prepare_model_data")

    demo_settings = config["demo_data"]
    raw_directory = PROJECT_ROOT / config["storage"]["raw_data_directory"]
    processed_directory = (
        PROJECT_ROOT / config["storage"]["processed_data_directory"]
    )
    reports_directory = PROJECT_ROOT / config["artifacts"]["reports_directory"]

    processed_directory.mkdir(parents=True, exist_ok=True)
    reports_directory.mkdir(parents=True, exist_ok=True)

    observations_path = raw_directory / demo_settings["observations_filename"]
    events_path = raw_directory / demo_settings["events_filename"]

    observations = read_observations(observations_path)
    events = read_events(events_path)

    facility_ids = sorted(observations["facility_id"].unique())
    dates = sorted(observations["date"].unique())

    (
        daily_event_features,
        event_feature_columns,
        event_type_mapping,
    ) = aggregate_daily_events(
        events=events,
        facility_ids=facility_ids,
        dates=dates,
    )

    frame = observations.merge(
        daily_event_features,
        on=["facility_id", "date"],
        how="left",
        validate="many_to_one",
    )

    for column in event_feature_columns:
        frame[column] = frame[column].fillna(0)

    frame, train_end, boundaries = add_time_splits(frame, config)
    frame = add_outlier_review_flag(frame, train_end)
    frame = add_target_history_features(frame)
    frame = add_calendar_features(frame)

    frame, one_hot_features, category_mapping = add_one_hot_features(
        frame,
        train_end,
    )

    target_lag_features = [
        "target_lag_1",
        "target_lag_7",
        "target_lag_14",
        "target_rolling_mean_7",
        "target_rolling_mean_14",
    ]

    calendar_features = [
        "day_of_week_sin",
        "day_of_week_cos",
        "day_of_year_sin",
        "day_of_year_cos",
        "is_weekend",
    ]

    capacity_features = ["capacity_tons_per_day"]

    feature_columns = ordered_unique(
        target_lag_features
        + calendar_features
        + capacity_features
        + event_feature_columns
        + one_hot_features
    )

    forbidden = FORBIDDEN_FEATURES.intersection(feature_columns)
    if forbidden:
        raise ValueError(
            "Запрещённые поля попали в признаки: "
            + ", ".join(sorted(forbidden))
        )

    missing_features = set(feature_columns) - set(frame.columns)
    if missing_features:
        raise ValueError(
            "Не рассчитаны признаки: "
            + ", ".join(sorted(missing_features))
        )

    original_rows = len(frame)
    invalid_target_rows = int((~frame["target_valid"]).sum())

    # Пропущенные/невалидные цели исключаются из model rows.
    # Их предшествующие валидные значения уже использованы только
    # для вычисления лагов без заглядывания в будущее.
    model_frame = frame.loc[frame["target_valid"]].copy()
    model_frame["target_tons"] = model_frame["received_tons"]
    model_frame["actual_load_pct"] = (
        100
        * model_frame["received_tons"]
        / model_frame["capacity_tons_per_day"]
    )

    warmup_mask = model_frame[feature_columns].isna().any(axis=1)
    warmup_rows = int(warmup_mask.sum())
    model_frame = model_frame.loc[~warmup_mask].copy()

    if model_frame[feature_columns].isna().any().any():
        raise ValueError("В итоговых модельных признаках остались пропуски.")

    metadata_columns = [
        "date",
        "facility_id",
        "facility_name",
        "district",
        "facility_type",
        "waste_category_id",
        "waste_category",
        "capacity_tons_per_day",
        "target_tons",
        "actual_load_pct",
        "possible_target_outlier",
    ]

    output_columns = ordered_unique(metadata_columns + feature_columns)

    output_files = {}
    split_reports = {}

    for split_name in ("train", "validation", "test"):
        output_path = processed_directory / f"{split_name}.csv"
        subset = model_frame.loc[model_frame["split"] == split_name].copy()

        if subset.empty:
            raise ValueError(f"Выборка {split_name} оказалась пустой.")

        output = subset[output_columns].copy()

        if output.columns.duplicated().any():
            duplicates = output.columns[
                output.columns.duplicated()
            ].tolist()
            raise ValueError(
                f"В {split_name}.csv повторяются столбцы: {duplicates}"
            )

        output.to_csv(output_path, index=False, encoding="utf-8-sig")
        output_files[split_name] = str(
            output_path.relative_to(PROJECT_ROOT)
        )

        split_reports[split_name] = {
            "rows": int(len(output)),
            "columns": int(len(output.columns)),
            "date_min": output["date"].min().date().isoformat(),
            "date_max": output["date"].max().date().isoformat(),
            "possible_target_outliers_retained": int(
                output["possible_target_outlier"].sum()
            ),
            "file": output_files[split_name],
        }

    manifest = {
        "target": "target_tons",
        "target_unit": "tons_per_day",
        "random_seed": int(config["project"]["random_seed"]),
        "split_boundaries": boundaries,
        "feature_count": len(feature_columns),
        "features": feature_columns,
        "feature_groups": {
            "target_lags_and_windows": target_lag_features,
            "calendar": calendar_features,
            "capacity": capacity_features,
            "past_event_log_features": event_feature_columns,
            "one_hot_categories": one_hot_features,
        },
        "one_hot_category_mapping": category_mapping,
        "event_type_column_mapping": event_type_mapping,
        "technical_columns_not_used_as_features": [
            "record_id",
            "facility_id",
            "facility_name",
            "facility_type",
            "waste_category",
        ],
        "event_timing_rule": (
            "Для даты прогноза используются только события предыдущего "
            "дня и предыдущих 7 дней."
        ),
        "target_leakage_check": "passed",
    }

    manifest_path = processed_directory / "feature_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)

    report = {
        "raw_observation_rows": int(original_rows),
        "invalid_target_rows_removed": invalid_target_rows,
        "warmup_rows_removed": warmup_rows,
        "model_rows": int(len(model_frame)),
        "unique_facilities": int(model_frame["facility_id"].nunique()),
        "unique_categories": int(
            model_frame["waste_category_id"].nunique()
        ),
        "feature_count": len(feature_columns),
        "features": feature_columns,
        "possible_target_outliers_flagged_but_retained": int(
            model_frame["possible_target_outlier"].sum()
        ),
        "split_boundaries": boundaries,
        "splits": split_reports,
        "model_files": output_files,
        "feature_manifest": str(
            manifest_path.relative_to(PROJECT_ROOT)
        ),
    }

    report_path = reports_directory / "model_data_report.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    logger.info(
        "Подготовка завершена: исходных строк=%s, исключено строк с "
        "невалидной целью=%s, исключено строк разогрева=%s, "
        "строк для моделей=%s",
        original_rows,
        invalid_target_rows,
        warmup_rows,
        len(model_frame),
    )
    logger.info("Сформировано признаков: %s", len(feature_columns))
    logger.info("Временные границы: %s", boundaries)
    logger.info("Выборки: %s", split_reports)
    logger.info(
        "Необычные значения, сохранённые в выборках: %s",
        report["possible_target_outliers_flagged_but_retained"],
    )
    logger.info("Манифест признаков: %s", manifest_path)
    logger.info("Отчёт обработки: %s", report_path)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("Подготовка модельных данных завершилась ошибкой.")
        raise