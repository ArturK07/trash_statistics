from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

GROUP_COLUMNS = ["facility_id", "waste_category_id"]


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("config.yaml должен содержать YAML-словарь.")

    return config


def configure_logging(config: dict[str, Any]) -> None:
    log_path = PROJECT_ROOT / config["logging"]["file_path"]
    log_path.parent.mkdir(parents=True, exist_ok=True)

    level_name = config["logging"].get("level", "INFO").upper()
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


def load_events(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Не найден JSONL-файл: {path}")

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
            records.append(record)

    if not records:
        raise ValueError(f"В JSONL-файле нет событий: {path}")

    events = pd.DataFrame(records)
    required = {"event_id", "occurred_at", "facility_id", "event_type"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(
            "В журнале событий отсутствуют поля: "
            + ", ".join(sorted(missing))
        )

    events["date"] = pd.to_datetime(
        events["occurred_at"], errors="coerce"
    ).dt.normalize()

    if events["date"].isna().any():
        bad_count = int(events["date"].isna().sum())
        raise ValueError(f"У событий некорректная дата: {bad_count} записей.")

    events["facility_id"] = events["facility_id"].astype(str)
    events["event_type"] = events["event_type"].fillna("unknown").astype(str)
    events["message"] = events.get(
        "message", pd.Series("", index=events.index)
    ).fillna("").astype(str)

    events["severity"] = pd.to_numeric(
        events.get("severity", 0), errors="coerce"
    ).fillna(0)

    events["message_char_count"] = events["message"].str.len()
    events["message_word_count"] = (
        events["message"].str.split().str.len().fillna(0)
    )
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

    events["has_route_reference"] = (
        events.get("route_id", pd.Series(index=events.index))
        .notna()
        .astype(int)
    )
    events["has_equipment_reference"] = (
        events.get("equipment_id", pd.Series(index=events.index))
        .notna()
        .astype(int)
    )
    events["weather_temperature_c"] = pd.to_numeric(
        events.get(
            "temperature_c",
            pd.Series(index=events.index, dtype="float64"),
        ),
        errors="coerce",
    )

    # Извлекаем счётчики событий каждого типа из журнала.
    type_counts = pd.crosstab(
        index=[events["date"], events["facility_id"]],
        columns=events["event_type"],
    )
    type_counts.columns = [
        "event_type_count__" + re.sub(r"\W+", "_", str(name)).strip("_")
        for name in type_counts.columns
    ]
    type_counts = type_counts.reset_index()

    aggregations = {
        "event_count": ("event_id", "count"),
        "event_severity_sum": ("severity", "sum"),
        "message_char_count_sum": ("message_char_count", "sum"),
        "message_word_count_sum": ("message_word_count", "sum"),
        "message_mentions_delay_count": (
            "message_mentions_delay",
            "sum",
        ),
        "message_mentions_growth_count": (
            "message_mentions_growth",
            "sum",
        ),
        "route_reference_count": ("has_route_reference", "sum"),
        "equipment_reference_count": ("has_equipment_reference", "sum"),
        "weather_temperature_c": ("weather_temperature_c", "mean"),
        "weather_measurement_count": (
            "weather_temperature_c",
            "count",
        ),
    }

    daily = (
        events.groupby(["date", "facility_id"], as_index=False)
        .agg(**aggregations)
        .merge(type_counts, on=["date", "facility_id"], how="left")
    )

    return daily


def load_observations(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Не найден CSV-файл: {path}")

    observations = pd.read_csv(path, encoding="utf-8-sig")
    required = {
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
    missing = required - set(observations.columns)
    if missing:
        raise ValueError(
            "В CSV отсутствуют поля: " + ", ".join(sorted(missing))
        )

    observations["date"] = pd.to_datetime(
        observations["date"], errors="coerce"
    ).dt.normalize()

    if observations["date"].isna().any():
        bad_count = int(observations["date"].isna().sum())
        raise ValueError(f"В CSV некорректных дат: {bad_count}")

    observations["facility_id"] = observations["facility_id"].astype(str)
    observations["waste_category_id"] = (
        observations["waste_category_id"].astype(str)
    )
    observations["received_tons"] = pd.to_numeric(
        observations["received_tons"], errors="coerce"
    )
    observations["capacity_tons_per_day"] = pd.to_numeric(
        observations["capacity_tons_per_day"], errors="coerce"
    )

    if observations.duplicated("record_id").any():
        duplicate_count = int(observations.duplicated("record_id").sum())
        raise ValueError(f"Найдены повторные record_id: {duplicate_count}")

    if (observations["capacity_tons_per_day"] <= 0).any():
        raise ValueError("Обнаружена неположительная мощность объекта.")

    if (observations["received_tons"] < 0).any():
        raise ValueError("Обнаружена отрицательная масса отходов.")

    return observations


def add_time_splits_and_quality_flags(
    observations: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    settings = config["validation"]
    train_fraction = float(settings["train_fraction"])
    validation_fraction = float(settings["validation_fraction"])
    test_fraction = float(settings["test_fraction"])

    if abs(train_fraction + validation_fraction + test_fraction - 1.0) > 1e-9:
        raise ValueError("Доли train/validation/test должны в сумме давать 1.")

    unique_dates = sorted(observations["date"].dropna().unique())
    if len(unique_dates) < 3:
        raise ValueError("Для временного разбиения недостаточно уникальных дат.")

    train_end_index = max(1, int(len(unique_dates) * train_fraction))
    validation_end_index = max(
        train_end_index + 1,
        int(len(unique_dates) * (train_fraction + validation_fraction)),
    )
    validation_end_index = min(validation_end_index, len(unique_dates) - 1)

    train_end = unique_dates[train_end_index - 1]
    validation_end = unique_dates[validation_end_index - 1]

    observations = observations.copy()
    observations["split"] = "test"
    observations.loc[observations["date"] <= train_end, "split"] = "train"
    validation_mask = (
        (observations["date"] > train_end)
        & (observations["date"] <= validation_end)
    )
    observations.loc[validation_mask, "split"] = "validation"

    observations["target_missing"] = observations["received_tons"].isna()

    # Границы выбросов оцениваются только по обучающему отрезку.
    train_values = observations.loc[
        (observations["split"] == "train")
        & observations["received_tons"].notna()
    ]

    quantiles = (
        train_values.groupby(GROUP_COLUMNS)["received_tons"]
        .quantile([0.25, 0.75])
        .unstack()
        .rename(columns={0.25: "q1", 0.75: "q3"})
        .reset_index()
    )
    quantiles["iqr"] = quantiles["q3"] - quantiles["q1"]
    quantiles["lower_bound"] = (quantiles["q1"] - 3 * quantiles["iqr"]).clip(
        lower=0
    )
    quantiles["upper_bound"] = quantiles["q3"] + 3 * quantiles["iqr"]

    observations = observations.merge(
        quantiles[GROUP_COLUMNS + ["lower_bound", "upper_bound"]],
        on=GROUP_COLUMNS,
        how="left",
        validate="many_to_one",
    )

    observations["target_outlier"] = (
        observations["received_tons"].notna()
        & (
            (observations["received_tons"] < observations["lower_bound"])
            | (observations["received_tons"] > observations["upper_bound"])
        )
    )

    observations["target_clean_tons"] = observations["received_tons"].where(
        ~observations["target_missing"] & ~observations["target_outlier"]
    )

    return observations


def main() -> None:
    config = load_config()
    configure_logging(config)
    logger = logging.getLogger("waste_monitoring.process_data")

    demo_settings = config["demo_data"]
    raw_directory = PROJECT_ROOT / config["storage"]["raw_data_directory"]
    processed_directory = (
        PROJECT_ROOT / config["storage"]["processed_data_directory"]
    )
    processed_directory.mkdir(parents=True, exist_ok=True)

    observations_path = raw_directory / demo_settings["observations_filename"]
    events_path = raw_directory / demo_settings["events_filename"]

    observations = load_observations(observations_path)
    daily_events = load_events(events_path)

    observation_facilities = set(observations["facility_id"].unique())
    event_facilities = set(daily_events["facility_id"].unique())
    unknown_event_facilities = sorted(event_facilities - observation_facilities)
    if unknown_event_facilities:
        logger.warning(
            "В журнале есть неизвестные объектам facility_id: %s",
            unknown_event_facilities,
        )

    merged = observations.merge(
        daily_events,
        on=["date", "facility_id"],
        how="left",
        validate="many_to_one",
    )

    event_count_columns = [
        column
        for column in merged.columns
        if column == "event_count"
        or column.endswith("_sum")
        or column.endswith("_count")
        or column.startswith("event_type_count__")
        or column.startswith("message_mentions_")
        or column.startswith("route_reference_")
        or column.startswith("equipment_reference_")
    ]

    for column in event_count_columns:
        merged[column] = merged[column].fillna(0)

    merged["weather_temperature_observed"] = (
        merged["weather_temperature_c"].notna().astype(int)
    )
    merged["weather_temperature_c"] = merged[
        "weather_temperature_c"
    ].fillna(0)

    merged = add_time_splits_and_quality_flags(merged, config)

    output_path = (
        processed_directory / "observations_with_events.csv"
    )
    merged.to_csv(output_path, index=False, encoding="utf-8-sig")

    report = {
        "input_observations": len(observations),
        "input_events": int(daily_events["event_count"].sum()),
        "output_rows": len(merged),
        "output_columns": list(merged.columns),
        "event_feature_columns": [
            column
            for column in merged.columns
            if column.startswith("event_")
            or column.startswith("message_")
            or column.startswith("route_")
            or column.startswith("equipment_")
            or column.startswith("weather_")
        ],
        "split_date_ranges": {},
        "quality_by_split": {},
        "unknown_event_facilities": unknown_event_facilities,
        "output_file": str(output_path.relative_to(PROJECT_ROOT)),
    }

    for split_name, subset in merged.groupby("split"):
        report["split_date_ranges"][split_name] = {
            "date_min": subset["date"].min().date().isoformat(),
            "date_max": subset["date"].max().date().isoformat(),
            "rows": int(len(subset)),
        }
        report["quality_by_split"][split_name] = {
            "missing_targets": int(subset["target_missing"].sum()),
            "outlier_targets": int(subset["target_outlier"].sum()),
            "usable_targets": int(subset["target_clean_tons"].notna().sum()),
        }

    reports_directory = PROJECT_ROOT / config["artifacts"]["reports_directory"]
    reports_directory.mkdir(parents=True, exist_ok=True)
    report_path = reports_directory / "processing_report.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2, default=str)

    logger.info(
        "Обработка завершена: строк=%s, событий=%s, признаков событий=%s",
        len(merged),
        report["input_events"],
        len(report["event_feature_columns"]),
    )
    logger.info(
        "Пропуски цели=%s, выбросы цели=%s, пригодные значения=%s",
        int(merged["target_missing"].sum()),
        int(merged["target_outlier"].sum()),
        int(merged["target_clean_tons"].notna().sum()),
    )
    logger.info("Подготовленные данные: %s", output_path)
    logger.info("Отчёт обработки: %s", report_path)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("Обработка данных завершилась ошибкой.")
        raise