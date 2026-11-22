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

SPLITS = ("train", "validation", "test")

METADATA_COLUMNS = [
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

FORBIDDEN_MODEL_FEATURES = {
    "record_id",
    "date",
    "received_tons",
    "target_tons",
    "actual_load_pct",
    "possible_target_outlier",
    "facility_id",
    "facility_name",
    "waste_category",
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


def ordered_unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def manual_drop_reason(column: str) -> str | None:
    if column.startswith("facility_id_level_"):
        return "локальный идентификатор объекта; оставлен только в метаданных"

    if column == "facility_id_unknown":
        return "индикатор неизвестного ID; не несёт семантической информации"

    if column in {
        "target_missing_previous_day",
        "target_missing_previous_7d_count",
    }:
        return (
            "постоянный ноль в текущих данных; целевая масса не пропущена"
        )

    if column in {"month_sin", "month_cos"}:
        return (
            "дублирует годовую сезонность, уже представленную "
            "day_of_year_sin/cos"
        )

    redundant_prefixes = {
        "message_char_count_sum_": (
            "длина текста повторяет информацию о числе и типе событий"
        ),
        "message_word_count_sum_": (
            "число слов повторяет информацию о числе и типе событий"
        ),
        "route_reference_count_": (
            "счётчик маршрутов определяется типами событий"
        ),
        "equipment_reference_count_": (
            "счётчик оборудования определяется типами событий"
        ),
        "temperature_sum_c_": (
            "сумма температуры неинтерпретируема; оставлена средняя "
            "температура предыдущей недели"
        ),
        "temperature_measurement_count_": (
            "дублирует число weather_alert событий"
        ),
    }

    for prefix, reason in redundant_prefixes.items():
        if column.startswith(prefix):
            return reason

    return None


def read_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Не найден манифест признаков: {path}")

    with path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    if not isinstance(manifest, dict) or "features" not in manifest:
        raise ValueError("Манифест должен содержать список features.")

    if not isinstance(manifest["features"], list):
        raise ValueError("Поле features в манифесте должно быть списком.")

    return manifest


def read_split(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Не найдена модельная выборка: {path}")

    return pd.read_csv(path, encoding="utf-8-sig")


def select_features(
    original_features: list[str],
    train: pd.DataFrame,
) -> tuple[list[str], dict[str, str], dict[str, str]]:
    dropped: dict[str, str] = {}
    aliases: dict[str, str] = {}

    unique_features = ordered_unique(original_features)
    duplicate_manifest_names = len(original_features) - len(unique_features)

    candidates: list[str] = []

    for column in unique_features:
        if column in FORBIDDEN_MODEL_FEATURES:
            dropped[column] = "запрещено: утечка цели или метаданные"
            continue

        reason = manual_drop_reason(column)
        if reason:
            dropped[column] = reason
            continue

        if column not in train.columns:
            raise ValueError(
                f"Признак из манифеста отсутствует в train.csv: {column}"
            )

        if train[column].nunique(dropna=False) <= 1:
            dropped[column] = "постоянный признак в обучающей выборке"
            continue

        candidates.append(column)

    # При точном совпадении двух столбцов оставляем первый по порядку
    # манифеста. В нём район расположен раньше типа объекта.
    selected: list[str] = []
    selected_arrays: dict[str, np.ndarray] = {}

    for column in candidates:
        values = train[column].to_numpy()
        matching_column = None

        for kept_column, kept_values in selected_arrays.items():
            try:
                is_identical = np.array_equal(
                    values,
                    kept_values,
                    equal_nan=True,
                )
            except TypeError:
                is_identical = np.array_equal(values, kept_values)

            if is_identical:
                matching_column = kept_column
                break

        if matching_column is not None:
            dropped[column] = (
                f"точный дубль признака {matching_column}"
            )
            aliases[column] = matching_column
            continue

        selected.append(column)
        selected_arrays[column] = values

    if not selected:
        raise ValueError("После отбора не осталось модельных признаков.")

    return selected, dropped, {
        "duplicate_manifest_names": str(duplicate_manifest_names),
        **aliases,
    }


def main() -> None:
    config = load_config()
    configure_logging(config)
    logger = logging.getLogger("waste_monitoring.select_model_features")

    processed_directory = (
        PROJECT_ROOT / config["storage"]["processed_data_directory"]
    )
    reports_directory = PROJECT_ROOT / config["artifacts"]["reports_directory"]
    processed_directory.mkdir(parents=True, exist_ok=True)
    reports_directory.mkdir(parents=True, exist_ok=True)

    manifest_path = processed_directory / "feature_manifest.json"
    manifest = read_manifest(manifest_path)

    split_frames = {
        split: read_split(processed_directory / f"{split}.csv")
        for split in SPLITS
    }

    selected_features, dropped_features, aliases = select_features(
        manifest["features"],
        split_frames["train"],
    )

    leaked = FORBIDDEN_MODEL_FEATURES.intersection(selected_features)
    if leaked:
        raise ValueError(
            "В итоговые признаки попала цель или метаданные: "
            + ", ".join(sorted(leaked))
        )

    missing_in_split = {}
    for split, frame in split_frames.items():
        missing = sorted(set(selected_features) - set(frame.columns))
        if missing:
            missing_in_split[split] = missing

    if missing_in_split:
        raise ValueError(
            "В некоторых выборках отсутствуют признаки: "
            + json.dumps(missing_in_split, ensure_ascii=False)
        )

    output_files = {}
    split_reports = {}

    for split, frame in split_frames.items():
        output_path = processed_directory / f"{split}_model.csv"

        columns = ordered_unique(
            [
                *METADATA_COLUMNS,
                *selected_features,
            ]
        )
        missing_columns = sorted(set(columns) - set(frame.columns))
        if missing_columns:
            raise ValueError(
                f"В {split}.csv отсутствуют выходные поля: "
                + ", ".join(missing_columns)
            )

        output = frame[columns].copy()

        if output.columns.duplicated().any():
            duplicates = output.columns[
                output.columns.duplicated()
            ].tolist()
            raise ValueError(
                f"В выходном файле повторяются столбцы: {duplicates}"
            )

        if output[selected_features].isna().any().any():
            missing_count = int(
                output[selected_features].isna().sum().sum()
            )
            raise ValueError(
                f"В {split} модельных признаках есть пропуски: "
                f"{missing_count}"
            )

        output.to_csv(output_path, index=False, encoding="utf-8-sig")
        output_files[split] = str(output_path.relative_to(PROJECT_ROOT))

        split_reports[split] = {
            "rows": int(len(output)),
            "columns": int(len(output.columns)),
            "features": len(selected_features),
            "date_min": str(output["date"].min()),
            "date_max": str(output["date"].max()),
        }

    original_mapping = manifest.get("one_hot_category_mapping", {})
    retained_mapping = {
        source: categories
        for source, categories in original_mapping.items()
        if any(
            feature.startswith(f"{source}_level_")
            for feature in selected_features
        )
    }

    selected_manifest = {
        "target": manifest.get("target", "target_tons"),
        "target_unit": manifest.get("target_unit", "tons_per_day"),
        "random_seed": manifest.get("random_seed"),
        "split_boundaries": manifest.get("split_boundaries", {}),
        "feature_count": len(selected_features),
        "features": selected_features,
        "one_hot_category_mapping": retained_mapping,
        "target_leakage_check": "passed",
        "feature_selection": (
            "Удалены локальные ID, постоянные признаки, точные дубли "
            "и признаки, дублирующие типы событий."
        ),
    }

    selected_manifest_path = (
        processed_directory / "feature_manifest_model.json"
    )
    with selected_manifest_path.open("w", encoding="utf-8") as file:
        json.dump(
            selected_manifest,
            file,
            ensure_ascii=False,
            indent=2,
        )

    audit = {
        "source_manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
        "original_feature_entries": len(manifest["features"]),
        "unique_original_features": len(set(manifest["features"])),
        "selected_feature_count": len(selected_features),
        "selected_features": selected_features,
        "dropped_features": dropped_features,
        "exact_duplicate_aliases": aliases,
        "split_files": output_files,
        "splits": split_reports,
        "output_manifest": str(
            selected_manifest_path.relative_to(PROJECT_ROOT)
        ),
        "technical_ids": {
            "record_id": "не использовался как признак",
            "facility_id": (
                "сохранён в таблицах как метаданные для группировки, "
                "исключён из признаков модели"
            ),
        },
    }

    audit_path = reports_directory / "feature_selection_report.json"
    with audit_path.open("w", encoding="utf-8") as file:
        json.dump(audit, file, ensure_ascii=False, indent=2)

    logger.info(
        "Отбор признаков завершён: было уникальных=%s, оставлено=%s, "
        "исключено=%s",
        audit["unique_original_features"],
        len(selected_features),
        len(dropped_features),
    )
    logger.info("Выходные выборки: %s", split_reports)
    logger.info("Манифест: %s", selected_manifest_path)
    logger.info("Отчёт аудита: %s", audit_path)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("Отбор признаков завершился ошибкой.")
        raise