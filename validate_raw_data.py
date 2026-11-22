from __future__ import annotations

import csv
import json
import logging
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

OBSERVATION_FIELDS = {
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

EVENT_REQUIRED_FIELDS = {
    "event_id",
    "occurred_at",
    "facility_id",
    "event_type",
    "severity",
    "message",
}


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


def inspect_observations(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Не найден CSV-файл: {path}")

    record_ids: set[str] = set()
    facilities: set[str] = set()
    categories: set[str] = set()
    districts: set[str] = set()
    facility_types: set[str] = set()
    dates: set[str] = set()

    duplicate_ids = 0
    missing_targets = 0
    invalid_dates = 0
    invalid_targets = 0
    negative_targets = 0
    invalid_capacities = 0
    target_min: float | None = None
    target_max: float | None = None
    row_count = 0

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fields = set(reader.fieldnames or [])

        missing_fields = sorted(OBSERVATION_FIELDS - fields)
        if missing_fields:
            raise ValueError(
                "В CSV отсутствуют обязательные поля: "
                + ", ".join(missing_fields)
            )

        for row in reader:
            row_count += 1

            record_id = (row.get("record_id") or "").strip()
            if record_id in record_ids:
                duplicate_ids += 1
            elif record_id:
                record_ids.add(record_id)

            facility_id = (row.get("facility_id") or "").strip()
            category_id = (row.get("waste_category_id") or "").strip()
            district = (row.get("district") or "").strip()
            facility_type = (row.get("facility_type") or "").strip()

            if facility_id:
                facilities.add(facility_id)
            if category_id:
                categories.add(category_id)
            if district:
                districts.add(district)
            if facility_type:
                facility_types.add(facility_type)

            date_text = (row.get("date") or "").strip()
            try:
                date.fromisoformat(date_text)
                dates.add(date_text)
            except ValueError:
                invalid_dates += 1

            target_text = (row.get("received_tons") or "").strip()
            if not target_text:
                missing_targets += 1
            else:
                try:
                    target = float(target_text)
                    if target < 0:
                        negative_targets += 1
                    target_min = (
                        target if target_min is None else min(target_min, target)
                    )
                    target_max = (
                        target if target_max is None else max(target_max, target)
                    )
                except ValueError:
                    invalid_targets += 1

            capacity_text = (row.get("capacity_tons_per_day") or "").strip()
            try:
                if float(capacity_text) <= 0:
                    invalid_capacities += 1
            except ValueError:
                invalid_capacities += 1

    return {
        "file": str(path.relative_to(PROJECT_ROOT)),
        "rows": row_count,
        "unique_record_ids": len(record_ids),
        "duplicate_record_ids": duplicate_ids,
        "date_min": min(dates) if dates else None,
        "date_max": max(dates) if dates else None,
        "unique_dates": len(dates),
        "unique_facilities": len(facilities),
        "unique_categories": len(categories),
        "unique_districts": len(districts),
        "unique_facility_types": len(facility_types),
        "missing_targets": missing_targets,
        "invalid_dates": invalid_dates,
        "invalid_targets": invalid_targets,
        "negative_targets": negative_targets,
        "invalid_capacities": invalid_capacities,
        "target_min": target_min,
        "target_max": target_max,
        "facility_ids": sorted(facilities),
        "category_ids": sorted(categories),
    }


def inspect_events(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Не найден JSONL-файл: {path}")

    event_ids: set[str] = set()
    facility_ids: set[str] = set()
    event_types: Counter[str] = Counter()
    malformed_lines = 0
    missing_fields_count = 0
    row_count = 0

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            row_count += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                malformed_lines += 1
                continue

            if not isinstance(record, dict):
                malformed_lines += 1
                continue

            if EVENT_REQUIRED_FIELDS - record.keys():
                missing_fields_count += 1

            event_id = str(record.get("event_id", "")).strip()
            if event_id in event_ids:
                logging.warning(
                    "Повтор event_id=%s в строке JSONL %s",
                    event_id,
                    line_number,
                )
            elif event_id:
                event_ids.add(event_id)

            facility_id = str(record.get("facility_id", "")).strip()
            if facility_id:
                facility_ids.add(facility_id)

            event_type = str(record.get("event_type", "unknown"))
            event_types[event_type] += 1

    return {
        "file": str(path.relative_to(PROJECT_ROOT)),
        "events": row_count,
        "unique_event_ids": len(event_ids),
        "malformed_lines": malformed_lines,
        "records_missing_required_fields": missing_fields_count,
        "unique_facilities": len(facility_ids),
        "event_types": dict(sorted(event_types.items())),
        "facility_ids": sorted(facility_ids),
    }


def main() -> None:
    config = load_config()
    configure_logging(config)

    settings = config["demo_data"]
    raw_directory = PROJECT_ROOT / config["storage"]["raw_data_directory"]

    observations = inspect_observations(
        raw_directory / settings["observations_filename"]
    )
    events = inspect_events(raw_directory / settings["events_filename"])

    observation_facilities = set(observations["facility_ids"])
    event_facilities = set(events["facility_ids"])

    report = {
        "observations": observations,
        "events": events,
        "event_facilities_without_observations": sorted(
            event_facilities - observation_facilities
        ),
        "observation_facilities_without_events": sorted(
            observation_facilities - event_facilities
        ),
    }

    reports_directory = PROJECT_ROOT / config["artifacts"]["reports_directory"]
    reports_directory.mkdir(parents=True, exist_ok=True)
    report_path = reports_directory / "raw_data_audit.json"

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    logging.info("Проверка исходных данных завершена.")
    logging.info(
        "Наблюдения: %s строк, %s объектов, %s категорий, %s пропусков цели.",
        observations["rows"],
        observations["unique_facilities"],
        observations["unique_categories"],
        observations["missing_targets"],
    )
    logging.info(
        "События: %s записей, %s типов, некорректных строк: %s.",
        events["events"],
        len(events["event_types"]),
        events["malformed_lines"],
    )
    logging.info("Отчёт сохранён: %s", report_path)

    has_critical_errors = any(
        observations[key] > 0
        for key in (
            "duplicate_record_ids",
            "invalid_dates",
            "invalid_targets",
            "negative_targets",
            "invalid_capacities",
        )
    ) or events["malformed_lines"] > 0

    if has_critical_errors:
        raise ValueError(
            "Найдены критические ошибки исходных данных. "
            "Подробности сохранены в отчёте."
        )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("Проверка исходных данных завершилась ошибкой.")
        raise