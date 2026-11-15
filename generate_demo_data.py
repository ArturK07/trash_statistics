from __future__ import annotations

import csv
import json
import logging
import math
import random
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

EVENT_DEFINITIONS = {
    "city_event": {
        "probability": 0.025,
        "impact": 0.20,
        "messages": [
            "Городское событие увеличило поток отходов.",
            "Зафиксирован рост отходов в районе проведения мероприятия.",
        ],
    },
    "road_restriction": {
        "probability": 0.035,
        "impact": -0.18,
        "messages": [
            "Ограничено движение транспорта на маршруте вывоза.",
            "Возможна задержка доставки отходов на объект.",
        ],
    },
    "equipment_delay": {
        "probability": 0.020,
        "impact": -0.12,
        "messages": [
            "Зарегистрирована задержка работы оборудования.",
            "Отмечено временное снижение пропускной способности.",
        ],
    },
    "weather_alert": {
        "probability": 0.060,
        "impact": -0.04,
        "messages": [
            "Неблагоприятные погодные условия могут повлиять на вывоз.",
            "Поступило предупреждение о сложных погодных условиях.",
        ],
    },
    "planned_maintenance": {
        "probability": 0.018,
        "impact": -0.15,
        "messages": [
            "На объекте проводится плановое обслуживание линии.",
            "Часть оборудования временно выведена на обслуживание.",
        ],
    },
    "route_change": {
        "probability": 0.025,
        "impact": 0.05,
        "messages": [
            "Изменён маршрут движения мусоровозов.",
            "На объект перенаправлена дополнительная партия отходов.",
        ],
    },
}


def load_config() -> dict[str, Any]:
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Не найден файл конфигурации: {CONFIG_PATH}")

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict) or "demo_data" not in config:
        raise ValueError("В config.yaml отсутствует раздел demo_data.")

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


def date_range(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def build_facilities(settings: dict[str, Any]) -> list[dict[str, Any]]:
    districts = settings["districts"]
    facility_types = settings["facility_types"]
    facility_count = int(settings["facility_count"])

    if not districts or not facility_types or facility_count <= 0:
        raise ValueError(
            "Укажите районы, типы объектов и положительный facility_count."
        )

    facilities = []

    for index in range(facility_count):
        number = index + 1
        facilities.append(
            {
                "id": f"REC-{number:02d}",
                "name": f"Объект переработки {number:02d}",
                "district": districts[index % len(districts)],
                "facility_type": facility_types[index % len(facility_types)],
                "base_tons_per_day": 7.0 + (index % 6) * 1.7,
                "capacity_factor": 0.85 + (index % 5) * 0.08,
            }
        )

    return facilities


def generate_events(
    dates: list[date],
    facilities: list[dict[str, Any]],
    rng: random.Random,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], float]]:
    records: list[dict[str, Any]] = []
    daily_impacts: dict[tuple[str, str], float] = {}

    for current_date in dates:
        date_text = current_date.isoformat()

        for facility in facilities:
            facility_id = facility["id"]
            daily_impact = 0.0

            for event_type, definition in EVENT_DEFINITIONS.items():
                if rng.random() >= definition["probability"]:
                    continue

                severity = rng.randint(1, 3)
                record = {
                    "event_id": f"EV-{len(records) + 1:08d}",
                    "occurred_at": (
                        f"{date_text}T{rng.randint(0, 23):02d}:00:00"
                    ),
                    "facility_id": facility_id,
                    "district": facility["district"],
                    "event_type": event_type,
                    "severity": severity,
                    "message": rng.choice(definition["messages"]),
                }

                if event_type in {"road_restriction", "route_change"}:
                    record["route_id"] = f"R-{rng.randint(1, 20):02d}"

                if event_type in {
                    "equipment_delay",
                    "planned_maintenance",
                }:
                    record["equipment_id"] = f"EQ-{rng.randint(1, 12):02d}"

                if event_type == "weather_alert":
                    record["weather_code"] = rng.choice(
                        ["rain", "snow", "wind", "heat"]
                    )
                    record["temperature_c"] = round(
                        rng.uniform(-25, 35), 1
                    )

                records.append(record)
                daily_impact += definition["impact"] * severity / 2

            daily_impacts[(date_text, facility_id)] = daily_impact

    return records, daily_impacts


def generate_observations(
    dates: list[date],
    facilities: list[dict[str, Any]],
    categories: list[dict[str, Any]],
    capacity: float,
    daily_impacts: dict[tuple[str, str], float],
    rng: random.Random,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    start_date = dates[0]

    for current_date in dates:
        date_text = current_date.isoformat()
        day_index = (current_date - start_date).days

        annual_seasonality = 1 + 0.12 * math.sin(
            2 * math.pi * day_index / 365.25
        )
        long_term_trend = 1 + 0.00006 * day_index

        if current_date.weekday() in (5, 6):
            weekday_factor = 0.78
        elif current_date.weekday() in (0, 1):
            weekday_factor = 1.08
        else:
            weekday_factor = 1.0

        for facility in facilities:
            event_impact = daily_impacts.get(
                (date_text, facility["id"]),
                0.0,
            )

            for category in categories:
                category_factor = float(category["demand_factor"])

                if category["id"] == "organic":
                    month_factor = (
                        1.15 if current_date.month in (6, 7, 8) else 1.0
                    )
                else:
                    month_factor = 1.0

                expected_tons = (
                    facility["base_tons_per_day"]
                    * category_factor
                    * annual_seasonality
                    * weekday_factor
                    * long_term_trend
                    * month_factor
                    * max(0.50, 1 + event_impact)
                )

                received_tons = max(
                    0.05,
                    expected_tons + rng.gauss(0, 1.3),
                )

                rows.append(
                    {
                        "record_id": (
                            f"{date_text}|{facility['id']}|{category['id']}"
                        ),
                        "date": date_text,
                        "facility_id": facility["id"],
                        "facility_name": facility["name"],
                        "district": facility["district"],
                        "facility_type": facility["facility_type"],
                        "waste_category_id": category["id"],
                        "waste_category": category["name"],
                        "received_tons": round(received_tons, 2),
                        "capacity_tons_per_day": round(
                            capacity * facility["capacity_factor"],
                            2,
                        ),
                    }
                )

    return rows


def write_csv_if_missing(
    path: Path,
    rows: list[dict[str, Any]],
) -> bool:
    if path.exists():
        logging.info("Файл уже существует, оставляю без изменений: %s", path)
        return False

    if not rows:
        raise ValueError(f"Нет строк для записи в {path}")

    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    logging.info("Создан CSV: %s; строк данных: %s", path, len(rows))
    return True


def write_jsonl_if_missing(
    path: Path,
    records: list[dict[str, Any]],
) -> bool:
    if path.exists():
        logging.info("Файл уже существует, оставляю без изменений: %s", path)
        return False

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    logging.info("Создан JSONL: %s; событий: %s", path, len(records))
    return True


def main() -> None:
    config = load_config()
    configure_logging(config)

    settings = config["demo_data"]
    start = date.fromisoformat(settings["start_date"])
    end = date.fromisoformat(settings["end_date"])

    if end < start:
        raise ValueError("end_date не может быть раньше start_date.")

    seed = int(settings.get("random_seed", config["project"]["random_seed"]))
    rng = random.Random(seed)

    dates = list(date_range(start, end))
    facilities = build_facilities(settings)
    categories = settings["categories"]

    events, impacts = generate_events(dates, facilities, rng)
    observations = generate_observations(
        dates=dates,
        facilities=facilities,
        categories=categories,
        capacity=float(settings["capacity_tons_per_day_per_category"]),
        daily_impacts=impacts,
        rng=rng,
    )

    raw_directory = PROJECT_ROOT / config["storage"]["raw_data_directory"]
    raw_directory.mkdir(parents=True, exist_ok=True)

    observations_path = raw_directory / settings["observations_filename"]
    events_path = raw_directory / settings["events_filename"]

    created_observations = write_csv_if_missing(
        observations_path,
        observations,
    )
    created_events = write_jsonl_if_missing(events_path, events)

    logging.info(
        "Подготовка завершена: дат=%s, объектов=%s, категорий=%s, "
        "наблюдений=%s, событий=%s",
        len(dates),
        len(facilities),
        len(categories),
        len(observations),
        len(events),
    )
    logging.info(
        "Новые файлы созданы: CSV=%s, JSONL=%s",
        created_observations,
        created_events,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("Не удалось подготовить синтетические данные.")
        raise