from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(config_path: Path) -> dict[str, Any]:
    """Читает YAML-конфигурацию и проверяет её верхний уровень."""
    if not config_path.is_file():
        raise FileNotFoundError(f"Файл конфигурации не найден: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Конфигурация должна содержать YAML-словарь.")

    required_sections = (
        "storage",
        "artifacts",
        "logging",
        "validation",
    )

    missing_sections = [
        section for section in required_sections if section not in config
    ]
    if missing_sections:
        raise ValueError(
            "В config.yaml отсутствуют разделы: "
            + ", ".join(missing_sections)
        )

    return config


def validate_config(config: dict[str, Any]) -> None:
    """Проверяет основные настройки проекта."""
    validation = config["validation"]

    try:
        fractions = (
            float(validation["train_fraction"]),
            float(validation["validation_fraction"]),
            float(validation["test_fraction"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "В config.yaml должны быть числовые доли train, validation и test."
        ) from error

    if any(fraction <= 0 or fraction >= 1 for fraction in fractions):
        raise ValueError("Каждая доля разбиения должна быть между 0 и 1.")

    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError(
            "Сумма train_fraction, validation_fraction и test_fraction "
            "должна быть равна 1."
        )

    forecast = config.get("forecast", {})
    if int(forecast.get("horizon_days", 0)) <= 0:
        raise ValueError("forecast.horizon_days должен быть положительным.")

    warning_ratio = float(forecast.get("warning_load_ratio", 0))
    if not 0 < warning_ratio <= 1:
        raise ValueError(
            "forecast.warning_load_ratio должен быть больше 0 и не больше 1."
        )


def create_project_directories(config: dict[str, Any]) -> list[Path]:
    """Создаёт каталоги из конфигурации, не затрагивая их содержимое."""
    relative_directories = [
        config["storage"]["raw_data_directory"],
        config["storage"]["processed_data_directory"],
        config["storage"]["incoming_data_directory"],
        str(Path(config["storage"]["database_path"]).parent),
        config["artifacts"]["models_directory"],
        config["artifacts"]["reports_directory"],
        config["artifacts"]["figures_directory"],
        str(Path(config["logging"]["file_path"]).parent),
    ]

    created_paths: list[Path] = []

    for relative_path in relative_directories:
        directory = (PROJECT_ROOT / relative_path).resolve()

        # Не позволяем конфигурации создавать каталоги за пределами проекта.
        try:
            directory.relative_to(PROJECT_ROOT)
        except ValueError as error:
            raise ValueError(
                f"Путь каталога выходит за пределы проекта: {relative_path}"
            ) from error

        directory.mkdir(parents=True, exist_ok=True)
        created_paths.append(directory)

    return created_paths


def configure_logging(config: dict[str, Any]) -> None:
    """Настраивает вывод журнала в файл и в консоль."""
    log_path = (PROJECT_ROOT / config["logging"]["file_path"]).resolve()
    log_level_name = str(config["logging"].get("level", "INFO")).upper()
    log_level = getattr(logging, log_level_name, None)

    if not isinstance(log_level, int):
        raise ValueError(f"Недопустимый уровень логирования: {log_level_name}")

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )


def main() -> None:
    config = load_config(CONFIG_PATH)
    validate_config(config)
    directories = create_project_directories(config)
    configure_logging(config)

    logger = logging.getLogger("waste_monitoring.bootstrap")
    logger.info("Стартовая проверка проекта завершена.")
    logger.info("Создано или проверено каталогов: %s", len(directories))

    print("Настройки проверены.")
    print("Каталоги созданы или уже существовали.")
    print(f"Журнал: {PROJECT_ROOT / config['logging']['file_path']}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        logging.basicConfig(level=logging.ERROR)
        logging.exception("Ошибка подготовки проекта: %s", error)
        raise