from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


ROOT = Path(__file__).resolve().parent

TRAIN_PATH = ROOT / "data" / "processed" / "train.csv"
VALIDATION_PATH = ROOT / "data" / "processed" / "validation.csv"
TEST_PATH = ROOT / "data" / "processed" / "test.csv"
MANIFEST_PATH = ROOT / "data" / "processed" / "feature_manifest.json"

REPORTS_DIR = ROOT / "artifacts" / "reports"
MODELS_DIR = ROOT / "artifacts" / "models"
FINALISTS_DIR = MODELS_DIR / "finalists"
LOG_PATH = ROOT / "logs" / "application.log"

RANDOM_STATE = 42
TARGET_COLUMN = "target_tons"
BASELINE_COLUMN = "target_lag_7"

TARGET_DERIVED_COLUMNS = {
    "possible_target_outlier",
    "target_outlier",
    "is_target_outlier",
}

CAPACITY_CANDIDATES = (
    "capacity_tons",
    "daily_capacity_tons",
    "processing_capacity_tons",
    "capacity",
)


def configure_logging() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("waste_monitoring.compare_models")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


LOGGER = configure_logging()


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Не найден файл: {path}")

    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Файл не содержит строк: {path}")
    return frame


def load_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        LOGGER.warning(
            "Манифест признаков не найден; будут использованы числовые "
            "столбцы таблиц с исключением целевой переменной и служебных полей."
        )
        return {}

    with MANIFEST_PATH.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    leakage_status = manifest.get("target_leakage_check")
    if leakage_status is not None:
        valid_statuses = {"passed", "pass", "ok", "true"}
        if str(leakage_status).lower() not in valid_statuses:
            raise ValueError(
                "Проверка утечки целевого признака в манифесте не пройдена: "
                f"{leakage_status}"
            )

    return manifest


def find_manifest_features(manifest: dict[str, Any]) -> list[str]:
    accepted_keys = {
        "feature_columns",
        "features",
        "model_features",
        "predictors",
        "selected_features",
    }
    result: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key.lower() in accepted_keys:
                    if isinstance(child, list):
                        for item in child:
                            if isinstance(item, str):
                                result.append(item)
                            elif isinstance(item, dict):
                                name = item.get("name") or item.get("column")
                                if isinstance(name, str):
                                    result.append(name)
                    elif isinstance(child, dict):
                        result.extend(
                            str(name)
                            for name, enabled in child.items()
                            if enabled is True
                        )
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(manifest)
    return list(dict.fromkeys(result))


def is_id_or_date_column(column: str) -> bool:
    name = column.lower()

    if name in {
        "date",
        "datetime",
        "timestamp",
        "record_id",
        "facility_id",
        "id",
    }:
        return True

    return name.endswith("_id") or "uuid" in name or "identifier" in name


def select_features(
    frames: list[pd.DataFrame],
    manifest: dict[str, Any],
) -> list[str]:
    common_columns = set(frames[0].columns)
    for frame in frames[1:]:
        common_columns.intersection_update(frame.columns)

    manifest_features = find_manifest_features(manifest)
    if manifest_features:
        candidates = [
            name for name in manifest_features if name in common_columns
        ]
        LOGGER.info(
            "Из манифеста получено признаков: %d; присутствуют во всех выборках: %d.",
            len(manifest_features),
            len(candidates),
        )
    else:
        candidates = list(frames[0].columns)
        LOGGER.warning(
            "Список признаков не извлечён из манифеста; "
            "используется автоматический отбор."
        )

    selected: list[str] = []
    for column in candidates:
        if column not in common_columns:
            continue
        if column == TARGET_COLUMN:
            continue
        if column in TARGET_DERIVED_COLUMNS:
            continue
        if is_id_or_date_column(column):
            continue
        if "possible_target_outlier" in column.lower():
            continue
        if "target_outlier" in column.lower():
            continue

        if not all(
            pd.api.types.is_numeric_dtype(frame[column]) for frame in frames
        ):
            continue

        selected.append(column)

    selected = list(dict.fromkeys(selected))
    if not selected:
        raise ValueError(
            "Не удалось определить числовые признаки для обучения. "
            "Проверьте feature_manifest.json и подготовленные CSV-файлы."
        )

    return selected


def validate_data(
    frames: list[pd.DataFrame],
    split_names: list[str],
    feature_columns: list[str],
) -> None:
    for frame, split_name in zip(frames, split_names):
        required = [TARGET_COLUMN, *feature_columns]
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(
                f"В выборке {split_name} отсутствуют столбцы: {missing}"
            )

        if frame[TARGET_COLUMN].isna().any():
            raise ValueError(
                f"В целевом столбце {split_name} обнаружены пропуски."
            )

        if frame[feature_columns].isna().any().any():
            missing_counts = (
                frame[feature_columns].isna().sum().sort_values(ascending=False)
            )
            problem_columns = missing_counts[missing_counts > 0].to_dict()
            raise ValueError(
                f"В признаках выборки {split_name} есть пропуски: "
                f"{problem_columns}"
            )

        feature_values = frame[feature_columns].to_numpy(dtype=float)
        target_values = frame[TARGET_COLUMN].to_numpy(dtype=float)

        if not np.isfinite(feature_values).all():
            raise ValueError(
                f"В признаках выборки {split_name} есть бесконечные значения."
            )
        if not np.isfinite(target_values).all():
            raise ValueError(
                f"В целевом столбце выборки {split_name} есть "
                "бесконечные значения."
            )


def find_capacity_column(frames: list[pd.DataFrame]) -> str | None:
    common_columns = set(frames[0].columns)
    for frame in frames[1:]:
        common_columns.intersection_update(frame.columns)

    for candidate in CAPACITY_CANDIDATES:
        if candidate in common_columns:
            return candidate

    for column in common_columns:
        name = column.lower()
        if "capacity" in name or "мощност" in name:
            if all(
                pd.api.types.is_numeric_dtype(frame[column])
                for frame in frames
            ):
                return column

    return None


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    capacity: np.ndarray | None = None,
) -> dict[str, float | None]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    error = y_pred - y_true

    total_actual = float(np.abs(y_true).sum())
    wape = (
        float(np.abs(error).sum() / total_actual * 100.0)
        if total_actual > 0
        else None
    )

    metrics: dict[str, float | None] = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "wape_percent": wape,
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else None,
        "bias": float(np.mean(error)),
        "underforecast_total": float(np.maximum(-error, 0).sum()),
        "overforecast_total": float(np.maximum(error, 0).sum()),
    }

    if capacity is not None:
        capacity = np.asarray(capacity, dtype=float)
        valid = capacity > 0
        metrics["load_mae_percentage_points"] = (
            float(np.mean(np.abs(error[valid] / capacity[valid])) * 100.0)
            if valid.any()
            else None
        )
    else:
        metrics["load_mae_percentage_points"] = None

    return metrics


def predict_nonnegative(estimator: Any, x: pd.DataFrame) -> np.ndarray:
    return np.maximum(np.asarray(estimator.predict(x), dtype=float), 0.0)


def make_model_grids() -> dict[str, tuple[Any, dict[str, list[Any]]]]:
    """Возвращает модели и сетки значений для полного перебора ParameterGrid."""
    return {
        "ridge": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", Ridge()),
                ]
            ),
            {
                "model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 500.0],
            },
        ),
        "elastic_net": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        ElasticNet(
                            max_iter=5000,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            {
                "model__alpha": [0.0005, 0.002, 0.01, 0.05],
                "model__l1_ratio": [0.1, 0.5, 0.9],
            },
        ),
        "decision_tree": (
            DecisionTreeRegressor(random_state=RANDOM_STATE),
            {
                "max_depth": [6, 12, None],
                "min_samples_split": [10, 30],
                "min_samples_leaf": [3, 10],
            },
        ),
        "random_forest": (
            RandomForestRegressor(
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            {
                "n_estimators": [60, 100],
                "max_depth": [12, None],
                "min_samples_split": [5, 20],
                "min_samples_leaf": [2, 8],
                "max_features": [0.7, 1.0],
            },
        ),
        "extra_trees": (
            ExtraTreesRegressor(
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            {
                "n_estimators": [60, 100],
                "max_depth": [12, None],
                "min_samples_split": [5, 20],
                "min_samples_leaf": [2, 8],
                "max_features": [0.7, 1.0],
            },
        ),
        "hist_gradient_boosting": (
            HistGradientBoostingRegressor(
                random_state=RANDOM_STATE,
                early_stopping=False,
            ),
            {
                "learning_rate": [0.04, 0.08, 0.12],
                "max_iter": [120, 220],
                "max_leaf_nodes": [15, 31],
                "l2_regularization": [0.0, 1.0],
            },
        ),
        "gradient_boosting": (
            GradientBoostingRegressor(random_state=RANDOM_STATE),
            {
                "n_estimators": [80, 140, 200],
                "learning_rate": [0.03, 0.07],
                "max_depth": [2, 3],
                "subsample": [0.75, 1.0],
                "min_samples_leaf": [5, 15],
            },
        ),
    }


def format_candidate(family: str, parameters: dict[str, Any]) -> str:
    values = ", ".join(
        f"{key}={value}" for key, value in sorted(parameters.items())
    )
    return f"{family} | {values}"


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FINALISTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train = load_csv(TRAIN_PATH)
    validation = load_csv(VALIDATION_PATH)
    test = load_csv(TEST_PATH)
    frames = [train, validation, test]
    split_names = ["train", "validation", "test"]

    if TARGET_COLUMN not in set(train.columns):
        raise ValueError(
            f"Ожидался целевой столбец {TARGET_COLUMN!r}, "
            f"но он не найден. Столбцы train.csv: {list(train.columns)}"
        )

    manifest = load_manifest()
    feature_columns = select_features(frames, manifest)
    validate_data(frames, split_names, feature_columns)

    if BASELINE_COLUMN not in validation.columns:
        raise ValueError(
            f"Для недельного базового прогноза нужен столбец {BASELINE_COLUMN!r} "
            "в validation.csv."
        )
    if BASELINE_COLUMN not in test.columns:
        raise ValueError(
            f"Для недельного базового прогноза нужен столбец {BASELINE_COLUMN!r} "
            "в test.csv."
        )

    capacity_column = find_capacity_column(frames)
    if capacity_column:
        LOGGER.info("Столбец мощности для оценки загрузки: %s.", capacity_column)
    else:
        LOGGER.info(
            "Столбец мощности не найден; метрика загрузки будет пустой."
        )

    x_train = train[feature_columns]
    y_train = train[TARGET_COLUMN].to_numpy(dtype=float)
    x_validation = validation[feature_columns]
    y_validation = validation[TARGET_COLUMN].to_numpy(dtype=float)
    x_test = test[feature_columns]
    y_test = test[TARGET_COLUMN].to_numpy(dtype=float)

    validation_capacity = (
        validation[capacity_column].to_numpy(dtype=float)
        if capacity_column
        else None
    )
    test_capacity = (
        test[capacity_column].to_numpy(dtype=float)
        if capacity_column
        else None
    )

    # Наивный прогноз использует значение цели семидневной давности.
    naive_validation = np.maximum(
        validation[BASELINE_COLUMN].to_numpy(dtype=float),
        0.0,
    )
    naive_test = np.maximum(
        test[BASELINE_COLUMN].to_numpy(dtype=float),
        0.0,
    )
    naive_validation_metrics = calculate_metrics(
        y_validation,
        naive_validation,
        validation_capacity,
    )
    naive_test_metrics = calculate_metrics(
        y_test,
        naive_test,
        test_capacity,
    )

    model_grids = make_model_grids()
    candidates_by_family = {
        family: list(ParameterGrid(grid))
        for family, (_, grid) in model_grids.items()
    }
    total_candidates = sum(
        len(candidates) for candidates in candidates_by_family.values()
    )

    LOGGER.info(
        "Начало сравнения: алгоритмов=%d, конфигураций=%d, "
        "признаков=%d, строк train=%d.",
        len(model_grids),
        total_candidates,
        len(feature_columns),
        len(train),
    )

    validation_rows: list[dict[str, Any]] = [
        {
            "family": "weekly_naive_baseline",
            "candidate": BASELINE_COLUMN,
            "parameters": "{}",
            "validation_mae": naive_validation_metrics["mae"],
            **{
                f"validation_{key}": value
                for key, value in naive_validation_metrics.items()
            },
        }
    ]
    family_winners: dict[str, dict[str, Any]] = {}

    completed = 0
    for family, (base_estimator, _) in model_grids.items():
        best_for_family: dict[str, Any] | None = None
        candidates = candidates_by_family[family]

        LOGGER.info(
            "Алгоритм %s: конфигураций для перебора: %d.",
            family,
            len(candidates),
        )

        for parameters in candidates:
            completed += 1
            candidate_name = format_candidate(family, parameters)
            estimator = clone(base_estimator).set_params(**parameters)

            LOGGER.info(
                "[%d/%d] Обучение: %s",
                completed,
                total_candidates,
                candidate_name,
            )
            estimator.fit(x_train, y_train)

            validation_predictions = predict_nonnegative(
                estimator,
                x_validation,
            )
            metrics = calculate_metrics(
                y_validation,
                validation_predictions,
                validation_capacity,
            )

            validation_rows.append(
                {
                    "family": family,
                    "candidate": candidate_name,
                    "parameters": json.dumps(
                        parameters,
                        ensure_ascii=False,
                        default=str,
                    ),
                    "validation_mae": metrics["mae"],
                    **{
                        f"validation_{key}": value
                        for key, value in metrics.items()
                    },
                }
            )

            if (
                best_for_family is None
                or metrics["mae"] < best_for_family["validation_mae"]
            ):
                best_for_family = {
                    "family": family,
                    "candidate": candidate_name,
                    "parameters": parameters,
                    "validation_mae": metrics["mae"],
                    "validation_metrics": metrics,
                }

        if best_for_family is None:
            raise RuntimeError(f"Для алгоритма {family} нет результатов.")

        family_winners[family] = best_for_family
        LOGGER.info(
            "Победитель %s по validation MAE: %s; MAE=%.6f.",
            family,
            best_for_family["candidate"],
            best_for_family["validation_mae"],
        )

    validation_results = pd.DataFrame(validation_rows).sort_values(
        "validation_mae",
        ascending=True,
    )
    validation_results_path = REPORTS_DIR / "model_comparison_validation.csv"
    validation_results.to_csv(
        validation_results_path,
        index=False,
        encoding="utf-8-sig",
    )

    # После выбора по validation обучаем семейных победителей на train + validation.
    # Test применяется только для окончательной оценки.
    train_validation = pd.concat(
        [train, validation],
        ignore_index=True,
    )
    x_train_validation = train_validation[feature_columns]
    y_train_validation = train_validation[TARGET_COLUMN].to_numpy(dtype=float)

    final_rows: list[dict[str, Any]] = [
        {
            "family": "weekly_naive_baseline",
            "candidate": BASELINE_COLUMN,
            "parameters": "{}",
            **{
                f"validation_{key}": value
                for key, value in naive_validation_metrics.items()
            },
            **{
                f"test_{key}": value
                for key, value in naive_test_metrics.items()
            },
        }
    ]

    for family, winner in family_winners.items():
        estimator = clone(model_grids[family][0]).set_params(
            **winner["parameters"]
        )
        estimator.fit(x_train_validation, y_train_validation)

        test_predictions = predict_nonnegative(estimator, x_test)
        test_metrics = calculate_metrics(
            y_test,
            test_predictions,
            test_capacity,
        )

        final_rows.append(
            {
                "family": family,
                "candidate": winner["candidate"],
                "parameters": json.dumps(
                    winner["parameters"],
                    ensure_ascii=False,
                    default=str,
                ),
                **{
                    f"validation_{key}": value
                    for key, value in winner["validation_metrics"].items()
                },
                **{
                    f"test_{key}": value
                    for key, value in test_metrics.items()
                },
            }
        )

        model_path = FINALISTS_DIR / f"{family}.joblib"
        joblib.dump(estimator, model_path)

        metadata = {
            "family": family,
            "candidate": winner["candidate"],
            "parameters": winner["parameters"],
            "target_column": TARGET_COLUMN,
            "feature_columns": feature_columns,
            "random_state": RANDOM_STATE,
            "selection_metric": "validation_mae",
            "validation_metrics": winner["validation_metrics"],
            "test_metrics": test_metrics,
            "trained_on": "train + validation",
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        metadata_path = FINALISTS_DIR / f"{family}.json"
        with metadata_path.open("w", encoding="utf-8") as file:
            json.dump(metadata, file, ensure_ascii=False, indent=2)

    final_results = pd.DataFrame(final_rows).sort_values(
        "validation_mae",
        ascending=True,
    )
    final_results_path = REPORTS_DIR / "model_comparison_final.csv"
    final_results.to_csv(
        final_results_path,
        index=False,
        encoding="utf-8-sig",
    )

    model_results = final_results[
        final_results["family"] != "weekly_naive_baseline"
    ]
    best_row = model_results.sort_values(
        "validation_mae",
        ascending=True,
    ).iloc[0]
    best_family = str(best_row["family"])

    best_model_path = MODELS_DIR / "best_model.joblib"
    joblib.dump(
        joblib.load(FINALISTS_DIR / f"{best_family}.joblib"),
        best_model_path,
    )

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_column": TARGET_COLUMN,
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "selection_metric": "validation_mae",
        "test_used_for_selection": False,
        "evaluation_type": (
            "rolling one-step-ahead evaluation using temporal data splits"
        ),
        "random_state": RANDOM_STATE,
        "dataset_rows": {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
        },
        "model_families": len(model_grids),
        "candidate_configurations": total_candidates,
        "configurations_by_family": {
            family: len(candidates)
            for family, candidates in candidates_by_family.items()
        },
        "best_family_by_validation_mae": best_family,
        "best_candidate": str(best_row["candidate"]),
        "validation_results_csv": str(
            validation_results_path.relative_to(ROOT)
        ),
        "final_results_csv": str(final_results_path.relative_to(ROOT)),
        "best_model_path": str(best_model_path.relative_to(ROOT)),
        "baseline": {
            "validation": naive_validation_metrics,
            "test": naive_test_metrics,
        },
    }

    report_path = REPORTS_DIR / "model_comparison_report.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    LOGGER.info("Сравнение завершено.")
    LOGGER.info("Таблица перебора: %s", validation_results_path)
    LOGGER.info("Итоговая таблица: %s", final_results_path)
    LOGGER.info("Отчёт: %s", report_path)
    LOGGER.info("Лучшая модель: %s", best_model_path)
    LOGGER.info(
        "Лучшая модель по validation MAE: %s.",
        best_family,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        LOGGER.exception("Сравнение моделей завершилось с ошибкой.")
        raise