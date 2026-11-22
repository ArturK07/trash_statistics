from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error


ROOT = Path(__file__).resolve().parent

VALIDATION_PATH = ROOT / "data" / "processed" / "validation.csv"
COMPARISON_REPORT_PATH = (
    ROOT / "artifacts" / "reports" / "model_comparison_report.json"
)
REPORTS_DIR = ROOT / "artifacts" / "reports"
LOG_PATH = ROOT / "logs" / "application.log"

MAX_VALIDATION_ROWS = 10_000
N_REPEATS = 5
RANDOM_STATE = 42
TOP_FEATURES_TO_LOG = 15


def configure_logging() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("waste_monitoring.explain_model")
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


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Не найден файл: {path}")

    with path.open("r", encoding="utf-8") as file:
        result = json.load(file)

    if not isinstance(result, dict):
        raise ValueError(f"Ожидался JSON-объект в файле: {path}")

    return result


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if not VALIDATION_PATH.exists():
        raise FileNotFoundError(f"Не найдена validation-выборка: {VALIDATION_PATH}")

    comparison_report = load_json(COMPARISON_REPORT_PATH)

    if comparison_report.get("test_used_for_selection") is not False:
        raise ValueError(
            "В отчёте сравнения не подтверждено, что test не использовался "
            "для выбора модели. Объяснение остановлено."
        )

    best_family = comparison_report.get("best_family_by_validation_mae")
    best_model_relative_path = comparison_report.get("best_model_path")
    feature_columns = comparison_report.get("feature_columns")
    target_column = comparison_report.get("target_column")

    if not best_family:
        raise ValueError(
            "В отчёте сравнения отсутствует best_family_by_validation_mae."
        )
    if not best_model_relative_path:
        raise ValueError("В отчёте сравнения отсутствует best_model_path.")
    if not feature_columns or not isinstance(feature_columns, list):
        raise ValueError("В отчёте сравнения отсутствует список признаков.")
    if not target_column:
        raise ValueError("В отчёте сравнения отсутствует целевой столбец.")

    model_path = ROOT / best_model_relative_path
    if not model_path.exists():
        raise FileNotFoundError(f"Не найдена выбранная модель: {model_path}")

    validation = pd.read_csv(VALIDATION_PATH)

    required_columns = [*feature_columns, target_column]
    missing_columns = [
        column for column in required_columns if column not in validation.columns
    ]
    if missing_columns:
        raise ValueError(
            "В validation.csv отсутствуют необходимые столбцы: "
            f"{missing_columns}"
        )

    validation = validation.dropna(subset=required_columns)
    if validation.empty:
        raise ValueError(
            "После удаления строк с пропусками validation-выборка пуста."
        )

    sample_size = min(MAX_VALIDATION_ROWS, len(validation))
    sample = validation.sample(
        n=sample_size,
        replace=False,
        random_state=RANDOM_STATE,
    )

    x_validation = sample[feature_columns]
    y_validation = sample[target_column].to_numpy(dtype=float)

    if not np.isfinite(x_validation.to_numpy(dtype=float)).all():
        raise ValueError("В признаках validation-выборки есть бесконечные значения.")
    if not np.isfinite(y_validation).all():
        raise ValueError(
            "В целевом столбце validation-выборки есть "
            "бесконечные значения."
        )

    LOGGER.info("Загрузка модели: %s", model_path)
    model = joblib.load(model_path)

    baseline_predictions = np.maximum(
        np.asarray(model.predict(x_validation), dtype=float),
        0.0,
    )
    baseline_mae = float(mean_absolute_error(y_validation, baseline_predictions))

    LOGGER.info(
        "Модель: %s; строк для объяснения: %d; признаков: %d.",
        best_family,
        len(sample),
        len(feature_columns),
    )
    LOGGER.info(
        "Вычисление permutation importance: повторов на признак=%d.",
        N_REPEATS,
    )

    importance_result = permutation_importance(
        estimator=model,
        X=x_validation,
        y=y_validation,
        scoring="neg_mean_absolute_error",
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "mae_increase_mean": importance_result.importances_mean,
            "mae_increase_std": importance_result.importances_std,
        }
    )

    # Положительное значение означает, что после перемешивания признака
    # модель в среднем стала ошибаться сильнее.
    importance["relative_mae_increase_percent"] = (
        importance["mae_increase_mean"] / baseline_mae * 100.0
        if baseline_mae > 0
        else np.nan
    )

    importance = importance.sort_values(
        "mae_increase_mean",
        ascending=False,
    ).reset_index(drop=True)
    importance.insert(0, "rank", np.arange(1, len(importance) + 1))

    csv_path = REPORTS_DIR / "model_feature_importance.csv"
    importance.to_csv(csv_path, index=False, encoding="utf-8-sig")

    top_features = importance.head(TOP_FEATURES_TO_LOG)
    results = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_family": best_family,
        "model_path": str(model_path.relative_to(ROOT)),
        "target_column": target_column,
        "method": "permutation importance",
        "scoring_metric": "MAE",
        "evaluation_split": "validation",
        "test_used": False,
        "sample_rows": len(sample),
        "validation_rows_available": len(validation),
        "feature_count": len(feature_columns),
        "n_repeats": N_REPEATS,
        "random_state": RANDOM_STATE,
        "baseline_validation_mae_on_sample": baseline_mae,
        "interpretation": (
            "mae_increase_mean — среднее увеличение MAE после перемешивания "
            "значений признака. Положительное значение указывает, что признак "
            "помогал модели на выбранной validation-выборке. При наличии "
            "коррелирующих признаков их важность может распределяться между ними."
        ),
        "top_features": top_features.to_dict(orient="records"),
        "full_results_csv": str(csv_path.relative_to(ROOT)),
    }

    json_path = REPORTS_DIR / "model_explanation_report.json"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    LOGGER.info(
        "Базовый MAE модели на выбранной части validation: %.6f.",
        baseline_mae,
    )
    LOGGER.info("Топ признаков по росту MAE после перемешивания:")
    for row in top_features.itertuples(index=False):
        LOGGER.info(
            "%2d. %s: увеличение MAE %.6f ± %.6f.",
            row.rank,
            row.feature,
            row.mae_increase_mean,
            row.mae_increase_std,
        )

    LOGGER.info("Полная таблица важности: %s", csv_path)
    LOGGER.info("JSON-отчёт: %s", json_path)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        LOGGER.exception("Объяснение модели завершилось с ошибкой.")
        raise