import inspect
import numpy as np

from tqdm import tqdm
from scipy import stats
from enum import Enum, auto
from typing import Callable
from sklearn.metrics import (
    auc,
    confusion_matrix,
    mean_squared_error,
    roc_curve,
    average_precision_score,
)

from metrics_library.type_check import type_check


class ClassificationMetrics(Enum):
    AUC = auto()
    AUPRC = auto()
    SENSITIVITY = auto()
    SPECIFICITY = auto()
    PPV = auto()
    NPV = auto()
    OPTIMAL_CUTOFF = auto()
    ALL = auto()


class RegressionMetrics(Enum):
    MAE = auto()
    MSE = auto()
    PEARSON_CORRELATION = auto()
    SPEARMAN_CORRELATION = auto()
    ALL = auto()


class MetricsComputer:
    @classmethod
    @type_check(enabled=True, dtypes={"y_true": np.int64}, y_true=np.ndarray)
    def __check_ground_truth(cls, y_true: np.ndarray):
        unique_values = np.unique(y_true)
        if len(unique_values) > 2:
            raise ValueError("y_true must be binary for classification metrics")
        if unique_values[0] != 0 and unique_values[1] != 1:
            raise ValueError("y_true values must be equal to 0 or 1")

    @classmethod
    @type_check(enabled=True, dtypes={"y_pred": np.float64}, y_pred=np.ndarray)
    def __check_prediction_values(cls, y_pred: np.ndarray):
        if np.max(y_pred) > 1 or np.min(y_pred) < 0:
            raise ValueError("y_pred values must be between 0 and 1")

    @classmethod
    @type_check(
        enabled=True,
        y_true=np.ndarray,
        y_pred=np.ndarray,
        dtypes={"y_true": np.int64, "y_pred": np.float64},
        metric_func=Callable,
        n_iterations=int,
    )
    def __bootstrap(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metric_func: Callable,
        n_iterations: int = 1000,
    ) -> tuple[float, tuple[float, float]]:
        results = []
        for _ in tqdm(range(n_iterations), desc="Bootstrapping"):
            indices = np.random.choice(len(y_true), size=len(y_true), replace=True)
            y_true_sample = y_true[indices]
            y_pred_sample = y_pred[indices]
            results.append(metric_func(y_true_sample, y_pred_sample))
        return np.mean(results), np.percentile(results, [2.5, 97.5])

    # Classification metrics
    @classmethod
    @type_check(
        enabled=True,
        y_true=np.ndarray,
        y_pred=np.ndarray,
        dtypes={"y_true": np.int64, "y_pred": np.float64},
    )
    def auc(cls, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        cls.__check_ground_truth(y_true)
        cls.__check_prediction_values(y_pred)
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        return auc(fpr, tpr)

    @classmethod
    @type_check(
        enabled=True,
        y_true=np.ndarray,
        y_pred=np.ndarray,
        dtypes={"y_true": np.int64, "y_pred": np.float64},
    )
    def auprc(cls, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        cls.__check_ground_truth(y_true)
        cls.__check_prediction_values(y_pred)
        return average_precision_score(y_true, y_pred)

    @classmethod
    @type_check(
        enabled=True,
        y_true=np.ndarray,
        y_pred=np.ndarray,
        dtypes={"y_true": np.int64, "y_pred": np.float64},
        threshold=float,
    )
    def confusion_matrix(
        cls, y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5
    ) -> np.ndarray:
        cls.__check_ground_truth(y_true)
        cls.__check_prediction_values(y_pred)
        y_pred_binary = (y_pred > threshold).astype(int)
        return confusion_matrix(y_true, y_pred_binary)

    @classmethod
    @type_check(
        enabled=True,
        y_true=np.ndarray,
        y_pred=np.ndarray,
        dtypes={"y_true": np.int64, "y_pred": np.float64},
        threshold=float,
    )
    def sensitivity(
        cls, y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5
    ) -> float:
        cls.__check_ground_truth(y_true)
        cls.__check_prediction_values(y_pred)
        try:
            tn, fp, fn, tp = cls.confusion_matrix(y_true, y_pred, threshold).ravel()
            return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        except ValueError:
            return 0.0

    @classmethod
    @type_check(
        enabled=True,
        y_true=np.ndarray,
        y_pred=np.ndarray,
        dtypes={"y_true": np.int64, "y_pred": np.float64},
        threshold=float,
    )
    def specificity(
        cls, y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5
    ) -> float:
        cls.__check_ground_truth(y_true)
        cls.__check_prediction_values(y_pred)
        try:
            tn, fp, fn, tp = cls.confusion_matrix(y_true, y_pred, threshold).ravel()
            return float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        except ValueError:
            return 0.0

    @classmethod
    @type_check(
        enabled=True,
        y_true=np.ndarray,
        y_pred=np.ndarray,
        dtypes={"y_true": np.int64, "y_pred": np.float64},
        threshold=float,
    )
    def ppv(
        cls, y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5
    ) -> float:
        cls.__check_ground_truth(y_true)
        cls.__check_prediction_values(y_pred)
        try:
            tn, fp, fn, tp = cls.confusion_matrix(y_true, y_pred, threshold).ravel()
            return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        except ValueError:
            return 0.0

    @classmethod
    @type_check(
        enabled=True,
        y_true=np.ndarray,
        y_pred=np.ndarray,
        dtypes={"y_true": np.int64, "y_pred": np.float64},
        threshold=float,
    )
    def npv(
        cls, y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5
    ) -> float:
        cls.__check_ground_truth(y_true)
        cls.__check_prediction_values(y_pred)
        try:
            tn, fp, fn, tp = cls.confusion_matrix(y_true, y_pred, threshold).ravel()
            return float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
        except ValueError:
            return 0.0

    @classmethod
    @type_check(
        enabled=True,
        y_true=np.ndarray,
        y_pred=np.ndarray,
        dtypes={"y_true": np.int64, "y_pred": np.float64},
    )
    def optimal_cutoff(cls, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        cls.__check_ground_truth(y_true)
        cls.__check_prediction_values(y_pred)
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred)
        # Compute Youden Index
        youden_index = tpr - fpr
        optimal_idx = roc_thresholds[np.argmax(youden_index)]
        return float(optimal_idx)

    # Regression metrics
    @classmethod
    @type_check(
        enabled=True,
        y_true=np.ndarray,
        y_pred=np.ndarray,
        dtypes={"y_true": np.int64, "y_pred": np.float64},
    )
    def mae(cls, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(np.abs(y_true - y_pred)))

    @classmethod
    @type_check(
        enabled=True,
        y_true=np.ndarray,
        y_pred=np.ndarray,
        dtypes={"y_true": np.int64, "y_pred": np.float64},
    )
    def mse(cls, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(mean_squared_error(y_true, y_pred))

    @classmethod
    @type_check(
        enabled=True,
        y_true=np.ndarray,
        y_pred=np.ndarray,
        dtypes={"y_true": np.int64, "y_pred": np.float64},
    )
    def pearson_correlation(cls, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(stats.pearsonr(y_true, y_pred)[0])

    @classmethod
    @type_check(
        enabled=True,
        y_true=np.ndarray,
        y_pred=np.ndarray,
        dtypes={"y_true": np.int64, "y_pred": np.float64},
    )
    def spearman_correlation(cls, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(stats.spearmanr(y_true, y_pred)[0])

    @classmethod
    @type_check(
        enabled=True,
        y_true=np.ndarray,
        y_pred=np.ndarray,
        dtypes={"y_true": np.int64, "y_pred": np.float64},
        metrics=list,
        cutoff=str,
        bootstrap=bool,
        n_iterations=int,
    )
    def compute_classification_metrics(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metrics: list,
        cutoff: str = "default",
        bootstrap: bool = False,
        n_iterations: int = 1000,
    ) -> dict:
        if cutoff not in ["youden", "default"]:
            raise ValueError("Cutoff must be 'youden' or 'default'")

        actual_cutoff = 0.5
        if cutoff == "youden":
            actual_cutoff = cls.optimal_cutoff(y_true, y_pred)

        metrics_to_compute = []
        for metric in metrics:
            if metric == ClassificationMetrics.ALL:
                metrics_to_compute.extend(
                    [m for m in ClassificationMetrics if m != ClassificationMetrics.ALL]
                )
            else:
                metrics_to_compute.append(metric)

        results = {}
        for metric in metrics_to_compute:
            func = getattr(MetricsComputer, metric.name.lower())

            # Inspect the signature of the metric function
            sig = inspect.signature(func)
            func_params = sig.parameters

            # Prepare arguments based on the function's parameters
            kwargs = {}
            if "threshold" in func_params:
                kwargs["threshold"] = actual_cutoff

            if bootstrap and metric != ClassificationMetrics.OPTIMAL_CUTOFF:

                def bootstrap_func(y_t, y_p):
                    return func(y_t, y_p, **kwargs)

                mean, ci = cls.__bootstrap(y_true, y_pred, bootstrap_func, n_iterations)
                results[metric.name.lower()] = {"mean": mean, "ci": ci}
            else:
                results[metric.name.lower()] = func(y_true, y_pred, **kwargs)

        return results

    @classmethod
    @type_check(
        enabled=True,
        y_true=np.ndarray,
        y_pred=np.ndarray,
        dtypes={"y_true": np.int64, "y_pred": np.float64},
        metrics=list,
        bootstrap=bool,
        n_iterations=int,
    )
    def compute_regression_metrics(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metrics: list,
        bootstrap: bool = False,
        n_iterations: int = 1000,
    ) -> dict:
        results = {}
        metrics_to_compute = []

        for metric in metrics:
            if metric == RegressionMetrics.ALL:
                metrics_to_compute.extend(
                    [m for m in RegressionMetrics if m != RegressionMetrics.ALL]
                )
            else:
                metrics_to_compute.append(metric)

        for metric in metrics_to_compute:
            if isinstance(metric, (ClassificationMetrics, RegressionMetrics)):
                func = getattr(MetricsComputer, metric.name.lower())
            else:
                raise ValueError(f"Unknown metric type: {type(metric)}")

            if (
                bootstrap
                and metric != RegressionMetrics.PEARSON_CORRELATION
                and metric != RegressionMetrics.SPEARMAN_CORRELATION
            ):
                mean, ci = cls.__bootstrap(y_true, y_pred, func, n_iterations)
                results[metric.name.lower()] = {"mean": mean, "ci": ci}
            else:
                results[metric.name.lower()] = func(y_true, y_pred)

        return results
