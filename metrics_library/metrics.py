import inspect

import numpy as np
from scipy import stats
from sklearn.metrics import auc, confusion_matrix, mean_squared_error, roc_curve


class BaseMetrics:
    def __init__(self, y_true, y_pred):
        """
        Initialize the BaseMetrics class with true and predicted values.

        Args:
            y_true (array-like): The ground truth (correct) target values.
            y_pred (array-like): The predicted target values.
        """
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)

    def _check_binary(self):
        """
        Check if the true values are binary.

        Raises:
            ValueError: If the true values are not binary.
        """
        unique_values = np.unique(self.y_true)
        if len(unique_values) != 2:
            raise ValueError("y_true must be binary for classification metrics")

    def bootstrap(self, metric_func, n_iterations=1000):
        """
        Perform bootstrap resampling to estimate the metric.

        Args:
            metric_func (function): The metric function to be evaluated.
            n_iterations (int): The number of bootstrap iterations. Default is 1000.

        Returns:
            tuple: Mean and 95% confidence interval of the metric.
        """
        results = []
        for _ in range(n_iterations):
            indices = np.random.choice(len(self.y_true), len(self.y_true), replace=True)
            y_true_sample = self.y_true[indices]
            y_pred_sample = self.y_pred[indices]

            if len(inspect.signature(metric_func).parameters) > 1:
                results.append(metric_func(y_true_sample, y_pred_sample))
            else:
                results.append(metric_func())

        return np.mean(results), np.percentile(results, [2.5, 97.5])


class ClassificationMetrics(BaseMetrics):
    """
    Initialize the ClassificationMetrics class with true and predicted values.

    This constructor sets up the necessary parameters for calculating classification metrics,
    including a threshold for binary classification. It also verifies that the true values are binary.

    Args:
        y_true: The ground truth (correct) target values.
        y_pred: The predicted target values.
        threshold: The threshold for classifying probabilities into binary outcomes. Default is 0.5.

    Raises:
        ValueError: If the true values are not binary.
    """

    def __init__(self, y_true, y_pred, threshold=0.5):
        super().__init__(y_true, y_pred)
        self._check_binary()
        self.threshold = threshold

    def _binarize_predictions(self, y_pred=None):
        """
        Binarize the predicted values based on the threshold.

        Args:
            y_pred (array-like, optional): The predicted target values. If None, use self.y_pred.

        Returns:
            array: Binarized predicted values.
        """
        if y_pred is None:
            y_pred = self.y_pred
        return (y_pred > self.threshold).astype(int)

    def compute_auc(self, y_true=None, y_pred=None):
        """
        Compute the Area Under the Curve (AUC) for the given true and predicted values.

        Args:
            y_true (array-like, optional): The ground truth (correct) target values. If None, use self.y_true.
            y_pred (array-like, optional): The predicted target values. If None, use self.y_pred.

        Returns:
            float: AUC score.
        """
        if y_true is None:
            y_true = self.y_true
        if y_pred is None:
            y_pred = self.y_pred
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        return auc(fpr, tpr)

    def compute_confusion_matrix(self, y_true=None, y_pred=None):
        """
        Compute the confusion matrix for the given true and predicted values.

        Args:
            y_true (array-like, optional): The ground truth (correct) target values. If None, use self.y_true.
            y_pred (array-like, optional): The predicted target values. If None, use self.y_pred.

        Returns:
            ndarray: Confusion matrix.
        """
        if y_true is None:
            y_true = self.y_true
        if y_pred is None:
            y_pred = self.y_pred
        y_pred_binary = self._binarize_predictions(y_pred)
        return confusion_matrix(y_true, y_pred_binary)

    def compute_sensitivity(self, y_true=None, y_pred=None):
        """
        Compute the sensitivity (recall) for the given true and predicted values.

        Args:
            y_true (array-like, optional): The ground truth (correct) target values. If None, use self.y_true.
            y_pred (array-like, optional): The predicted target values. If None, use self.y_pred.

        Returns:
            float: Sensitivity score.
        """
        tn, fp, fn, tp = self.compute_confusion_matrix(y_true, y_pred).ravel()
        return tp / (tp + fn) if (tp + fn) > 0 else 0

    def compute_specificity(self, y_true=None, y_pred=None):
        """
        Compute the specificity for the given true and predicted values.

        Args:
            y_true (array-like, optional): The ground truth (correct) target values. If None, use self.y_true.
            y_pred (array-like, optional): The predicted target values. If None, use self.y_pred.

        Returns:
            float: Specificity score.
        """
        tn, fp, fn, tp = self.compute_confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0

    def compute_ppv(self, y_true=None, y_pred=None):
        """
        Compute the Positive Predictive Value (PPV) for the given true and predicted values.

        Args:
            y_true (array-like, optional): The ground truth (correct) target values. If None, use self.y_true.
            y_pred (array-like, optional): The predicted target values. If None, use self.y_pred.

        Returns:
            float: PPV score.
        """
        tn, fp, fn, tp = self.compute_confusion_matrix(y_true, y_pred).ravel()
        return tp / (tp + fp) if (tp + fp) > 0 else 0

    def compute_npv(self, y_true=None, y_pred=None):
        """
        Compute the Negative Predictive Value (NPV) for the given true and predicted values.

        Args:
            y_true (array-like, optional): The ground truth (correct) target values. If None, use self.y_true.
            y_pred (array-like, optional): The predicted target values. If None, use self.y_pred.

        Returns:
            float: NPV score.
        """
        tn, fp, fn, tp = self.compute_confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fn) if (tn + fn) > 0 else 0

    def find_optimal_cutoff(self, y_true=None, y_pred=None):
        """
        Find the optimal cutoff threshold for the given true and predicted values.

        Args:
            y_true (array-like, optional): The ground truth (correct) target values. If None, use self.y_true.
            y_pred (array-like, optional): The predicted target values. If None, use self.y_pred.

        Returns:
            float: Optimal cutoff threshold.
        """
        if y_true is None:
            y_true = self.y_true
        if y_pred is None:
            y_pred = self.y_pred
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        optimal_idx = np.argmax(tpr - fpr)
        return thresholds[optimal_idx]


class RegressionMetrics(BaseMetrics):
    def compute_mae(self, y_true=None, y_pred=None):
        """
        Compute the Mean Absolute Error (MAE) for the given true and predicted values.

        Args:
            y_true (array-like, optional): The ground truth (correct) target values. If None, use self.y_true.
            y_pred (array-like, optional): The predicted target values. If None, use self.y_pred.

        Returns:
            float: MAE score.
        """
        if y_true is None:
            y_true = self.y_true
        if y_pred is None:
            y_pred = self.y_pred
        return np.mean(np.abs(y_true - y_pred))

    def compute_mse(self, y_true=None, y_pred=None):
        """
        Compute the Mean Squared Error (MSE) for the given true and predicted values.

        Args:
            y_true (array-like, optional): The ground truth (correct) target values. If None, use self.y_true.
            y_pred (array-like, optional): The predicted target values. If None, use self.y_pred.

        Returns:
            float: MSE score.
        """
        if y_true is None:
            y_true = self.y_true
        if y_pred is None:
            y_pred = self.y_pred
        return mean_squared_error(y_true, y_pred)

    def compute_pearson_correlation(self, y_true=None, y_pred=None):
        """
        Compute the Pearson correlation coefficient for the given true and predicted values.

        Args:
            y_true (array-like, optional): The ground truth (correct) target values. If None, use self.y_true.
            y_pred (array-like, optional): The predicted target values. If None, use self.y_pred.

        Returns:
            float: Pearson correlation coefficient.
        """
        if y_true is None:
            y_true = self.y_true
        if y_pred is None:
            y_pred = self.y_pred
        return stats.pearsonr(y_true, y_pred)[0]

    def compute_spearman_correlation(self, y_true=None, y_pred=None):
        """
        Compute the Spearman correlation coefficient for the given true and predicted values.

        Args:
            y_true (array-like, optional): The ground truth (correct) target values. If None, use self.y_true.
            y_pred (array-like, optional): The predicted target values. If None, use self.y_pred.

        Returns:
            float: Spearman correlation coefficient.
        """
        if y_true is None:
            y_true = self.y_true
        if y_pred is None:
            y_pred = self.y_pred
        return stats.spearmanr(y_true, y_pred)[0]


class MetricsComputer:
    def __init__(self, y_true, y_pred, task="classification", threshold=0.5):
        """
        Initialize the MetricsComputer class with true and predicted values and the task type.

        Args:
            y_true (array-like): The ground truth (correct) target values.
            y_pred (array-like): The predicted target values.
            task (str): The type of task, either 'classification' or 'regression'. Default is 'classification'.
            threshold (float): The threshold for classifying probabilities into binary outcomes. Default is 0.5.

        Raises:
            ValueError: If the task is not 'classification' or 'regression'.
        """
        self.task = task.lower()
        if self.task == "classification":
            self.metrics = ClassificationMetrics(y_true, y_pred, threshold)
        elif self.task == "regression":
            self.metrics = RegressionMetrics(y_true, y_pred)
        else:
            raise ValueError("Task must be either 'classification' or 'regression'")

    def compute_metrics(self, bootstrap=False, n_iterations=1000):
        """
        Compute the metrics for the given task.

        Args:
            bootstrap (bool): Whether to use bootstrap resampling. Default is False.
            n_iterations (int): The number of bootstrap iterations. Default is 1000.

        Returns:
            dict: A dictionary containing the computed metrics.
        """
        if self.task == "classification":
            metrics = {
                "auc": self.metrics.compute_auc,
                "sensitivity": self.metrics.compute_sensitivity,
                "specificity": self.metrics.compute_specificity,
                "ppv": self.metrics.compute_ppv,
                "npv": self.metrics.compute_npv,
                "optimal_cutoff": self.metrics.find_optimal_cutoff,
            }
        else:  # regression
            metrics = {
                "mae": self.metrics.compute_mae,
                "mse": self.metrics.compute_mse,
                "pearson_correlation": self.metrics.compute_pearson_correlation,
                "spearman_correlation": self.metrics.compute_spearman_correlation,
            }

        results = {}
        for name, func in metrics.items():
            if bootstrap and name != "optimal_cutoff":
                mean, ci = self.metrics.bootstrap(func, n_iterations)
                results[name] = {"mean": mean, "ci": ci}
            else:
                results[name] = func()

        return results

    def compute_auc(
        self,
        data=None,
        y_true_cat=None,
        y_hat_cat=None,
        y_hat=None,
        invert_classes=False,
    ):
        """
        Compute the Area Under the Curve (AUC) for the given data.

        Args:
            data (array-like, optional): The data to be used for computing AUC.
            y_true_cat (array-like, optional): The ground truth (correct) target values for categorical data.
            y_hat_cat (array-like, optional): The predicted target values for categorical data.
            y_hat (array-like, optional): The predicted target values.
            invert_classes (bool, optional): Whether to invert the classes. Default is False.

        Returns:
            float: AUC score.

        Raises:
            ValueError: If the task is not 'classification'.
        """
        if self.task != "classification":
            raise ValueError("AUC can only be computed for classification tasks")
        return self.metrics.compute_auc(
            data, y_true_cat, y_hat_cat, y_hat, invert_classes
        )
