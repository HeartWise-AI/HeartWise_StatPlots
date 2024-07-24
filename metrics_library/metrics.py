import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix, mean_squared_error
from scipy import stats

class BaseMetrics:
    def __init__(self, y_true, y_pred):
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)

    def _check_binary(self):
        unique_values = np.unique(self.y_true)
        if len(unique_values) != 2:
            raise ValueError("y_true must be binary for classification metrics")

class ClassificationMetrics(BaseMetrics):
    def __init__(self, y_true, y_pred, y_pred_proba=None):
        super().__init__(y_true, y_pred)
        self._check_binary()
        self.y_pred_proba = y_pred_proba if y_pred_proba is not None else y_pred

    def compute_auc(self):
        fpr, tpr, _ = roc_curve(self.y_true, self.y_pred_proba)
        return auc(fpr, tpr)

    def compute_confusion_matrix(self):
        return confusion_matrix(self.y_true, self.y_pred)

    def compute_sensitivity_specificity(self):
        tn, fp, fn, tp = self.compute_confusion_matrix().ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        return sensitivity, specificity

    def compute_ppv_npv(self):
        tn, fp, fn, tp = self.compute_confusion_matrix().ravel()
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        return ppv, npv

    def find_optimal_cutoff(self):
        fpr, tpr, thresholds = roc_curve(self.y_true, self.y_pred_proba)
        optimal_idx = np.argmax(tpr - fpr)
        return thresholds[optimal_idx]

class RegressionMetrics(BaseMetrics):
    def compute_mae(self):
        return np.mean(np.abs(self.y_true - self.y_pred))

    def compute_mse(self):
        return mean_squared_error(self.y_true, self.y_pred)

    def compute_pearson_correlation(self):
        return stats.pearsonr(self.y_true, self.y_pred)[0]

    def compute_spearman_correlation(self):
        return stats.spearmanr(self.y_true, self.y_pred)[0]

class MetricsComputer:
    def __init__(self, y_true, y_pred, task='classification', y_pred_proba=None):
        self.task = task.lower()
        if self.task == 'classification':
            self.metrics = ClassificationMetrics(y_true, y_pred, y_pred_proba)
        elif self.task == 'regression':
            self.metrics = RegressionMetrics(y_true, y_pred)
        else:
            raise ValueError("Task must be either 'classification' or 'regression'")

    def compute_metrics(self):
        if self.task == 'classification':
            return {
                'auc': self.metrics.compute_auc(),
                'sensitivity': self.metrics.compute_sensitivity_specificity()[0],
                'specificity': self.metrics.compute_sensitivity_specificity()[1],
                'ppv': self.metrics.compute_ppv_npv()[0],
                'npv': self.metrics.compute_ppv_npv()[1],
                'optimal_cutoff': self.metrics.find_optimal_cutoff()
            }
        else:  # regression
            return {
                'mae': self.metrics.compute_mae(),
                'mse': self.metrics.compute_mse(),
                'pearson_correlation': self.metrics.compute_pearson_correlation(),
                'spearman_correlation': self.metrics.compute_spearman_correlation()
            }
