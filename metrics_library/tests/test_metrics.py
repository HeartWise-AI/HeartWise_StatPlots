import unittest
import numpy as np
from metrics_library.metrics import MetricsComputer, ClassificationMetrics, RegressionMetrics

class TestMetricsComputer(unittest.TestCase):
    def setUp(self):
        self.y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0], dtype=np.int64)
        self.y_pred = np.array([0.1, 0.9, 0.8, 0.3, 0.7, 0.2, 0.6, 0.9, 0.4, 0.1], dtype=np.float64)

    def test_check_ground_truth(self):
        # Test valid input
        MetricsComputer._MetricsComputer__check_ground_truth(self.y_true)
        
        # Test invalid input
        with self.assertRaises(ValueError):
            MetricsComputer._MetricsComputer__check_ground_truth(np.array([0, 1, 2]))
        with self.assertRaises(ValueError):
            MetricsComputer._MetricsComputer__check_ground_truth(np.array([1, 2]))

    def test_check_prediction_values(self):
        # Test valid input
        MetricsComputer._MetricsComputer__check_prediction_values(self.y_pred)
        
        # Test invalid input
        with self.assertRaises(ValueError):
            MetricsComputer._MetricsComputer__check_prediction_values(np.array([0.5, 1.1]))
        with self.assertRaises(ValueError):
            MetricsComputer._MetricsComputer__check_prediction_values(np.array([-0.1, 0.5]))

    def test_auc(self):
        auc = MetricsComputer.auc(self.y_true, self.y_pred)
        self.assertIsInstance(auc, float)
        self.assertTrue(0 <= auc <= 1)

    def test_auprc(self):
        auprc = MetricsComputer.auprc(self.y_true, self.y_pred)
        self.assertIsInstance(auprc, float)
        self.assertTrue(0 <= auprc <= 1)

    def test_confusion_matrix(self):
        cm = MetricsComputer.confusion_matrix(self.y_true, self.y_pred)
        self.assertIsInstance(cm, np.ndarray)
        self.assertEqual(cm.shape, (2, 2))

    def test_sensitivity(self):
        sensitivity = MetricsComputer.sensitivity(self.y_true, self.y_pred)
        self.assertIsInstance(sensitivity, float)
        self.assertTrue(0 <= sensitivity <= 1)

    def test_specificity(self):
        specificity = MetricsComputer.specificity(self.y_true, self.y_pred)
        self.assertIsInstance(specificity, float)
        self.assertTrue(0 <= specificity <= 1)

    def test_ppv(self):
        ppv = MetricsComputer.ppv(self.y_true, self.y_pred)
        self.assertIsInstance(ppv, float)
        self.assertTrue(0 <= ppv <= 1)

    def test_npv(self):
        npv = MetricsComputer.npv(self.y_true, self.y_pred)
        self.assertIsInstance(npv, float)
        self.assertTrue(0 <= npv <= 1)

    def test_optimal_cutoff(self):
        cutoff = MetricsComputer.optimal_cutoff(self.y_true, self.y_pred)
        self.assertIsInstance(cutoff, float)
        self.assertTrue(0 <= cutoff <= 1)

    def test_mae(self):
        mae = MetricsComputer.mae(self.y_true, self.y_pred)
        self.assertIsInstance(mae, float)
        self.assertTrue(mae >= 0)

    def test_mse(self):
        mse = MetricsComputer.mse(self.y_true, self.y_pred)
        self.assertIsInstance(mse, float)
        self.assertTrue(mse >= 0)

    def test_pearson_correlation(self):
        corr = MetricsComputer.pearson_correlation(self.y_true, self.y_pred)
        self.assertIsInstance(corr, float)
        self.assertTrue(-1 <= corr <= 1)

    def test_spearman_correlation(self):
        corr = MetricsComputer.spearman_correlation(self.y_true, self.y_pred)
        self.assertIsInstance(corr, float)
        self.assertTrue(-1 <= corr <= 1)

    def test_compute_classification_metrics(self):
        metrics = [ClassificationMetrics.AUC, ClassificationMetrics.AUPRC, RegressionMetrics.MAE]
        results = MetricsComputer.compute_classification_metrics(self.y_true, self.y_pred, metrics)
        self.assertIsInstance(results, dict)
        self.assertEqual(set(results.keys()), {'auc', 'auprc', 'mae'})

    def test_compute_regression_metrics(self):
        metrics = [RegressionMetrics.MAE, RegressionMetrics.MSE, ClassificationMetrics.AUC]
        results = MetricsComputer.compute_regression_metrics(self.y_true, self.y_pred, metrics)
        self.assertIsInstance(results, dict)
        self.assertEqual(set(results.keys()), {'mae', 'mse', 'auc'})

    def test_compute_classification_metrics_with_bootstrap(self):
        metrics = [ClassificationMetrics.AUC]
        results = MetricsComputer.compute_classification_metrics(self.y_true, self.y_pred, metrics, bootstrap=True, n_iterations=100)
        self._extracted_from_test_compute_regression_metrics_with_bootstrap_4(
            results, 'auc'
        )
            
    def test_compute_regression_metrics_with_bootstrap(self):
        metrics = [RegressionMetrics.MAE]
        results = MetricsComputer.compute_regression_metrics(self.y_true, self.y_pred, metrics, bootstrap=True, n_iterations=100)
        self._extracted_from_test_compute_regression_metrics_with_bootstrap_4(
            results, 'mae'
        )

    def _extracted_from_test_compute_regression_metrics_with_bootstrap_4(self, results, arg1):
        self.assertIsInstance(results, dict)
        self.assertIn(arg1, results)
        self.assertIn('mean', results[arg1])
        self.assertIn('ci', results[arg1])

if __name__ == '__main__':
    unittest.main()