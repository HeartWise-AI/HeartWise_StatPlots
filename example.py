

import os
import numpy as np
from scipy.special import expit
from heartwise_statplots.metrics import MetricsComputer, ClassificationMetrics, RegressionMetrics

mode = 'classification' # 'classification' or 'regression'

if mode == 'classification':

    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100).astype(np.int64)  # 100 random binary true values
    y_pred = np.random.rand(100).astype(np.float64)  # 100 random predicted probabilities between 0 and 1

    y_pred = expit(y_pred)

    y_pred = y_hat.to_numpy().astype(np.float64)
    y_true = outcome.to_numpy().astype(np.int64)

    all_reg_metrics = MetricsComputer.compute_classification_metrics(y_true, y_pred, [ClassificationMetrics.ALL], cutoff='youden', bootstrap=True, n_iterations=100)
    print("All Classification Metrics:", all_reg_metrics)
    print(f"AUPRC: {all_reg_metrics[ClassificationMetrics.AUPRC.name.lower()]}")
    print(f"AUC: {all_reg_metrics[ClassificationMetrics.AUC.name.lower()]}")

    print(f"PPV: {MetricsComputer.ppv(y_true, y_pred, threshold=0.5)}")
    print(f"NPV: {MetricsComputer.npv(y_true, y_pred)}")
    print(f"Optimal Cutoff: {MetricsComputer.optimal_cutoff(y_true, y_pred)}")
    
elif mode == 'regression':
    # Generate random regression data
    np.random.seed(42)
    y_true = np.random.rand(100).astype(np.float64)  # 100 random true values between 0 and 1
    y_pred = y_true + np.random.normal(0, 0.1, 100).astype(np.float64)  # Add some noise to the true values to get predictions

    print(f"MAE: {MetricsComputer.mae(y_true, y_pred)}")
    print(f"MSE: {MetricsComputer.mse(y_true, y_pred)}")
    print(f"Pearson Correlation: {MetricsComputer.pearson_correlation(y_true, y_pred)}")
    print(f"Spearman Correlation: {MetricsComputer.spearman_correlation(y_true, y_pred)}")
    
    all_reg_metrics = MetricsComputer.compute_regression_metrics(y_true, y_pred, [RegressionMetrics.ALL], bootstrap=True, n_iterations=1000)
    print("All Classification Metrics:", all_reg_metrics)
    print(f"MAE: {all_reg_metrics[RegressionMetrics.MAE.name.lower()]}")
    print(f"MSE: {all_reg_metrics[RegressionMetrics.MSE.name.lower()]}")

else:
    raise ValueError(f"Unknown mode: {mode}")