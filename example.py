

import os
import numpy as np
from metrics_library.metrics import MetricsComputer, ClassificationMetrics, RegressionMetrics

root = '/media/data1/achilsowa/for_jacques/'

y_pred = np.load(os.path.join(root, 'y_pred_lqts_type.npy'))
y_true = np.load(os.path.join(root, 'y_true_lqts_type.npy'))

mode = 'classification' # 'classification' or 'regression'

if mode == 'classification':

    def stable_sigmoid(x):
        x_safe = np.clip(x, -500, 500)
        return np.where(x_safe >= 0, 
                        1 / (1 + np.exp(-x_safe)), 
                        np.exp(x_safe) / (1 + np.exp(x_safe)))

    y_pred = stable_sigmoid(y_pred)

    y_pred = y_pred.astype(np.float64)
    y_true = y_true.astype(np.int64)

    all_reg_metrics = MetricsComputer.compute_classification_metrics(y_true, y_pred, [ClassificationMetrics.ALL], cutoff='default', bootstrap=True, n_iterations=1000)
    print("All Classification Metrics:", all_reg_metrics)
    print(f"AUPRC: {all_reg_metrics[ClassificationMetrics.AUPRC.name.lower()]}")
    print(f"AUC: {all_reg_metrics[ClassificationMetrics.AUC.name.lower()]}")

    print(f"PPV: {MetricsComputer.ppv(y_true, y_pred, threshold=0.5)}")
    print(f"NPV: {MetricsComputer.npv(y_true, y_pred)}")
    print(f"Optimal Cutoff: {MetricsComputer.optimal_cutoff(y_true, y_pred)}")
    
elif mode == 'regression':
    all_reg_metrics = MetricsComputer.compute_regression_metrics(y_true, y_pred, [RegressionMetrics.ALL], bootstrap=True, n_iterations=1000)
    print("All Classification Metrics:", all_reg_metrics)
    print(f"MAE: {all_reg_metrics[RegressionMetrics.MAE.name.lower()]}")
    print(f"MSE: {all_reg_metrics[RegressionMetrics.MSE.name.lower()]}")

else:
    raise ValueError(f"Unknown mode: {mode}")