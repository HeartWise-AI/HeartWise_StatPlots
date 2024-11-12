

import os
import numpy as np
from heartwise_statplots.metrics import MetricsComputer, ClassificationMetrics

def main():
    # create random binary true and predicted values
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100).astype(np.int64)  # 100 random binary true values
    y_pred = np.random.rand(100).astype(np.float64)  # 100 random predicted probabilities between 0 and 1

    # compute all classification metrics
    all_reg_metrics = MetricsComputer.compute_classification_metrics(
        y_true, 
        y_pred, 
        [ClassificationMetrics.ALL], 
        cutoff='youden', # available options: youden, default (0.5) 
        bootstrap=True, 
        n_iterations=100
    )
    # Display all classification metrics
    print("All Classification Metrics:", all_reg_metrics)
    
    # Display individual metrics
    print(f"AUPRC: {all_reg_metrics[ClassificationMetrics.AUPRC.name.lower()]}")
    print(f"AUC: {all_reg_metrics[ClassificationMetrics.AUC.name.lower()]}")

    # Compute individual metrics
    print(f"PPV: {MetricsComputer.ppv(y_true, y_pred, threshold=0.5)}")
    print(f"NPV: {MetricsComputer.npv(y_true, y_pred)}")
    print(f"Optimal Cutoff: {MetricsComputer.optimal_cutoff(y_true, y_pred)}")

if __name__ == "__main__":
    main()