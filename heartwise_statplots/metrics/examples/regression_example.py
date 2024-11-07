import os
import numpy as np
from heartwise_statplots.metrics import MetricsComputer, RegressionMetrics


def main():
    # Generate random regression data
    np.random.seed(42)
    y_true = np.random.rand(100).astype(np.float64)  # 100 random true values between 0 and 1
    y_pred = y_true + np.random.normal(0, 0.1, 100).astype(np.float64)  # Add some noise to the true values to get predictions

    # compute all regression metrics
    all_reg_metrics = MetricsComputer.compute_regression_metrics(
        y_true, 
        y_pred, 
        [RegressionMetrics.ALL], 
        bootstrap=True, 
        n_iterations=1000
    )
    
    # Display all regression metrics
    print("All Regression Metrics:", all_reg_metrics)

    # Display individual metrics
    print(f"MAE: {all_reg_metrics[RegressionMetrics.MAE.name.lower()]}")
    print(f"MSE: {all_reg_metrics[RegressionMetrics.MSE.name.lower()]}")

    # Compute individual metrics
    print(f"MAE: {MetricsComputer.mae(y_true, y_pred)}")
    print(f"MSE: {MetricsComputer.mse(y_true, y_pred)}")
    print(f"Pearson Correlation: {MetricsComputer.pearson_correlation(y_true, y_pred)}")
    print(f"Spearman Correlation: {MetricsComputer.spearman_correlation(y_true, y_pred)}")


if __name__ == "__main__":
    main()