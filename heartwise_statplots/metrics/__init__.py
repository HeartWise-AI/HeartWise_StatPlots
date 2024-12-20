import numpy as np

from heartwise_statplots.metrics.metrics import MetricsComputer, ClassificationMetrics, RegressionMetrics

__all__ = ["MetricsComputer", "ClassificationMetrics", "RegressionMetrics"]


def main():
    """Demonstrate the functionality of the Metrics Library.

    This function provides examples of how to use the Metrics Library for both classification
    and regression tasks. It initializes the metrics computer with sample data, computes the
    relevant metrics, and prints the results to the console.

    Args:
        None

    Returns:
        None

    Examples:
        Running the main function will display classification and regression metrics based on
        predefined true and predicted values.
    """

    print("Demonstrating the Metrics Library")
    print("\nClassification Example:")
    y_true_class = np.array([0, 1, 1, 0, 1, 1, 0, 0, 1, 0])
    y_pred_class = np.array([0, 1, 0, 0, 1, 1, 0, 1, 1, 0])
    y_pred_proba = np.array([0.1, 0.9, 0.4, 0.3, 0.8, 0.7, 0.2, 0.6, 0.9, 0.1])

    class_metrics = MetricsComputer(y_true_class, y_pred_class, task="classification")
    class_results = class_metrics.compute_metrics()

    for metric, value in class_results.items():
        print(f"{metric}: {value}")

    print("\nRegression Example:")
    y_true_reg = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    y_pred_reg = np.array([1.1, 2.2, 2.9, 3.8, 5.2, 6.1, 6.8, 7.9, 9.2, 9.9])

    reg_metrics = MetricsComputer(y_true_reg, y_pred_reg, task="regression")
    reg_results = reg_metrics.compute_metrics()

    for metric, value in reg_results.items():
        print(f"{metric}: {value}")


if __name__ == "__main__":
    main()
