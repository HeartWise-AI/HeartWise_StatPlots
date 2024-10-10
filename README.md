# HeartWise StatPlots Metrics Library

A Python library for computing and visualizing classification and regression metrics, with a focus on medical and healthcare applications.

## Table of Contents

- [Features](#features)
  - [Classification Metrics](#classification-metrics)
  - [Regression Metrics](#regression-metrics)
- [Installation](#installation)
- [Usage](#usage)
  - [Classification Example](#classification-example)
  - [Regression Example](#regression-example)
- [Development](#development)
  - [Setting Up the Development Environment](#setting-up-the-development-environment)
  - [Code Formatting and Linting](#code-formatting-and-linting)
  - [Pre-commit Hooks](#pre-commit-hooks)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## 🚀 Features

- **Comprehensive Metric Computation**: Supports a wide range of classification and regression metrics.
- **Type-Safe Implementation**: Utilizes custom type checking to ensure data integrity and prevent errors.
- **Flexible Input Handling**: Works with NumPy arrays, supporting both integer and float data types.
- **Bootstrapping Capability**: Offers bootstrapping for robust statistical analysis and confidence interval estimation.
- **Customizable Threshold Selection**: Allows for optimal cutoff point determination in classification tasks.


### Classification Metrics

- **Area Under the Curve (AUC)**
- **Average Precision (AUPRC)**
- **Sensitivity (True Positive Rate)**
- **Specificity (True Negative Rate)**
- **Positive Predictive Value (PPV)**
- **Negative Predictive Value (NPV)**
- **Optimal Cutoff Point**

### Regression Metrics

- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **Pearson Correlation**
- **Spearman Correlation**

## 🛠️ Installation

This project uses Poetry for dependency management. To install the library and its dependencies:

1. Ensure you have Poetry installed. If not, install it by following the instructions at https://python-poetry.org/docs/#installation

2. Clone this repository:
   ```
   git clone https://github.com/HeartWise-AI/HeartWise_StatPlots.git
   cd HeartWise_StatPlots
   ```

3. Install the dependencies using Poetry:
   ```
   poetry install
   ```

## 📄 Usage

See more examples in [examples.py](examples.py)

### Classification Example

```python
import numpy as np
from metrics_library.metrics import MetricsComputer, ClassificationMetrics

# Classification example
y_true = np.array([0, 1, 1, 0, 1], dtype=np.int64)
y_pred = np.array([0.1, 0.9, 0.4, 0.3, 0.8], dtype=np.float64)

metrics = [ClassificationMetrics.AUC, ClassificationMetrics.SENSITIVITY, ClassificationMetrics.SPECIFICITY]
results = MetricsComputer.compute_classification_metrics(y_true, y_pred, metrics=metrics, bootstrap=True, n_iterations=1000)
print("Classification Metrics:", results)
```

### Regression Example

```python
import numpy as np
from metrics_library.metrics import MetricsComputer, RegressionMetrics

# Regression example
y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.int64)
y_pred = np.array([1.1, 2.2, 2.9, 3.8, 5.2], dtype=np.float64)

metrics = [RegressionMetrics.MAE, RegressionMetrics.MSE, RegressionMetrics.PEARSON_CORRELATION]
results = MetricsComputer.compute_regression_metrics(y_true, y_pred, metrics=metrics, bootstrap=True, n_iterations=1000)
print("Regression Metrics:", results)
```

### 🔧 Pre-commit Hooks

We use pre-commit hooks to ensure code quality before committing changes. To set up pre-commit hooks:

1. Install pre-commit:
   ```
   poetry add --group dev pre-commit
   ```

2. Install the git hooks:
   ```
   poetry run pre-commit install
   ```

Now, the formatting and linting checks will run automatically before each commit.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Here's how you can contribute:

1. Fork the repository
2. Create a new branch for your feature or bug fix
3. Add tests for your changes
4. Make your changes and commit them with clear, descriptive messages
5. Push your changes to your fork
6. Submit a pull request to the main repository

## 📞 Contact

For any questions, please contact:

- [Jacques Delfrate](mailto:jacques.delfrate@heartwise.ai)
