# Metrics Library

A dataframe-agnostic Python library for computing classification and regression metrics.

## Features

- Supports both classification and regression tasks
- Dataframe-agnostic: works with lists, numpy arrays, pandas Series, etc.
- Easy-to-use interface with a single `MetricsComputer` class
- Utilizes Poetry for dependency management and packaging

### Classification Metrics

- Area Under the Curve (AUC)
- Sensitivity (True Positive Rate)
- Specificity (True Negative Rate)
- Positive Predictive Value (PPV)
- Negative Predictive Value (NPV)
- Optimal Cutoff Point

### Regression Metrics

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Pearson Correlation
- Spearman Correlation

## Installation

This project uses Poetry for dependency management. To install the library and its dependencies:

1. Ensure you have Poetry installed. If not, install it by following the instructions at https://python-poetry.org/docs/#installation

2. Clone this repository:
   ```
   git clone https://github.com/yourusername/metrics_library.git
   cd metrics_library
   ```

3. Install the dependencies using Poetry:
   ```
   poetry install
   ```

## Usage

Here's a simple example of how to use the Metrics Library:

```python
from metrics_library.metrics import MetricsComputer
import numpy as np

# Classification example
y_true_class = np.array([0, 1, 1, 0, 1])
y_pred_class = np.array([0, 1, 0, 0, 1])
y_pred_proba = np.array([0.1, 0.9, 0.4, 0.3, 0.8])

class_metrics = MetricsComputer(y_true_class, y_pred_class, task='classification', y_pred_proba=y_pred_proba)
class_results = class_metrics.compute_metrics()
print("Classification Metrics:", class_results)

# Regression example
y_true_reg = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_pred_reg = np.array([1.1, 2.2, 2.9, 3.8, 5.2])

reg_metrics = MetricsComputer(y_true_reg, y_pred_reg, task='regression')
reg_results = reg_metrics.compute_metrics()
print("Regression Metrics:", reg_results)
```

You can also run the example script directly using Poetry:

```
poetry run metrics-example
```

For more detailed examples, see the `main.py` file in the `metrics_library` directory.

## Development

To set up the development environment:

1. Clone the repository (if you haven't already):
   ```
   git clone https://github.com/yourusername/metrics_library.git
   cd metrics_library
   ```

2. Install the dependencies, including development dependencies:
   ```
   poetry install
   ```

3. Activate the virtual environment:
   ```
   poetry shell
   ```

4. Run tests:
   ```
   pytest
   ```

### Code Formatting and Linting

We use several tools to maintain code quality and consistency:

- Black for code formatting
- isort for sorting imports
- pylint for linting
- mypy for static type checking

You can run these tools using the following commands:

```
poetry run black .
poetry run isort .
poetry run pylint metrics_library tests
poetry run mypy metrics_library tests
```

Alternatively, you can use the provided Makefile:

```
make format  # Runs black and isort
make lint    # Runs pylint
make test    # Runs pytest
make all     # Runs format, lint, and test
```

### Pre-commit Hooks

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

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Here's how you can contribute:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Please make sure to update tests as appropriate and adhere to the code formatting guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to all contributors who have helped shape this project
- Inspired by the need for a flexible, dataframe-agnostic metrics library

## Contact

Your Name - your.email@example.com

Project Link: [https://github.com/yourusername/metrics_library](https://github.com/yourusername/metrics_library)
