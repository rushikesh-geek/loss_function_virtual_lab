# datasets/__init__.py — expose dataset loaders cleanly
from datasets.mnist_loader import load_mnist
from datasets.binary_loader import load_breast_cancer_data
from datasets.regression_loader import load_california_housing_data
from datasets.fraud_loader import load_fraud_data

__all__ = [
    "load_mnist",
    "load_breast_cancer_data",
    "load_california_housing_data",
    "load_fraud_data",
]
