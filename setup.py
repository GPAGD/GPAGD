from setuptools import setup, find_packages

setup(
    name="gpagd",
    version="0.1.0",
    author="Mohsen Mostafa",
    description="Geometric Physics‑Aware Gradient Descent Optimizer for PINNs",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "scikit-learn>=1.0.0",
        "pandas>=1.3.0",
    ],
    python_requires=">=3.9",
)
