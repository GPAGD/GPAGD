This is a Jupyter notebook that demonstrates GPAGD on a simple Poisson 1D problem. You can create it by copying your existing Colab notebook content and removing the CIFAR‑10 part, 
then saving as .ipynb. For brevity, I'll provide a minimal version.

How to Use These Files

    Create the folder structure as shown above.

    Copy each code block into the corresponding file.

    Replace YOUR_USERNAME in README.md and the notebook with your GitHub username.

    Initialize git and push to your GitHub repository.

After pushing, users can install GPAGD with:

```python
pip install git+https://github.com/YOUR_USERNAME/GPAGD-Optimizer.git
```

And run the benchmarks:

```python
cd GPAGD-Optimizer/experiments
python run_benchmarks.py
```
