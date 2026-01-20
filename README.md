# monte_carlo_hydroponic_strawberry

Jupyter notebook prototype in VS Code with a reproducible Conda environment and clean Git diffs.

## Setup

conda env create -f environment.yml
conda run -n monte_carlo_hydroponic_strawberry python -m ipykernel install --user --name monte_carlo_hydroponic_strawberry --display-name "Python (monte_carlo_hydroponic_strawberry)"
conda run -n monte_carlo_hydroponic_strawberry pre-commit install
