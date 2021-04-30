# cadCAD Edu Eth2 MasterClass
[![Python package](https://github.com/cadCAD-edu/eth2.0-masterclass/actions/workflows/python.yml/badge.svg)](https://github.com/cadCAD-edu/eth2.0-masterclass/actions/workflows/python.yml)

A post-merge Eth2 Validator Economics model.

**Eth2 specs version**: implements the [Altair](https://github.com/ethereum/eth2.0-specs#altair) updates in the [Beige Gorgon / v1.1.0-alpha.3](https://github.com/ethereum/eth2.0-specs/releases/tag/v1.1.0-alpha.3) release.

Features:
* Enables both State Space (simulation over time) and Phase Space (analysis of system state) simulations and analyses
* Customizable processes for ETH price, ETH staked, validator adoption, EIP1559, and transaction rates
* Modular model structure
* Referrences to [Eth2 specs](https://github.com/ethereum/eth2.0-specs) in Policy and State Update Function logic

## Table of Contents
* [Experiments](#experiments)
* [Development](#development)
* [Tests](#tests)
* [Jupyter Lab Environment](#jupyter-lab-environment)
* [Change Log](#change-log)
* [Contributors](#contributors)

## Directory Structure
* [data/](data/): datasets used in model (e.g. for stochastic processes)
* [docs/](docs/): work-in-progress documentation of model software architecture
* [experiments/](experiments/): experiment workflow configuration and execution
* [logs/](logs/): experiment log files
* [model/](model/): model structure, parts, and configuration
* [notebooks/](notebooks/): experiment analysis notebooks
* [outputs/](outputs/): experiment outputs (images, datasets, etc.)
* [tests/](tests/): unit and integration tests for model and notebooks

---

## Model

### Parts

The model is composed of different modules in the [model/parts/](model/parts/) directory:

| Module | Description |
| --- | --- |
| [ethereum.py](model/parts/ethereum.py) | Genereal Ethereum mechanisms, such as the EIP1559 transaction pricing mechanism, and updating the ETH price and ETH supply |
| [incentives.py](model/parts/incentives.py) | Proof of Stake incentives |
| [metrics.py](model/parts/metrics.py) | Calculation of validator costs, revenue, profit, and yield metrics |
| [phases.py](model/parts/phases.py) | Management of phases of the Ethereum system upgrade process |
| [spec.py](model/parts/spec.py) | Relevant extracts from the Eth2 spec |
| [validators.py](model/parts/validators.py) | Validator processes such as validator activation, staking, uptime |

### Configuration

The model is configured using modules in the [model/](model/) directory:

| Module | Description |
| --- | --- |
| [constants.py](model/constants.py) ||
| [parameters.py](model/parameters.py) ||
| [simulation_configuration.py](model/simulation_configuration.py) ||
| [state_update_blocks.py](model/state_update_blocks.py) ||
| [state_variables.py](model/state_variables.py) ||
| [types.py](model/types.py) ||
| [utils.py](model/utils.py) ||

## Experiments

The [experiments/](experiments/) directory contains different experiment configurations, where each experiment has a corresponding analysis Jupyter notebook in the [notebooks/](notebooks/) directory.

### State Space
An experiment that simulates the State Space of the model, with phases (Phase 0, EIP1559 enabled, The Merge) representing the upgrade process of the Eth2 system.

* Experiment: [experiments/state_space/experiment.py](experiments/state_space/experiment.py)
* Notebook: [notebooks/state_space.ipynb](notebooks/state_space.ipynb)

![Validator rewards](outputs/validator_rewards.png)

### Revenue Yields vs Network Inflation
Analysing the revenue yields of validators and network inflation for a static ETH price over discrete ETH staked values.

* Experiment: [experiments/revenue_yields_vs_network_inflation/experiment.py](experiments/revenue_yields_vs_network_inflation/experiment.py)
* Notebook: [notebooks/revenue_yields_vs_network_inflation.ipynb](notebooks/revenue_yields_vs_network_inflation.ipynb)

### Validator Environment Yields
Analysing different validator environment yields for static ETH staked over discrete ETH price values.

* Experiment: [experiments/validator_environment_yields/experiment.py](experiments/validator_environment_yields/experiment.py)
* Notebook: [notebooks/validator_environment_yields.ipynb](notebooks/validator_environment_yields.ipynb)

### EIP1559

Analysing the effect of enabling EIP1559 under different conditions.

* Experiment: [experiments/eip1559/experiment.py](experiments/eip1559/experiment.py)
* Notebook: [notebooks/eip1559.ipynb](notebooks/eip1559.ipynb)

### Validation
Various experiments used to validate the results of our model against the Hoban/Borgers Report model.

* Experiments: [experiments/validation/*/experiment.py](experiments/validation/)

### Experiment Execution

The default experiment is an experiment that uses the default cadCAD System Parameters, Initial State, and State Update Blocks defined in the [models/](models/) directory. To run the default experiment from the terminal, execute the `experiments.run` module:
```bash
python3 -m experiments.run
```

Alternatively, open and run one of the analysis Jupyter notebooks in Jupyter Lab.

### Experiment Workflow

1. Create a new directory with the name of the experiment in the [experiments/](experiments/) directory
2. Copy the template experiment from [experiments/template.py](experiments/template.py) into the directory
3. Customize the default experiment using the template
4. Create a new Jupyter notebook using the [notebooks/template.ipynb](notebooks/template.ipynb) experiment notebook template
5. Execute your experiment, post-process and analyze the results, and create Plotly charts!

## Development

A [Makefile](Makefile) is included for convenience, for example to setup your environment and start Jupyter Lab:

```bash
python3 -m venv venv
source venv/bin/activate

make setup # Setup environment
make start # Start Jupyter Lab
```

Otherwise, follow the steps below.

### Requirements

* Python versions: tested with 3.7, 3.8
* Python dependencies: tested against versions in `requirements.txt`

### Setup

To setup a Python 3 development environment:
```bash
# Create a virtual environment using Python 3 venv module
python3 -m venv venv
# Activate virtual environment
source venv/bin/activate
# Install Python 3 dependencies inside virtual environment
pip install -r requirements.txt
```

## Tests

To run the tests using Pytest:
```bash
source venv/bin/activate
python3 -m pytest tests
```

## Jupyter Lab Environment

### Jupyter kernel

To setup your Jupyter Kernel within your virtual environment:
```bash
source venv/bin/activate
python3 -m ipykernel install --user --name python-eth2 --display-name "Python (Eth2)"
```

### Plotly Jupyter Lab support

See https://plotly.com/python/getting-started/

```bash
pip install jupyterlab "ipywidgets>=7.5"
jupyter labextension install jupyterlab-plotly@4.14.3
```

## Roadmap

The following is a non-exhaustive list of possible model extensions and future features:
* Implement a dynamic EIP1559 basefee with a feedback loop based on blockspace demand / network congestion
* Backtest the model against historical data such as the ETH price, ETH staked to determine expected historical yields
* Extend the model to cover future Eth2 phases after merge, such as sharding
* ...

## Change Log

See [CHANGELOG.md](CHANGELOG.md) for notable changes and versions.

## Contributors

See [CONTRIBUTORS.md](CONTRIBUTORS.md) for contributions to this project repo.
