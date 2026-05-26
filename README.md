# DualCast

This repository contains the official Python implementation for running experimental demos of DualCast.

## Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management. Follow the steps below to set up your environment:

### Clone the repository
```sh
$ git clone https://github.com/KKKTa/DualCast.git
```

### Poetry install (optional)
```sh
# Linux, macOS, Windows (WSL)
$ curl -sSL https://install.python-poetry.org | python3 -

# Windows (Powershell)
$ (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
```

### Install dependencies
```sh
$ poetry install
```

## Quick Demo
```sh
$ sh bin/run.sh <dataset_name> (e.g., programming)
```
The execution logs are saved in `logs` directory, and the results are in `_results` directory.