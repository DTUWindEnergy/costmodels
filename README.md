[![pipeline status](https://gitlab.windenergy.dtu.dk/TOPFARM/costmodels/badges/main/pipeline.svg)](https://gitlab.windenergy.dtu.dk/TOPFARM/costmodels/-/commits/main)
[![coverage report](https://gitlab.windenergy.dtu.dk/TOPFARM/costmodels/badges/main/coverage.svg)](https://gitlab.windenergy.dtu.dk/TOPFARM/costmodels/commits/main)
[![PyPi](https://img.shields.io/pypi/v/costmodels)](https://pypi.org/project/costmodels/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17789182.svg)](https://doi.org/10.5281/zenodo.17789182)
# CostModels

CostModels provides a small collection of cost models built using JAX and can provide gradients of finantial metrics with respect to arbitrary design variables. The package is under active development and the API may change without notice.

Available models can be found in `src/costmodels/models` directory. And `examples` folder contains some common use cases of the package.

## Install

- Stable PyPi
```bash
pip install costmodels
```

- Source
```bash
pip install -e .
```

- Development
```bash
pip install -e .[test]
```

## Development with `pixi`

Installation requires `pixi` binary that can be obtained from https://pixi.sh/latest/#installation;

```bash
# development environment install & activation (equivalent to `conda activate`)
pixi shell
# pre-commit formatting hooks (run only once)
pre-commit install
# run tests
pytest
```
