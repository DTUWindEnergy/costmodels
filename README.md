# CostModels

CostModels provides a small collection of cost models built using JAX and can provide gradients of finantial metrics with respect to arbitrary design variables. The package is under active development and the API may change without notice.

Available models can be found in `src/costmodels/models` directory. And `examples` folder contains some commong use cases of the package.

## Install

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
