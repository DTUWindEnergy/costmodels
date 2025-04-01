# CostModels

!!! Please READ ME !!!

## Install

```bash
git clone git@gitlab.windenergy.dtu.dk:TOPFARM/costmodels.git
pip install -e costmodels/
```

## Usage

### Notice

- The package is under heavy development; No stability is to be expected! It will change abruply and break more often than not, thus please regularly update the installation and follow updates in the `README.md` file. And report any bugs you find : )
- All percentage inputs are in `0` to `100` range; Be cautions not to input `0.01` thinking that it will result in `1%`; This will evaluate to litaral `0.01%`.

### Models

Available models:
 - DTUOffshoreCostModel
 - MinimalisticCostModel
 - NRELCostModel
 - PVCostModel

You can import them simply by, for instance, `from costmodels import PVCostModel`. For input specification please look at the `_cm_input_def` method in the respective model. Notebooks in the examples folder includes usage samples of each of them.

### Examples

- [Basic usage](examples/usage.ipynb)
- [Custom cost models](examples/custom.ipynb)
- [Cost optimization with PyWake windfarm simulation](examples/optimization.ipynb)


## Development 

Installation requires `pixi` https://pixi.sh/latest/#installation;

```bash
# development environment install & activation (equivalent to `conda activate`)
pixi shell -e test
# pre-commit formatting hooks (run only once)
pre-commit install
# run tests
pytest 
```
