# CostModels

!!! Please READ ME !!!

## Usage

- There are no docs yet; Take a sneek peak into the testing folder for desired model and how to run it.
    - Test for specific model are in the files prefixed with `test_`; For instance, DTU Offshore Cost model tests are in `test_dtu_offshore.py`
    - Go through source code for units and extensive input reference; There is some info on those in the docstrings of classes and functions.
- All percentage inputs are in `0` to `100` range; Be cautions not to input `0.01` thinking that it will result in `1%`; This will evaluate to litaral `0.01%`.

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
