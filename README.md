# CostModels

!!! Please READ ME !!!

## Install

```bash
git clone git@gitlab.windenergy.dtu.dk:TOPFARM/costmodels.git
pip install -e costmodels/
```

## Usage

- The package is under heavy development; No stability is to be expected! It will change abruply and break more often than not, thus please regularly update the installation and follow updates in the `README.md` file. And report any bugs you find : )
- There are no docs yet; Take a sneek peak into the testing folder for desired model and how to run it.
    - Test for specific model are in the files prefixed with `test_`; For instance, DTU Offshore Cost model tests are in `test_dtu_offshore.py`
    - Go through source code for units and extensive input reference; There is some info on those in the docstrings of classes and functions.
- All percentage inputs are in `0` to `100` range; Be cautions not to input `0.01` thinking that it will result in `1%`; This will evaluate to litaral `0.01%`.

### Models

Available models:
 - DTUOffshoreCostModel
 - MinimalisticCostModel
 - NRELCostModel
 - PVCostModel

You can import them simply by, for instance, `from costmodels import PVCostModel`. For input specification please look at the `_cm_input_def` method in the respective model. Please find the notebooks in the examples folder to see the usage of each of them.

### Implementing new model

There are two interface methods to implement in the child class extending the parent. `_cm_input_def` simply defines a dictionary of inputs that will be assigned to class attributes with the dictionary key being the attribute name in the class. The values of that dictionary will be used for specification of default values and units of the physical quantity, if default is not applicable you can use `np.nan` for the value and deal with the input validation in the `_run` method. The later will be used to compute a cost model output. A full example: 

```python
from costmodels.base import CostModel
from costmodels.units import Quant

class MyCostModel(CostModel):

    @property
    def _cm_input_def(self) -> dict:
        """Definition of cost model input."""
        return {
            "nwt": 0,
            "turbine_cost": Quant(0, "EUR"),
        }

    def _run(self) -> dict:
        """Method to run the cost model."""
        assert self.nwt > 0
        assert self.turbine_cost > 0
        CAPEX = self.nwt * self.turbine_cost
        return {"CAPEX": CAPEX}

if __name__ == "__main__":
    cm = MyCostModel( # static model variables
        nwt=1000,
        turbine_cost=Quant(1e6, "EUR"),
    )
    print(cm.run()["CAPEX"])
    # change number of turbines on run call; accomodate dynamic values
    print(cm.run(nwt=500)["CAPEX"])
```

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
