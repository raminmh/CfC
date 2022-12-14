![PyPI 📦   ](https://github.com/nightvision04/CfC/workflows/PyPI%20%F0%9F%93%A6%20%20%20/badge.svg?branch=main)
![PyTests](https://github.com/nightvision04/CfC/workflows/PyTests/badge.svg?branch=main)
[![Downloads](https://pepy.tech/badge/cfc-model)](https://pepy.tech/project/cfc-model)


# Closed-form Continuous-time Models

Closed-form Continuous-time Neural Networks (CfCs) are powerful sequential neural information processing units. 

Paper Open Access: https://www.nature.com/articles/s42256-022-00556-7

Arxiv: https://arxiv.org/abs/2106.13898

## Installation

```
pip install cfc-model
```

## Requirements

- Python 3.7 or newer
- Tensorflow 2.4 or newer
- Pandas
- Numpy

For a fresh anaconda environment with the required dependencies:
```
conda env create --file environment.yml
conda activate cfc
```

## Usage

### Example
```
from cfc_model.dense_models import SequentialModel
X = np.array([[1, 1, 1, 0], [1, 1, 0, 1], [1, 0, 0, 1], [1, 1, 0, 0],
              [1, 0, 1, 0], [1, 1, 0, 1], [1, 0, 0, 1], [1, 0, 1, 0]])
y = np.array([0, 0, 1, 1, 1, 0, 1, 1])
model = SequentialModel()
model.fit(X, y)
y_pred = model.predict([1, 1, 0, 1]) # y_pred equals 0
```

The following configuration states can be used

- ```no_gate``` Runs the CfC without the (1-sigmoid) part
- ```minimal``` Runs the CfC direct solution
- ```use_ltc``` Runs an LTC with a semi-implicit ODE solver instead of a CfC
- ```use_mixed``` Mixes the CfC's RNN-state with a LSTM to avoid vanishing gradients

If none of these flags are provided, the full CfC model is used

## Example

```

from cfc_model.dense_models import SequentialModel
X = np.array([[1, 1, 1, 0], [1, 1, 0, 1], [1, 0, 0, 1], [1, 1, 0, 0],
              [1, 0, 1, 0], [1, 1, 0, 1], [1, 0, 0, 1], [1, 0, 1, 0]])
y = np.array([0, 0, 1, 1, 1, 0, 1, 1])
model = SequentialModel()

# Runs an LTC with a semi-implicit ODE solver instead of a CfC
config = {"use_ltc": True}
model.fit(X, y, config=config)
y_pred = model.predict([1, 1, 0, 1]) # y_pred equals 0
```



## Cite

```@article{hasani_closed-form_2022,
	title = {Closed-form continuous-time neural networks},
	journal = {Nature Machine Intelligence},
	author = {Hasani, Ramin and Lechner, Mathias and Amini, Alexander and Liebenwein, Lucas and Ray, Aaron and Tschaikowski, Max and Teschl, Gerald and Rus, Daniela},
  issn = {2522-5839},
	month = nov,
	year = {2022},
}
