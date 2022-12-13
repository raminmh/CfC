import unittest
from cfc_model.data_types import GenericData
import cfc_model.dense_models as dense_model
from cfc_model.dense_models import SequentialModel
import numpy as np
import math
import random

class TestDenseModel(unittest.TestCase):

    def test_convert_xy_data_fit(self):
        # Predict sine direction, pad_size = 10
        X = np.array([[math.sin(ii+i) for i in range(10)] for ii in range(10)])
        y = np.array([int(math.sin(ii+11) >= 0) for ii in range(10)]).astype('int')
        dense_model.convert_xy_data_fit(X, y, train_size=0.7)

    def test_convert_xy_data_fit(self):
        # Predict sine direction, pad_size = 10
        X = np.array([[math.sin(ii+i) for i in range(10)] for ii in range(10)])
        data = dense_model.convert_xy_data_predict(X)
        data

    def test_sine_predict_hardest(self):
        # Predict sine direction, pad_size = 10
        X = np.array(
            [[ii * math.sin((ii + i) / 10) + (random.random() - 0.5) * 20 for i in range(10)] for ii in range(100)])
        y = np.array([int(math.cos((ii + 11) / 10) >= 0) for ii in range(100)])
        model = SequentialModel()
        model.fit(X, y)

    def test_fit_predict(self):

        # Problem: Do the events sum to 2?
        X = np.array([[1, 1, 1, 0], [1, 1, 0, 1], [1, 0, 0, 1], [1, 1, 0, 0],
                      [1, 0, 1, 0], [1, 1, 0, 1], [1, 0, 0, 1], [1, 0, 1, 0]])
        y = np.array([0, 0, 1, 1, 1, 0, 1, 1])

        # cfc_model Config
        config = {
            "backbone_activation": "relu",
            "backbone_dr": 0.0,
            "forget_bias": 3.0,
            "backbone_units": 128,
            "backbone_layers": 1,
            "weight_decay": 0,
            "use_lstm": False,
            "no_gate": False,
            "minimal": False,
        }

        model = SequentialModel()
        model.fit(X, y, config=config)

        # Predict out of sample
        assert model.predict([1, 1, 0, 1]) == 0
        assert model.predict([1, 0, 0, 1]) == 1
