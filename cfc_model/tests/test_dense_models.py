import math
import random
import unittest

import numpy as np

import cfc_model.dense_models as dense_model
from cfc_model.dense_models import SequentialModel


class TestDenseModel(unittest.TestCase):

    def test_convert_xy_data_fit(self):
        # Predict sine direction, pad_size = 10
        X = np.array([[math.sin(ii + i) for i in range(10)] for ii in range(10)])
        y = np.array([int(math.sin(ii + 11) >= 0) for ii in range(10)]).astype('int')
        dense_model.convert_xy_data_fit(X, y, train_size=0.7)

    def test_convert_xy_data_predict(self):
        # Predict sine direction, pad_size = 10
        X = np.array([[math.sin(ii + i) for i in range(10)] for ii in range(10)])
        dense_model.convert_xy_data_predict(X)

    def test_sine_predict_hardest(self):
        # Predict sine direction, pad_size = 10
        X = np.array(
            [[ii * math.sin((ii + i) / 10) + (random.random() - 0.5) * 20 for i in range(10)] for ii in range(100)])
        y = np.array([int(math.cos((ii + 11) / 10) >= 0) for ii in range(100)])
        model = SequentialModel()
        model.fit(X, y, config={'epochs': 5})

    def test_sine_predict_use_mixed(self):
        # Predict sine direction, pad_size = 10
        X = np.array(
            [[ii * math.sin((ii + i) / 10) + (random.random() - 0.5) * 20 for i in range(10)] for ii in range(100)])
        y = np.array([int(math.cos((ii + 11) / 10) >= 0) for ii in range(100)])
        model = SequentialModel()
        model.fit(X, y, config={'use_mixed': True, 'epochs': 5})

    def test_sine_predict_use_ltc(self):
        # Predict sine direction, pad_size = 10
        X = np.array(
            [[ii * math.sin((ii + i) / 10) + (random.random() - 0.5) * 20 for i in range(10)] for ii in range(100)])
        y = np.array([int(math.cos((ii + 11) / 10) >= 0) for ii in range(100)])
        model = SequentialModel()
        model.fit(X, y, config={'use_ltc': True, 'epochs': 5})

    def test_sine_predict_minimal(self):
        # Predict sine direction, pad_size = 10
        X = np.array(
            [[ii * math.sin((ii + i) / 10) + (random.random() - 0.5) * 20 for i in range(10)] for ii in range(100)])
        y = np.array([int(math.cos((ii + 11) / 10) >= 0) for ii in range(100)])
        model = SequentialModel()
        model.fit(X, y, config={'minimal': True, 'epochs': 5})

    def test_sine_predict_no_gate(self):
        # Predict sine direction, pad_size = 10
        X = np.array(
            [[ii * math.sin((ii + i) / 10) + (random.random() - 0.5) * 20 for i in range(10)] for ii in range(100)])
        y = np.array([int(math.cos((ii + 11) / 10) >= 0) for ii in range(100)])
        model = SequentialModel()
        model.fit(X, y, config={'no_gate': True, 'epochs': 5})

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
            "batch_size": len(X)
        }

        model = SequentialModel()
        model.fit(X, y, config=config)

        # Predict out of sample
        assert model.predict([1, 1, 0, 1]) == 0
        assert model.predict([1, 0, 0, 1]) == 1
