import unittest
from cfc_model.data_types import GenericData
import cfc_model.dense_model as dense_model
import numpy as np
import math
import random

class TestDenseModel(unittest.TestCase):

    def test_convert_xy_data(self):

        # Predict sine direction, pad_size = 10
        X = np.array([[math.sin(ii+i) for i in range(10)] for ii in range(10)])
        y = np.array([int(math.sin(ii+11) >= 0) for ii in range(10)]).astype('int')
        data = dense_model.convert_xy_data(X, y, train_size=0.7)
        data

    def test_sine_predict(self):
        # Predict sine direction, pad_size = 10
        X = np.array([[math.sin(ii + i) for i in range(10)] for ii in range(10)])
        y = np.array([int(math.sin(ii+11) >= 0) for ii in range(10)])
        model = dense_model.fit(X, y)

    def test_sine_predict_hard(self):
        # Predict sine direction, pad_size = 10
        X = np.array([[math.sin(ii + i) for i in range(10)] for ii in range(10)])
        y = np.array([int(math.cos(ii+11) >= 0) for ii in range(10)])
        model = dense_model.fit(X, y)

    def test_sine_predict_hardest(self):
        # Predict sine direction, pad_size = 10
        X = np.array(
            [[ii * math.sin((ii + i) / 10) + (random.random() - 0.5) * 20 for i in range(10)] for ii in range(100)])
        y = np.array([int(math.cos((ii + 11) / 10) >= 0) for ii in range(100)])
        model = dense_model.fit(X, y)


    def test_load_data(self):

        # Assumes an event size of 4, and a time
        # Label: Do the events sum to 3?
        # Only dense data
        data = GenericData()

        data.pad_size = 4
        data.train_elapsed = np.array([[.25, .5, .75, 1],
                                       [.25, .5, .75, 1],
                                       [.25, .5, .75, 1],
                                       [.25, .5, .75, 1]])
        data.train_events = np.array([[1, 1, 1, 0],
                                      [1, 1, 0, 1],
                                      [1, 0, 0, 1],
                                      [1, 1, 0, 0]])
        data.train_mask = np.array([[True, True, True, True],
                                    [True, True, True, True],
                                    [True, True, True, True],
                                    [True, True, True, True]])
        data.train_y = np.array([3, 3, 2, 2])

        data.test_elapsed = np.array([[.25, .5, .75, 1],
                                      [.25, .5, .75, 1],
                                      [.25, .5, .75, 1],
                                      [.25, .5, .75, 1]])
        data.test_events = np.array([[1, 0, 1, 0],
                                     [1, 1, 0, 1],
                                     [1, 0, 0, 1],
                                     [1, 0, 1, 0]])
        data.test_mask = np.array([[True, True, True, True],
                                   [True, True, True, True],
                                   [True, True, True, True],
                                   [True, True, True, True]])
        data.test_y = np.array([3, 3, 2, 2])

        # cfc_model Config
        config = {
            "backbone_activation": "gelu",
            "backbone_dr": 0.0,
            "forget_bias": 3.0,
            "backbone_units": 128,
            "backbone_layers": 1,
            "weight_decay": 0,
            "use_lstm": False,
            "no_gate": False,
            "minimal": False,
        }

        model = dense_model.fit(data=data, config=config)