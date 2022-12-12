import unittest
import CFC.dense_model as dense_model
import numpy as np

class TestDenseModel(unittest.TestCase):

    def test_load_data(self):

        # Assumes an event size of 4, and a time
        # Label: Do the events sum to 3?
        # Only dense data
        class GenericData:

            def __init__(self):
                self.pad_size = 4
                self.train_elapsed = np.array([[.25, .5, .75, 1],
                                               [.25, .5, .75, 1],
                                               [.25, .5, .75, 1],
                                               [.25, .5, .75, 1]])
                self.train_events = np.array([[1, 1, 1, 0],
                                              [1, 1, 0, 1],
                                              [1, 0, 0, 1],
                                              [1, 1, 0, 0]])
                self.train_mask = np.array([[True, True, True, True],
                                            [True, True, True, True],
                                            [True, True, True, True],
                                            [True, True, True, True]])
                self.train_y = np.array([3, 3, 2, 2])

                self.test_elapsed = np.array([[.25, .5, .75, 1],
                                              [.25, .5, .75, 1],
                                              [.25, .5, .75, 1],
                                              [.25, .5, .75, 1]])
                self.test_events = np.array([[1, 0, 1, 0],
                                             [1, 1, 0, 1],
                                             [1, 0, 0, 1],
                                             [1, 0, 1, 0]])
                self.test_mask = np.array([[True, True, True, True],
                                           [True, True, True, True],
                                           [True, True, True, True],
                                           [True, True, True, True]])
                self.test_y = np.array([3, 3, 2, 2])

        # CFC Config
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

        data = GenericData()
        model = dense_model.fit(data, config)