import numpy as np

class MissingDataError(BaseException):
    """Raised if data is expected to create GenericData instance, but none is given."""

class GenericData:

    def __init__(self,
                 pad_size = 256):
        self.pad_size = pad_size
        self.train_elapsed = []
        self.train_events = []
        self.train_mask = []
        self.train_y = []

        self.test_elapsed = []
        self.test_events = []
        self.test_mask = []
        self.test_y = []