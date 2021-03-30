import tensorflow as tf
from tensorflow.keras.constraints import Constraint

class Bounded(Constraint):
    """Constrains weights to be bounded between [a,b] """

    def __init__(self, min=0, max=1):
        self.min = float(min)
        self.max = float(max)
        assert self.min < self.max, "constraint min must be less than max"

    def __call__(self, w):
        w_constrained = tf.clip_by_value(w, self.min, self.max)
        return w_constrained

class Sigmoid(Constraint):
    """Constrains weights to be softly bounded between (a,b) using a sigmoid """

    def __init__(self, min=0, max=1):
        self.min = float(min)
        self.max = float(max)
        assert self.min < self.max, "constraint min must be less than max"

    def __call__(self, w):
        w_constrained = (self.max-self.min) * tf.nn.sigmoid(w) + self.min
        return w_constrained

class ExpMinimum(Constraint):
    """Constrains weights to be softly bounded above threshold (using exp) """

    def __init__(self, min=0):
        self.min = float(min)

    def __call__(self, w):
        w_constrained = tf.exp(w) + self.min
        return w_constrained
