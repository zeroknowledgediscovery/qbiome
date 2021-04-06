import numpy as np
import pandas as pd
from quasinet import qnet

class MaskChecker:

    def __init__(self):
        pass

    def mask_and_predict(self, qnet, data, mask_percent, n_samples=100):
        """
        n_samples: number of samples for each index qnet is trying to predict, will take the mean
        returns the predicted df
        """
