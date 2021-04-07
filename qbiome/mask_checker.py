import numpy as np
from itertools import product

from quasinet import qnet

class MaskChecker:

    def __init__(self, qnet_orchestrator):
        """
        qnet_orchestrator: instance of class QnetOrchestrator
        """
        self.qnet_orchestrator = qnet_orchestrator
        self.quantizer = qnet_orchestrator.quantizer

    def mask_and_predict(self, data, mask_percent, n_samples=100):
        """
        data: numpy.ndarray
        n_samples: number of samples for each index qnet is trying to predict, will take the mean

        returns the predicted df in plottable format
        """
        masked = self.apply_random_mask(data, mask_percent)
        predicted_matrix = np.empty(data.shape)
        for idx, seq in enumerate(data):
            # numeric prediction
            predicted_matrix[idx] = self.qnet_orchestrator.predict_sequence(seq)

        df = self.quantizer.add_meta_to_matrix(predicted_matrix)
        # convert to plottable format
        plot_df = self.quantizer.melt_into_plot_format(df)
        return plot_df

    def apply_random_mask(self, data, mask_percent):
        """
        data: numpy.ndarray
        mask_percent: between 0 and 100
        """
        masked = data.copy()
        if not 0 <= mask_percent <= 100:
            raise Exception('Invalid mask percent', mask_percent)
        num_mask = masked.size * mask_percent // 100
        indices = list(product(range(masked.shape[0]), range(masked.shape[1])))
        idx_to_mask = np.random.choice(masked.size, num_mask, replace=False)
        for idx in idx_to_mask:
            row, col = indices[idx]
            masked[row, col] = ''
        return masked
