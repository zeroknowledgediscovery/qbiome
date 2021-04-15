import numpy as np
from itertools import product

from quasinet import qnet

class MaskChecker:
    """For sanity-checking the Quasinet model, randomly mask entries in the original data frame and let the qnet fill in predictions. If the qnet model is good, we expect a minimal amount of difference between the original data and the predicted data
    """

    def __init__(self, qnet_orchestrator):
        """Initialization

        Args:
            qnet_orchestrator (qbiome.QnetOrchestrator): an instance with a trained qnet model
        """
        self.qnet_orchestrator = qnet_orchestrator
        self.quantizer = qnet_orchestrator.quantizer

    def mask_and_predict(self, data, mask_percent, n_samples=100):
        """Mask the data matrix and let qnet fill in the predictions

        Output format:

        |   subject_id | variable         |   week |    value |
        |-------------:|:-----------------|-------:|---------:|
        |            1 | Actinobacteriota |     27 | 0.36665  |
        |            1 | Bacteroidota     |     27 | 0.507248 |
        |            1 | Campilobacterota |     27 | 0.002032 |

        Args:
            data (numpy.ndarray): 2D array of label strings, produced by `self.get_qnet_inputs`
            mask_percent (int): between 0 and 100, the percent of the data matrix to mask
            n_samples (int, optional): the number of times to sample from qnet predictions for one masked entry. Defaults to 100.

        Returns:
            pandas.DataFrame: see format above
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
        """Apply random mask to the data matrix by swapping out entries with empty string `''`

        Args:
            data (numpy.ndarray): 2D label matrix
            mask_percent (int): between 0 and 100

        Raises:
            Exception: mask percent is not between 0 and 100

        Returns:
            numpy.ndarray: 2D label matrix with some entries masked to the empty string
        """
        masked = data.copy()
        if not 0 <= mask_percent <= 100:
            raise Exception('Mask percent', mask_percent, 'is not between 0 and 100')
        num_mask = masked.size * mask_percent // 100
        indices = list(product(range(masked.shape[0]), range(masked.shape[1])))
        idx_to_mask = np.random.choice(masked.size, num_mask, replace=False)
        for idx in idx_to_mask:
            row, col = indices[idx]
            masked[row, col] = ''
        return masked
