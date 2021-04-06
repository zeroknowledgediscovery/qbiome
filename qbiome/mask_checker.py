import numpy as np
import pandas as pd
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
        returns the predicted df
        """
        masked = self.apply_mask(data, mask_percent)
        for seq in matrix:
            # predict distribs for the entire seq
            distribs = infbiome_qnet.predict_distributions(seq)
            for idx, letter in enumerate(seq):
                col = self.model.feature_names[idx]
                bin_arr = self.quantizervariable_bin_map[col]
                if letter != '': # dequantize
                    seq[idx] = dequantize(letter, bin_arr)
                else: # predict val
                    distrib_dict = distribs[idx]
                    # sample n_samples
                    samples = np.empty(n_samples)
                    for i in range(n_samples):
                        sampled = np.random.choice(
                            list(distrib_dict.keys()),
                            p=list(distrib_dict.values()))
                        samples[i] = dequantize(sampled, bin_arr)
                    seq[idx] = samples.mean()

        filled_df = pd.DataFrame(matrix, dtype=float)
        # add back column names
        filled_df.columns = colnames
        filled_df = pd.concat([pivot_df.subject_id, filled_df], axis=1)
        # melt
        melted_df = filled_df.melt(id_vars='subject_id')
        splitted = melted_df.variable.str.extract(r'([\D|\d]+)_(\d+)', expand=True)
        splitted.rename(columns={0: 'variable', 1: 'week'}, inplace=True)
        melted_df = pd.concat([
            melted_df.subject_id, splitted, melted_df.value
        ], axis=1)
        return melted_df

    def apply_mask(self, data, mask_percent):
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
