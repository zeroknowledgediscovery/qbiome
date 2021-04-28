import os
import numpy as np
from quasinet import qnet

class QnetOrchestrator:
    """Manages utilities related to the Quasinet model, for example, training, saving, loading, and different methods of predicting
    """

    def __init__(self, quantizer):
        """Initialization

        Args:
            quantizer (qbiome.Quantizer): an instance with populated quantization map and other states
        """
        self.model = None
        """qnet model"""

        self.quantizer = quantizer

    def train_qnet(self, features, data, alpha, min_samples_split, out_fname=None):
        """Train the qnet model. If `out_fname` is present, also saves the model. The inputs `features, data` are produced by `Quantizer.get_qnet_inputs`. See [Quasinet documentations](https://zeroknowledgediscovery.github.io/quasinet/build/html/quasinet.html#module-quasinet.qnet) for the other parameters

        Args:
            features (list): list: a list of feature names, ex. `['Acidobacteriota_35', 'Actinobacteriota_1', 'Actinobacteriota_2']`
            data (numpy.ndarray): 2D matrix of label strings
            alpha (float): threshold value for selecting feature with permutation tests. Smaller values correspond to shallower trees
            min_samples_split (int): minimum samples required for a split
            out_fname (str, optional): save file name. Defaults to None.
        """
        self.model = qnet.Qnet(feature_names=features, alpha=alpha,
        min_samples_split=min_samples_split, n_jobs=-1)
        self.model.fit(data)
        if out_fname:
            self.save_qnet(out_fname)

    def load_qnet(self, in_fname):
        """Load `self.model` from file

        Args:
            in_fname (str): input file containing a saved qnet model
        """
        self.model = qnet.load_qnet(in_fname)

    def save_qnet(self, out_fname):
        """Save `self.model` to file

        Args:
            out_fname (str): save file name
        """
        assert self.model is not None
        qnet.save_qnet(self.model, f=out_fname, low_mem=False)

    def export_qnet_tree_dotfiles(self, out_dirname):
        """Generate tree dotfiles for each feature of the model

        Args:
            out_dirname (str): the output directory, make one if doesn't exist
        """
        assert self.model is not None
        if not os.path.exists(out_dirname):
            os.mkdir(out_dirname)
        for idx, feature_name in enumerate(self.model.feature_names):
            qnet.export_qnet_tree(self.model, idx,
            os.path.join(out_dirname, '{}.dot'.format(feature_name)),
            outformat='graphviz', detailed_output=True)

    # the following functions can only be called when
    # self.model is not None

    # TODO: add tqdm for progress tracking

    def predict_value_given_distributions(self, seq, idx, distribs, n_samples=100):
        """Predict a numeric value for the specified index of the label sequence, given the label distributions generated by the qnet. Sample `n_samples` times from the predictions, dequantize the sampled labels and take average

        Args:
            seq (numpy.ndarray): 1D array of label strings
            idx (int): index into the input `seq`
            distribs (list): Produced by `quasinet.qnet.Qnet.predict_distributions(seq)`. See [Quasinet documentations](https://zeroknowledgediscovery.github.io/quasinet/build/html/quasinet.html#quasinet.qnet.Qnet.predict_distributions)
            n_samples (int, optional): the number of times to sample from qnet predictions for one masked entry. Defaults to 100.

        Returns:
            float: predicted numeric value
        """
        distrib_dict = distribs[idx]
        bin_arr = self.quantizer.get_bin_array_of_index(idx)
        # sample n_samples
        samples = np.empty(n_samples)
        for i in range(n_samples):
            sampled = np.random.choice(
                list(distrib_dict.keys()),
                p=list(distrib_dict.values()))
            samples[i] = self.quantizer.dequantize_label(sampled, bin_arr)
        ret = samples.mean()
        return ret

    def predict_sequence(self, seq, indices_to_predict=None, n_samples=100):
        """Convert the label sequence into a numeric one by filling qnet predictions for masked entries (represented as an empty string) or simply dequantizing the non-masked entries

        Args:
            seq (numpy.ndarray): 1D array of label strings
            indices_to_predict (list, optional): a list of indices at which masks have been applied, for which we need to make qnet predictions. Defaults to None.
            n_samples (int, optional): the number of times to sample from qnet predictions for one masked entry. Defaults to 100.

        Returns:
            numpy.ndarray: 1D array of floats
        """
        predicted = np.empty(seq.shape)
        distribs = self.model.predict_distributions(seq)
        if not indices_to_predict: # predict everything in the sequence
            indices_to_predict = range(len(seq))
        for idx in indices_to_predict:
            label = seq[idx]
            if label == '': # this is masked, predict
                num = self.predict_value_given_distributions(seq, idx, distribs, n_samples=n_samples)
            else: # not masked, simpily dequantize
                bin_arr = self.quantizer.get_bin_array_of_index(idx)
                num = self.quantizer.dequantize_label(label, bin_arr)
            predicted[idx] = num
        return predicted

    # sequantial prediction, i.e., the predicted sequence remain in labels for the iterative process

    def predict_sequence_at_week(self, seq, week, n_samples=100):
        """For a given week, predict all `{biome}_{week}` columns. Note that the return array consists of label strings instead of floats, as it is just an intermediate state and will be used for sequential prediction.

        Args:
            seq (numpy.ndarray): 1D array of label strings
            week (int): the week number
            n_samples (int, optional): the number of times to sample from qnet predictions for one masked entry. Defaults to 100.

        Returns:
            numpy.ndarray: 1D array of label strings
        """
        predicted = seq.copy()
        distribs = self.model.predict_distributions(seq)
        col_indices = np.where(self.model.feature_names.str.contains(str(week)))[0]
        for idx in col_indices:
            # predict
            num = self.predict_value_given_distributions(seq, idx, distribs, n_samples=n_samples)
            # re-quantize qnet-predicted numeric values
            bin_arr = self.quantizer.get_bin_array_of_index(idx)
            label = self.quantizer.quantize_value(num, bin_arr)
            # fill the spot in masked for sequential feeding into qnet
            predicted[idx] = label
        return predicted

    def predict_sequentially_by_week(self, seq, start_week, end_week, n_samples=100):
        """Use qnet to generate sequential, iterative prediction of the sequence from `start_week` to `end_week`. This is accomplished by masking the current week to predict, use the qnet to predict a label for this masked entry (after which the qnet can update its prediction for the label distributions), masking the next week, and repeat.

        Args:
            seq (numpy.ndarray): 1D array of label strings
            start_week (int): start predicting from this week
            end_week (int): end predicting after this week
            n_samples (int, optional): the number of times to sample from qnet predictions for one masked entry. Defaults to 100.

        Returns:
            numpy.ndarray: 1D array of floats
        """
        masked = seq.copy()
        for week in range(start_week, end_week + 1):
            col_indices = np.where(self.model.feature_names.str.contains(str(week)))[0]
            for idx in col_indices:
                masked[idx] = ''
        # feed into qnet sequentially, filling one week every iteration
        for week in range(start_week, end_week + 1):
            masked = self.predict_sequence_at_week(masked, week, n_samples=n_samples)

        # to generate a numeric seq result, dequantize all the labels
        ret = self.quantizer.dequantize_sequence(masked)
        return ret
