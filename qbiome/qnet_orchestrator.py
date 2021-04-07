import numpy as np
from quasinet import qnet

class QnetOrchestrator:

    def __init__(self, quantizer):
        """
        quantizer: instance of class Quantizer
        """
        self.model = None
        self.quantizer = quantizer

    def train_qnet(self, features, data, alpha, min_samples_split, out_fname=None):
        model = qnet.Qnet(feature_names=features, alpha=alpha,
        min_samples_split=min_samples_split, n_jobs=-1)
        model.fit(data)
        if out_fname:
            qnet.save_qnet(model, f=out_fname, low_mem=False)
        self.model = model

    def load_qnet(self, fname):
        self.model = qnet.load_qnet(fname)

    # the following functions can only be called when
    # self.model is not None

    # TODO: refactor the functions below and write up examples, add tqdm

    def predict_value_given_distributions(self, seq, idx, distribs, n_samples=100):
        """
        seq: np.ndarray
        return a numeric prediction
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
        """
        for each entry in the sequence, if it has been masked, i.e. is an empty string ''
        then make prediction, else dequantize the label

        return a predicted numeric sequence, np.ndarray
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

    # sequantial prediction, i.e., the predicted sequence remain in labels

    def predict_sequence_at_week(self, seq, week, n_samples=100):
        """
        predict all {biome}_{week} columns for a given week

        returns a label-string predicted np.ndarray
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
        """
        mask everything after start_week to an empty string
        feed into qnet sequentially

        return a numeric seq
        """
        masked = seq.copy()
        for week in range(start_week, end_week):
            col_indices = np.where(self.model.feature_names.str.contains(str(week)))[0]
            for idx in col_indices:
                masked[idx] = ''
        # feed into qnet sequentially, filling one week every iteration
        for week in range(start_week, end_week + 1):
            masked = self.predict_sequence_at_week(masked, week, n_samples=n_samples)

        # to generate a numeric seq result, dequantize all the labels
        ret = self.quantizer.dequantize_sequence(masked)
        return ret
