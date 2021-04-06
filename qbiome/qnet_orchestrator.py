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

    # TODO: refactor the three functions below and write up examples

    def predict_value_at_index(self, seq, idx, n_samples=100):
        """
        seq is an np.ndarray
        return a numeric prediction
        """
        distribs = self.model.predict_distributions(seq)
        letter = seq[idx]
        col = self.model.feature_names[idx]
        bin_arr = self.quantizer.variable_bin_map[col]
        # predict val
        distrib_dict = distribs[idx]
        # sample n_samples
        samples = np.empty(n_samples)
        for i in range(n_samples):
            sampled = np.random.choice(
                list(distrib_dict.keys()),
                p=list(distrib_dict.values()))
            samples[i] = self.quantizer.dequantize_label(sampled, bin_arr)
        ret = samples.mean()
        return ret

    def predict_sequence_at_week(self, seq, week, n_samples=100):
        predicted = seq.copy()
        col_indices = np.where(self.model.feature_names.str.contains(str(week)))[0]
        for idx in col_indices:
            num = qnet_predict_num_at_idx(seq, idx, n_samples=n_samples)
            # quantize qnet-predicted numeric values
            col = self.model.feature_names[idx]
            bin_arr = self.quantizer.variable_bin_map[col]
            letter = pd.cut([num], bin_arr, labels=list(self.quantizer.labels.keys()))[0]
            # fill the spot in masked for sequential feeding into qnet
            predicted[idx] = letter
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

        for week in range(start_week, end_week):
            masked = qnet_predict_seq_at_week(masked, week, n_samples=100)

        # to generate a numeric seq result, dequantize all the letters
        num_ret = np.empty(masked.shape)
        distribs = infbiome_qnet.predict_distributions(masked)
        for idx, letter in enumerate(masked):
            col = self.model.feature_names[idx]
            bin_arr = self.quantizer.variable_bin_map[col]
            num_ret[idx] = self.quantizer.dequantize_label(letter, bin_arr)

        return num_ret
