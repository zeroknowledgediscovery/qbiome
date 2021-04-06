from quasinet import qnet

class QnetGenerator():

    def __init__(self):
        pass

    def train_qnet(self, features, data, alpha, min_samples_split, out_fname=None):
        model = qnet.Qnet(feature_names=features, alpha=alpha,
        min_samples_split=min_samples_split, n_jobs=-1)
        model.fit(data)
        if out_fname:
            qnet.save_qnet(model, f=out_fname, low_mem=False)
        return model

    def load_qnet(self, fname):
        return qnet.load_qnet(fname)
