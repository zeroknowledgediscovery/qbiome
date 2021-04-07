import numpy as np

from quasinet import qnet

class Forcaster:
    """
    a sequantial, week-by-week forcaster
    """

    def __init__(self, qnet_orchestrator):
        """
        qnet_orchestrator: instance of class QnetOrchestrator
        """
        self.qnet_orchestrator = qnet_orchestrator
        self.quantizer = qnet_orchestrator.quantizer

    def forcast_data(self, data, start_week, end_week, n_samples=100):
        """
        data: a label matrix

        returns the forcasted df in plottable format
        """
        forcasted_matrix = np.empty(data.shape)
        for idx, seq in enumerate(data):
            forcasted_seq = self.qnet_orchestrator.predict_sequentially_by_week(
                seq, start_week, end_week, n_samples=n_samples
            )
            forcasted_matrix[idx] = forcasted_seq

        df = self.quantizer.add_meta_to_matrix(forcasted_matrix)
        # convert to plottable format
        plot_df = self.quantizer.melt_into_plot_format(df)
        return plot_df
