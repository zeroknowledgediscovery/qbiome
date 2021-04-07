import numpy as np

from quasinet import qnet

class Forecaster:
    """
    a sequantial, week-by-week forecaster
    """

    def __init__(self, qnet_orchestrator):
        """
        qnet_orchestrator: instance of class QnetOrchestrator
        """
        self.qnet_orchestrator = qnet_orchestrator
        self.quantizer = qnet_orchestrator.quantizer

    def forecast_data(self, data, start_week, end_week, n_samples=100):
        """
        data: a label matrix

        returns the forecasted df in plottable format
        """
        forecasted_matrix = np.empty(data.shape)
        for idx, seq in enumerate(data):
            forecasted_seq = self.qnet_orchestrator.predict_sequentially_by_week(
                seq, start_week, end_week, n_samples=n_samples
            )
            forecasted_matrix[idx] = forecasted_seq

        df = self.quantizer.add_meta_to_matrix(forecasted_matrix)
        # convert to plottable format
        plot_df = self.quantizer.melt_into_plot_format(df)
        return plot_df
